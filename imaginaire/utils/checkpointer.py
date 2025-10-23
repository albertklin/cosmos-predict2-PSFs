# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from imaginaire.model import ImaginaireModel
from imaginaire.utils import callback, distributed, log, misc
from imaginaire.utils.parallelism import ModelWrapper

if TYPE_CHECKING:
    from imaginaire.config import CheckpointConfig, JobConfig


class Checkpointer:
    """The checkpointer class. Supports checkpoint saving/loading to local disk."""

    def __init__(self, config_checkpoint: CheckpointConfig, config_job: JobConfig, callbacks: callback.CallBackGroup):
        """Constructor of the checkpointer.

        Args:
            config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        """
        # Set the callback functions.
        self.callbacks = callbacks
        self.checkpoint_dir_local = f"{config_job.path_local}/checkpoints"
        self.strict_resume = config_checkpoint.strict_resume
        self.load_path = config_checkpoint.load_path or None
        self.load_training_state = config_checkpoint.load_training_state
        self.only_load_scheduler_state = config_checkpoint.only_load_scheduler_state
        self.save_thread = None

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}.pt"

        if distributed.get_rank() == 0:
            # Prepare model state dict
            full_model_sd = model.state_dict()

            # Generic space-efficient split: save frozen weights once and per-iter trainable/mutable deltas.
            # Trainable keys are detected from optimizer param groups with robust name normalization.
            try:
                # Build mapping from Parameter id to its fully-qualified name
                id_to_name = {id(p): n for n, p in model.named_parameters(recurse=True)}

                # Normalization helper to align named_parameters() and state_dict() keys
                def normalize_name(name: str) -> str:
                    if name.startswith("net."):
                        return name.split(".", 1)[1]
                    if name.startswith("net_ema."):
                        return name.split(".", 1)[1]
                    if ".dit." in name:
                        return name.split(".dit.", 1)[1]
                    if ".dit_ema." in name:
                        return name.split(".dit_ema.", 1)[1]
                    return name.split(".", 1)[1] if "." in name else name

                trainable_suffixes = set()
                raw_trainable_names = set()
                for pg in optimizer.param_groups:
                    for p in pg.get("params", []):
                        raw = id_to_name.get(id(p))
                        if raw is None:
                            continue
                        raw_trainable_names.add(raw)
                        trainable_suffixes.add(normalize_name(raw))

                def is_trainable_key(k: str) -> bool:
                    k_norm = normalize_name(k)
                    if k_norm in trainable_suffixes:
                        return True
                    if "ema" in k.lower():
                        return k_norm in trainable_suffixes
                    return False

                frozen_sd = {}
                trainable_sd = {}
                for k, v in full_model_sd.items():
                    if is_trainable_key(k):
                        trainable_sd[k] = v
                    else:
                        frozen_sd[k] = v

                # Sanity-logging: parameter counts (to match training logs) and tensor counts per split
                all_param_names = {n for n, _ in model.named_parameters(recurse=True)}
                trainable_param_names = {n for n in all_param_names if n in raw_trainable_names}
                frozen_param_names = all_param_names - trainable_param_names
                trainable_param_count = sum(
                    p.numel() for n, p in model.named_parameters(recurse=True) if n in trainable_param_names
                )
                frozen_param_count = sum(
                    p.numel() for n, p in model.named_parameters(recurse=True) if n in frozen_param_names
                )
                log.info(
                    f"Checkpoint split -> frozen params: {frozen_param_count:,} | trainable params: {trainable_param_count:,}"
                )
                log.info(
                    f"Saving-once tensors (frozen): {len(frozen_sd)} | per-iter tensors (trainable): {len(trainable_sd)}"
                )

                # Downcast model tensors to BF16 on disk to avoid FP32 load-time spikes.
                def cast_fp_to_bf16(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                    out: dict[str, torch.Tensor] = {}
                    for kk, vv in d.items():
                        if isinstance(vv, torch.Tensor) and vv.is_floating_point():
                            out[kk] = vv.to(torch.bfloat16)
                        else:
                            out[kk] = vv
                    return out

                frozen_sd = cast_fp_to_bf16(frozen_sd)
                trainable_sd = cast_fp_to_bf16(trainable_sd)

                # Save the frozen weights once per job as checkpoints/frozen.pt (do not touch latest_checkpoint.txt).
                frozen_path = os.path.join(self.checkpoint_dir_local, "frozen.pt")
                if not os.path.exists(frozen_path) and len(frozen_sd) > 0:
                    try:
                        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
                        torch.save(misc.to(frozen_sd, device="cpu"), frozen_path)
                        log.success(f"Saved frozen checkpoint (local): {frozen_path}")
                    except Exception as e:
                        log.exception(f"Failed to save frozen checkpoint (local): {e}")

                # If a meaningful split exists, only keep trainable tensors per-iteration
                model_state_for_ckpt = trainable_sd if (len(trainable_sd) > 0 and len(frozen_sd) > 0) else full_model_sd
            except Exception as e:
                log.exception(f"Failed to split trainable/frozen for checkpointing; falling back to full save: {e}")
                model_state_for_ckpt = full_model_sd

            state_dict = dict(
                model=misc.to(model_state_for_ckpt, device="cpu"),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            state_dict = misc.to(state_dict, device="cpu")
            self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
            # Wait for previous saver thread to end.
            if self.save_thread:
                self.save_thread.join()
            # Run the checkpoint saver in a separate thread.
            self.save_thread = threading.Thread(
                target=self._save_worker_local,
                daemon=False,
                args=(state_dict, checkpoint_file, distributed.get_rank()),
            )
            self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    @misc.timer("checkpoint saving (local)")
    def _save_worker_local(self, state_dict: dict[str, torch.Tensor], checkpoint_file: str, rank: int = 0) -> None:
        """Worker to save checkpoint to local disk, spawned with a child thread (runs in parallel with the training).

        Args:
            state_dict (dict[str, torch.Tensor]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir_local, checkpoint_file)
        os.makedirs(self.checkpoint_dir_local, exist_ok=True)
        try:
            torch.save(state_dict, checkpoint_path)
            if rank == 0:
                self._write_latest_checkpoint_file(checkpoint_file)
            log.success(f"Saved checkpoint (local): {checkpoint_path}")
            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:
            log.exception(f"Checkpoint failed to save (local): {e}")

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = self.checkpoint_dir_local
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
            only_resume_scheduler = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                resume = self.load_training_state
                only_resume_scheduler = self.only_load_scheduler_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
                only_resume_scheduler = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            log.info(f"Loading checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            log.info("- Loading the model...")
            model_state_dict = state_dict["model"]
            # If a frozen checkpoint exists, merge it with the current (likely trainable-only) checkpoint
            frozen_path = os.path.join(self.checkpoint_dir_local, "frozen.pt")
            if os.path.exists(frozen_path):
                try:
                    base_sd = torch.load(frozen_path, map_location=lambda storage, loc: storage)
                    merged_sd = dict(base_sd)
                    merged_sd.update(model_state_dict)
                    model_state_dict = merged_sd
                    log.info("Merged checkpoint with frozen weights")
                except Exception as e:
                    log.exception(f"Failed to merge frozen checkpoint: {e}")

            model.load_state_dict(model_state_dict, strict=self.strict_resume)
            if resume or only_resume_scheduler:
                iteration = state_dict["iteration"]
                assert scheduler
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
            else:
                iteration = 0
            if resume:
                assert optimizer
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                log.success(f"Done with loading the checkpoint (iteration {iteration}).")
            else:
                log.success("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)

        return iteration

    def _read_latest_checkpoint_file(self) -> str | None:
        """Get the file name of the latest saved checkpoint. If it doesn't exist, return None.

        Returns:
            checkpoint_file (str | None): file name of the latest saved checkpoint.
        """
        checkpoint_file = None
        latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
        if os.path.isfile(latest_path):
            checkpoint_file = open(latest_path).read().strip()
        return checkpoint_file

    def _write_latest_checkpoint_file(self, checkpoint_file: str) -> None:
        """Track the file name of the latest saved checkpoint.

        Args:
            checkpoint_file (str): file name of the latest saved checkpoint.
        """
        content = f"{checkpoint_file}\n"
        latest_path = os.path.join(self.checkpoint_dir_local, "latest_checkpoint.txt")
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path: str) -> None:
        """If the file checkpoint_path does not exist, raise an error.

        Args:
            checkpoint_path (str): full path to the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"File not found (local): {checkpoint_path}")

    def finalize(self) -> None:
        """Finalize the checkpointer."""
        if self.save_thread:
            self.save_thread.join()


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", list[str]),
            ("unexpected_keys", list[str]),
            ("incorrect_shapes", list[tuple[str, tuple[int], tuple[int]]]),
        ],
    )
):
    pass


def load_checkpoint(
    model_parts: list[nn.Module],
    ckpt_dir,
    model_ckpt_key_map: dict[str, str] = {},  # noqa: B006
):
    log.info(f"Loading checkpoint from {ckpt_dir}.")

    _model_wrapper = ModelWrapper(model_parts)
    state_dict = _model_wrapper.state_dict()
    # remove _extra_state
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("._extra_state")}

    # remap keys if needed
    if model_ckpt_key_map:
        for model_key, checkpoint_key in model_ckpt_key_map.items():
            state_dict[checkpoint_key] = state_dict.pop(model_key)
            log.info(f"Re-mapping {model_key} to {checkpoint_key}")

    fs_storage_reader = dist.checkpoint.FileSystemReader(ckpt_dir)
    dist.checkpoint.load(state_dict=state_dict, storage_reader=fs_storage_reader)

    # inverse the remapping if needed
    if model_ckpt_key_map:
        for model_key, checkpoint_key in model_ckpt_key_map.items():
            state_dict[model_key] = state_dict.pop(checkpoint_key)
            log.info(f"Inverse re-mapping {checkpoint_key} to {model_key}")

    _model_wrapper.load_state_dict(state_dict)

    log.info(f"Finished loading checkpoint from {ckpt_dir}.")
