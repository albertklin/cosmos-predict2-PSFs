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

import math

import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh

from cosmos_predict2.models.utils import load_state_dict
from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from cosmos_predict2.utils.optim_instantiate import get_base_scheduler
from imaginaire.lazy_config import instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log


class Predict2Video2WorldActionConditionedModel(Predict2Video2WorldModel):
    def __init__(self, config: Predict2Video2WorldModelConfig):
        super(ImaginaireModel, self).__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # NOTE: replace the pipeline with action-conditioned setup
        self.pipe = Video2WorldActionConditionedPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
        )

        self.freeze_parameters()
        self._enable_action_heads_training()
        lora_reloaded_from_checkpoint = False
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
            lora_reloaded_from_checkpoint = self._maybe_reload_lora_from_checkpoint()
            self._enable_action_heads_training()
        else:
            self.pipe.denoising_model().requires_grad_(True)
        self._log_training_architecture_summary(lora_reloaded_from_checkpoint)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    def init_optimizer_scheduler(
        self,
        optimizer_config,
        scheduler_config,
    ):
        optimizer = instantiate(optimizer_config, model=self.net)
        self._configure_action_head_param_group(optimizer)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ helpers ------------------------

    def _collect_action_head_modules(self):
        modules = []
        for attr in ("action_embedder_B_D", "action_embedder_B_3D"):
            if hasattr(self.pipe.dit, attr):
                modules.append(getattr(self.pipe.dit, attr))
        return modules

    def _collect_action_head_parameters(self):
        parameters = []
        for module in self._collect_action_head_modules():
            parameters.extend(list(module.parameters()))
        return parameters

    def _enable_action_heads_training(self) -> None:
        for module in self._collect_action_head_modules():
            module.requires_grad_(True)

    def _configure_action_head_param_group(self, optimizer: torch.optim.Optimizer) -> None:
        action_parameters = [p for p in self._collect_action_head_parameters() if p.requires_grad]
        if not action_parameters:
            return

        action_param_ids = {id(param) for param in action_parameters}

        for group in optimizer.param_groups:
            group["params"] = [param for param in group["params"] if id(param) not in action_param_ids]

        if not optimizer.param_groups:
            raise RuntimeError("Optimizer has no parameter groups after removing action head parameters.")

        reference_group = optimizer.param_groups[0]
        base_lr = reference_group.get("lr", 0.0)
        weight_decay = reference_group.get("weight_decay", 0.0)
        multiplier = getattr(self.config, "action_embedder_lr_multiplier", 1.0)
        action_head_lr = base_lr * multiplier

        optimizer.add_param_group(
            {
                "params": action_parameters,
                "lr": action_head_lr,
                "weight_decay": weight_decay,
            }
        )

        log.info(
            "Assigned dedicated optimizer group for action heads: lr=%.2e (multiplier %.2fx base) | params=%d",
            action_head_lr,
            multiplier,
            sum(param.numel() for param in action_parameters),
        )

    def _maybe_reload_lora_from_checkpoint(self) -> bool:
        dit_path = getattr(self.config.model_manager_config, "dit_path", "")
        if not dit_path:
            log.info("No DiT checkpoint path provided; skipping LoRA reload.")
            return False

        state_dict = load_state_dict(dit_path)
        lora_keys = [key for key in state_dict if "lora_" in key]
        if not lora_keys:
            log.info("Checkpoint does not contain LoRA parameters; skipping LoRA reload.")
            return False

        def load_into_module(module: torch.nn.Module | None, prefix: str) -> bool:
            if module is None:
                return False
            filtered_state = {}
            for key, value in state_dict.items():
                if "lora_" not in key:
                    continue
                if key.startswith(prefix):
                    filtered_state[key[len(prefix) :]] = value
                elif key.startswith(prefix + "module."):
                    filtered_state[key[len(prefix) + len("module.") :]] = value
            if not filtered_state:
                return False
            missing, unexpected = module.load_state_dict(filtered_state, strict=False, assign=True)
            if missing:
                log.debug("Missing keys when loading LoRA params (%s): %s", prefix, missing)
            if unexpected:
                log.debug("Unexpected keys when loading LoRA params (%s): %s", prefix, unexpected)
            return True

        loaded_regular = load_into_module(self.pipe.dit, "net.")
        loaded_ema = load_into_module(self.pipe.dit_ema, "net_ema.")

        del state_dict

        if loaded_regular or loaded_ema:
            log.success(
                "Restored LoRA parameters from checkpoint (regular=%s, ema=%s)",
                loaded_regular,
                loaded_ema,
            )
            return True

        log.warning(
            "LoRA parameters were present in checkpoint but did not match any modules; they may use an unsupported prefix."
        )
        return False

    def _log_training_architecture_summary(self, lora_reloaded_from_checkpoint: bool) -> None:
        action_trainable = {
            module_name: any(param.requires_grad for param in getattr(self.pipe.dit, module_name).parameters())
            for module_name in ("action_embedder_B_D", "action_embedder_B_3D")
            if hasattr(self.pipe.dit, module_name)
        }

        log.info("=== Action-Conditioned Training Configuration ===")
        log.info("  Training architecture: %s", self.config.train_architecture)
        log.info(
            "  Action head LR multiplier: %.2f",
            getattr(self.config, "action_embedder_lr_multiplier", 1.0),
        )
        for name, is_trainable in action_trainable.items():
            log.info("  %s trainable: %s", name, is_trainable)

        if self.config.train_architecture == "lora":
            log.info(
                "  LoRA settings -> rank: %d | alpha: %d | targets: %s",
                self.config.lora_rank,
                self.config.lora_alpha,
                self.config.lora_target_modules,
            )
            log.info("  LoRA weights restored from checkpoint: %s", lora_reloaded_from_checkpoint)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info("  Total trainable parameters in model: %d", trainable_params)
