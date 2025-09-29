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
from collections.abc import Mapping
from typing import Any

import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from cosmos_predict2.pipelines.video2world import _log_uninitialized_tensors
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

        lora_injection_fn = None
        if config.train_architecture == "lora":
            def _inject_lora(module: torch.nn.Module) -> None:
                self.add_lora_to_model(
                    module,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )

            lora_injection_fn = _inject_lora

        # NOTE: replace the pipeline with action-conditioned setup
        self.pipe = Video2WorldActionConditionedPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
            lora_injection_fn=lora_injection_fn,
        )

        def _log_weight_status(stage: str) -> None:
            _log_uninitialized_tensors(self.pipe.dit, f"{stage} Action DiT")
            if self.pipe.dit_ema is not None:
                _log_uninitialized_tensors(self.pipe.dit_ema, f"{stage} Action DiT EMA")

        _log_weight_status("Video2WorldActionModel (post-pipeline-init)")

        self.freeze_parameters()
        self._enable_action_heads_training()
        if config.train_architecture == "lora":
            self._prepare_lora_trainable_parameters()
            self._enable_action_heads_training()
            self._log_lora_statistics()
        else:
            self.pipe.denoising_model().requires_grad_(True)
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
            _log_weight_status("Video2WorldActionModel (post-FSDP)")
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")
            _log_weight_status("Video2WorldActionModel (final)")

    def init_optimizer_scheduler(
        self,
        optimizer_config,
        scheduler_config,
    ):
        optimizer = instantiate(optimizer_config, model=self.net)
        self._configure_action_head_param_group(optimizer)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        self._log_training_overview(optimizer)
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

        action_param_count = sum(param.numel() for param in action_parameters)
        log.info(
            f"Assigned dedicated optimizer group for action heads: "
            f"lr={action_head_lr:.2e} (multiplier {multiplier:.2f}x base) "
            f"| params={action_param_count}"
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        results = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if isinstance(results, _IncompatibleKeys):
            self._reinitialize_missing_action_heads(results)
        return results

    def _reinitialize_missing_action_heads(self, load_results: _IncompatibleKeys) -> None:
        missing = set(load_results.missing_keys)
        if not missing:
            return

        reinitialized: list[str] = []
        for attr in ("action_embedder_B_D", "action_embedder_B_3D"):
            module = getattr(self.pipe.dit, attr, None)
            if module is None:
                continue
            prefix = f"{attr}."
            if any(key.startswith(prefix) for key in missing):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                    reinitialized.append(attr)
                    if self.pipe.dit_ema is not None and hasattr(self.pipe.dit_ema, attr):
                        getattr(self.pipe.dit_ema, attr).reset_parameters()

        if reinitialized:
            self._enable_action_heads_training()
            log.info(
                "Action-conditioned checkpoint missing head weights; reinitialized modules: "
                + ", ".join(reinitialized)
            )