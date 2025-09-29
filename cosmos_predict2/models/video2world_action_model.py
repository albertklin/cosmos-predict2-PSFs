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
from typing import Any

import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh

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
            load_weights=False,
        )

        self.freeze_parameters()
        self._enable_action_heads_training()
        checkpoint_report: dict[str, Any] | None = None
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
            checkpoint_report = self.pipe.load_checkpoint(
                config.model_manager_config.dit_path
            )
            self._log_lora_statistics()
            log.info(
                "LoRA weights restored from checkpoint: %s",
                bool(checkpoint_report and checkpoint_report.get("contains_lora")),
            )
            self._enable_action_heads_training()
        else:
            checkpoint_report = self.pipe.load_checkpoint(
                config.model_manager_config.dit_path
            )
            self.pipe.denoising_model().requires_grad_(True)
        self._log_training_architecture_summary(checkpoint_report)
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

        action_param_count = sum(param.numel() for param in action_parameters)
        log.info(
            f"Assigned dedicated optimizer group for action heads: "
            f"lr={action_head_lr:.2e} (multiplier {multiplier:.2f}x base) "
            f"| params={action_param_count}"
        )

    def _log_training_architecture_summary(self, checkpoint_report: dict[str, Any] | None) -> None:
        action_trainable = {
            module_name: any(param.requires_grad for param in getattr(self.pipe.dit, module_name).parameters())
            for module_name in ("action_embedder_B_D", "action_embedder_B_3D")
            if hasattr(self.pipe.dit, module_name)
        }

        log.info("=== Action-Conditioned Training Configuration ===")
        log.info(f"  Training architecture: {self.config.train_architecture}")
        log.info(
            f"  Action head LR multiplier: {getattr(self.config, 'action_embedder_lr_multiplier', 1.0):.2f}"
        )
        for name, is_trainable in action_trainable.items():
            log.info(f"  {name} trainable: {is_trainable}")

        if self.config.train_architecture == "lora":
            log.info(
                f"  LoRA settings -> rank: {self.config.lora_rank} | alpha: {self.config.lora_alpha} "
                f"| targets: {self.config.lora_target_modules}"
            )
            log.info(
                "  LoRA weights restored from checkpoint: %s",
                bool(checkpoint_report and checkpoint_report.get("contains_lora")),
            )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  Total trainable parameters in model: {trainable_params}")
