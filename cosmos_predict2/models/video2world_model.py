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
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.configs.base.config_video2world import (
    Video2WorldPipelineConfig,
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline, _prepare_dit_state_dicts
from cosmos_predict2.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2.utils.torch_future import clip_grad_norm_
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log


@attrs.define(slots=False)
class Predict2ModelManagerConfig:
    # Local path, use it in fast debug run
    dit_path: str = get_cosmos_predict2_video2world_checkpoint(model_size="2B")
    # For inference
    text_encoder_path: str = ""  # not used in training.


@attrs.define(slots=False)
class Predict2Video2WorldModelConfig:
    train_architecture: str = "base"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    init_lora_weights: bool = True
    action_embedder_lr_multiplier: float = 1.0

    precision: str = "bfloat16"

    def __attrs_post_init__(self):
        """Validate LoRA configuration after initialization."""
        if self.train_architecture == "lora":
            if self.lora_rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.lora_rank}")
            if self.lora_alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
            if not self.lora_target_modules.strip():
                raise ValueError("LoRA target_modules cannot be empty")

            # Warn about potentially inefficient configurations
            if self.lora_rank > 64:
                log.warning(f"High LoRA rank ({self.lora_rank}) may reduce training efficiency")
            if self.lora_alpha != self.lora_rank:
                log.info(
                    f"LoRA alpha ({self.lora_alpha}) != rank ({self.lora_rank}), scaling factor: {self.lora_alpha / self.lora_rank}"
                )

        if self.action_embedder_lr_multiplier <= 0:
            raise ValueError(
                f"action_embedder_lr_multiplier must be positive, got {self.action_embedder_lr_multiplier}"
            )

    input_video_key: str = "video"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0

    adjust_video_noise: bool = True

    # This is used for the original way to load models
    model_manager_config: Predict2ModelManagerConfig = Predict2ModelManagerConfig()  # noqa: RUF009
    # This is a new way to load models
    pipe_config: Video2WorldPipelineConfig = get_cosmos_predict2_video2world_pipeline(model_size="2B")  # noqa: RUF009
    # debug flag
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0  # 0 means not using fsdp, -1 means set to world size
    # High sigma strategy
    high_sigma_ratio: float = 0.0


class Predict2Video2WorldModel(ImaginaireModel):
    def __init__(self, config: Predict2Video2WorldModelConfig):
        super().__init__()

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

        lora_injection_fn: Callable[[torch.nn.Module], None] | None = None
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

        # New way to init pipe
        self.pipe = Video2WorldPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
            lora_injection_fn=lora_injection_fn,
        )

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self._prepare_lora_trainable_parameters()
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
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    # New function, added for i4 adaption
    @property
    def net(self) -> torch.nn.Module:
        return self.pipe.dit

    # New function, added for i4 adaption
    @property
    def net_ema(self) -> torch.nn.Module:
        return self.pipe.dit_ema

    # New function, added for i4 adaption
    def init_optimizer_scheduler(
        self,
        optimizer_config: LazyDict[torch.optim.Optimizer],
        scheduler_config: LazyDict[torch.optim.lr_scheduler.LRScheduler],
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = instantiate(optimizer_config, model=self.net)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        self._log_training_overview(optimizer)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.pipe_config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.pipe.dit_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    # New function, added for i4 adaption
    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.pipe_config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        for module in [self.net, self.pipe.tokenizer]:
            if module is not None:
                module.to(memory_format=memory_format, **self.tensor_kwargs)

    def freeze_parameters(self) -> None:
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model: torch.nn.Module,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights: bool = True,
    ) -> None:
        """Add LoRA (Low-Rank Adaptation) adapters to the model.

        This function injects LoRA adapters into specified modules of the model,
        enabling parameter-efficient fine-tuning by training only a small number
        of additional parameters.

        Args:
            model: The PyTorch model to add LoRA adapters to
            lora_rank: The rank of the LoRA adaptation matrices. Higher rank allows
                      more expressiveness but uses more parameters (default: 4)
            lora_alpha: Scaling parameter for LoRA. Controls the magnitude of the
                       LoRA adaptation (default: 4)
            lora_target_modules: Comma-separated string of module names to target
                               for LoRA adaptation (default: attention and MLP layers)
            init_lora_weights: Whether to initialize LoRA weights properly (default: True)

        Raises:
            ImportError: If PEFT library is not installed
            ValueError: If invalid parameters are provided
            RuntimeError: If LoRA injection fails
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "PEFT library is required for LoRA training. Please install it with: pip install peft"
            ) from e

        # Validate parameters
        if lora_rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {lora_rank}")
        if lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")

        target_modules_list = [module.strip() for module in lora_target_modules.split(",")]
        if not target_modules_list:
            raise ValueError("LoRA target_modules cannot be empty")

        # Validate target modules exist in model
        model_module_names = set(name for name, _ in model.named_modules())
        invalid_modules = []
        for target_module in target_modules_list:
            # Check if any module contains this target pattern
            if not any(target_module in module_name for module_name in model_module_names):
                invalid_modules.append(target_module)

        if invalid_modules:
            log.warning(f"Target modules not found in model: {invalid_modules}")

        # Add LoRA to model
        self.lora_alpha = lora_alpha

        log.info(f"Adding LoRA adapters: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules_list}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=target_modules_list,
        )

        try:
            model = inject_adapter_in_model(lora_config, model)
        except Exception as e:
            raise RuntimeError(f"Failed to inject LoRA adapters into model: {e}") from e

        # Count and log LoRA parameters
        lora_params = 0
        total_params = 0
        for name, param in model.named_parameters():  # noqa: B007
            total_params += param.numel()
            if param.requires_grad:
                lora_params += param.numel()
                # Upcast LoRA parameters into fp32
                param.data = param.to(torch.float32)

        log.info(
            f"LoRA injection successful: {lora_params:,} trainable parameters out of {total_params:,} total ({100 * lora_params / total_params:.3f}%)"
        )

    def _log_lora_statistics(self) -> None:
        """Log detailed LoRA parameter statistics."""
        lora_params_by_type = {}
        total_lora_params = 0

        for name, param in self.pipe.dit.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                param_type = "lora_A" if "lora_A" in name else "lora_B" if "lora_B" in name else "other_lora"
                if param_type not in lora_params_by_type:
                    lora_params_by_type[param_type] = 0
                lora_params_by_type[param_type] += param.numel()
                total_lora_params += param.numel()

        if total_lora_params > 0:
            log.info("LoRA parameter breakdown:")
            for param_type, count in lora_params_by_type.items():
                log.info(f"  {param_type}: {count:,} parameters")
            log.info(f"  Total LoRA: {total_lora_params:,} parameters")
        else:
            log.warning("No LoRA parameters found in model")

    def _prepare_lora_trainable_parameters(self) -> None:
        """Enable gradients for LoRA parameters while keeping base weights frozen."""

        if not hasattr(self.pipe, "dit"):
            log.warning("LoRA preparation skipped because DiT module is not initialized.")
            return

        trainable_names: list[str] = []
        for name, param in self.pipe.dit.named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(True)
                if param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)
                trainable_names.append(name)

        if self.pipe.dit_ema is not None:
            for name, param in self.pipe.dit_ema.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad_(False)

        if trainable_names:
            preview = ", ".join(trainable_names[:8])
            if len(trainable_names) > 8:
                preview += f", ... (+{len(trainable_names) - 8} more)"
            log.info(f"Enabled training for {len(trainable_names)} LoRA tensors: {preview}")
        else:
            log.warning(
                "LoRA training requested but no LoRA parameters were detected on the DiT module."
            )

    @staticmethod
    def _format_name_list(names: list[str], limit: int = 8) -> str:
        if not names:
            return "none"
        if len(names) <= limit:
            return ", ".join(names)
        return ", ".join(names[:limit]) + f", ... (+{len(names) - limit} more)"

    def _log_training_overview(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Log a summary of trainable parameters and optimizer learning rates."""

        params = list(self.named_parameters())
        trainable = [(name, param) for name, param in params if param.requires_grad]
        frozen = [(name, param) for name, param in params if not param.requires_grad]

        trainable_param_count = sum(param.numel() for _, param in trainable)
        frozen_param_count = sum(param.numel() for _, param in frozen)

        log.info("Training parameter summary:")
        log.info(
            f"  Trainable tensors: {len(trainable)} | parameters: {trainable_param_count:,}"
        )
        if trainable:
            log.info(f"  Trainable names: {self._format_name_list([name for name, _ in trainable])}")

        log.info(f"  Frozen tensors: {len(frozen)} | parameters: {frozen_param_count:,}")
        if frozen:
            log.info(f"  Frozen names: {self._format_name_list([name for name, _ in frozen])}")

        if optimizer is None:
            return

        param_name_lookup = {id(param): name for name, param in params}
        log.info("  Optimizer parameter groups:")
        for idx, group in enumerate(optimizer.param_groups):
            lr = group.get("lr", 0.0)
            lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
            group_params: list[torch.nn.Parameter] = group.get("params", [])
            total_group_params = sum(param.numel() for param in group_params)
            trainable_group_params = sum(param.numel() for param in group_params if param.requires_grad)
            group_names = [
                param_name_lookup.get(id(param), "<unnamed>") for param in group_params if param.requires_grad
            ]
            log.info(
                f"    Group {idx}: lr={lr_str} | trainable params={trainable_group_params:,} / {total_group_params:,} "
                f"| tensors={self._format_name_list(group_names)}"
            )

    def _log_checkpoint_details(
        self, label: str, keys: Iterable[str], results: _IncompatibleKeys
    ) -> None:
        keys = list(keys)
        missing = list(results.missing_keys)
        unexpected = list(results.unexpected_keys)
        loaded_count = max(len(keys) - len(unexpected), 0)

        log.info(
            f"[{label}] Checkpoint load summary: loaded={loaded_count} | new={len(missing)} | unexpected={len(unexpected)}"
        )
        if missing:
            log.info(f"[{label}] Newly initialized tensors: {self._format_name_list(missing)}")
        if unexpected:
            log.warning(f"[{label}] Unexpected checkpoint entries ignored: {self._format_name_list(unexpected)}")

    def setup_data_key(self) -> None:
        self.input_video_key = self.config.input_video_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_video_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_video_key should be present in the data_batch."
        )
        return is_image

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_video_key
        if isinstance(self.pipe.dit, WeightTrainingStat):
            if is_image:
                self.pipe.dit.accum_image_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size
            else:
                self.pipe.dit.accum_video_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size

    def draw_training_sigma_and_epsilon(self, x0_size: torch.Size, condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        if is_video_batch and self.config.high_sigma_ratio > 0:
            # Implement the high sigma strategy LOGUNIFORM200_100000
            LOG_200 = math.log(200)
            LOG_100000 = math.log(100000)
            mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
            log_new_sigma = (
                torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                + LOG_200
            )
            sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor):
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (1 + sigma) ** 2 / sigma**2

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: TextCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction
        model_pred = self.pipe.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        kendall_loss = edm_loss_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }
        output_batch["loss"] = kendall_loss.mean()  # check if this is what we want

        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        self.pipe.device = self.device

        # Loss
        self._update_train_stats(data_batch)

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

    @torch.no_grad()
    def validation_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        return self.training_step(data_batch, data_batch_idx)

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> dict[str, Any]:
        # the checkpoint format should be compatible with traditional imaginaire4
        # pipeline contains both net and net_ema
        # checkpoint should be saved/loaded from Model
        # checkpoint should be loadable from pipeline as well - We don't use Model for inference only jobs.

        net_state_dict = self.pipe.dit.state_dict(prefix="net.")
        if self.config.pipe_config.ema.enabled:
            ema_state_dict = self.pipe.dit_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)

        # convert DTensor to Tensor
        for key, val in net_state_dict.items():
            if isinstance(val, DTensor):
                # Convert to full tensor
                net_state_dict[key] = val.full_tensor().detach().cpu()
            else:
                net_state_dict[key] = val.detach().cpu()

        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load checkpoint weights into the DiT (and EMA) modules."""

        reg_state_dict, ema_state_dict = _prepare_dit_state_dicts(state_dict, load_ema_to_reg=False)

        reg_results: _IncompatibleKeys = self.pipe.dit.load_state_dict(
            reg_state_dict, strict=strict, assign=assign
        )

        ema_results: _IncompatibleKeys | None = None
        if self.config.pipe_config.ema.enabled and self.pipe.dit_ema is not None:
            if ema_state_dict:
                ema_results = self.pipe.dit_ema.load_state_dict(ema_state_dict, strict=strict, assign=assign)
            else:
                log.warning(
                    "EMA weights were not found in the checkpoint; initializing EMA parameters from the current DiT state."
                )
                if hasattr(self.pipe, "dit_ema_worker") and self.pipe.dit_ema is not None:
                    self.pipe.dit_ema_worker.copy_to(src_model=self.pipe.dit, tgt_model=self.pipe.dit_ema)
                ema_results = _IncompatibleKeys([], [])

        self._log_checkpoint_details("DiT", reg_state_dict.keys(), reg_results)
        if ema_results is not None:
            self._log_checkpoint_details("DiT EMA", ema_state_dict.keys(), ema_results)

        missing_keys = list(reg_results.missing_keys)
        unexpected_keys = list(reg_results.unexpected_keys)

        if ema_results is not None:
            missing_keys.extend(ema_results.missing_keys)
            unexpected_keys.extend(ema_results.unexpected_keys)

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.pipe_config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.pipe.ema_exp_coefficient + 1)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )