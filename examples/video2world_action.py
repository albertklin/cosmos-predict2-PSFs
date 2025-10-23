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

import argparse
import json
import os

import mediapy as mp
import numpy as np
from pathlib import Path
from typing import Iterable, Optional

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from omegaconf import OmegaConf

from cosmos_predict2.configs.action_conditioned.config import get_cosmos_predict2_action_conditioned_pipeline
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from imaginaire.constants import (
    CosmosPredict2ActionConditionedFPS,
    CosmosPredict2ActionConditionedModelSize,
    CosmosPredict2ActionConditionedResolution,
    CosmosPredict2Video2WorldAspectRatio,
    get_cosmos_predict2_action_conditioned_checkpoint,
)
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts

import time


def add_lora_to_model(
    model,
    lora_rank=16,
    lora_alpha=16,
    lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
    init_lora_weights=True,
):
    """Add LoRA adapters to a model using the PEFT library."""
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    # Upcast LoRA parameters to fp32 for better stability
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    return model


def setup_lora_pipeline(
    config, dit_path: str, args: argparse.Namespace
):
    """Set up an action-conditioned pipeline with LoRA support."""
    from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
    from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
    from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
    from cosmos_predict2.schedulers.rectified_flow_scheduler import (
        RectifiedFlowAB2Scheduler,
    )
    from imaginaire.lazy_config import instantiate
    from imaginaire.utils.ema import FastEmaModelUpdater

    pipe = Video2WorldActionConditionedPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.config = config
    pipe.precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.precision]
    pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
    log.warning(f"precision {pipe.precision}")

    pipe.sigma_data = config.sigma_data
    pipe.setup_data_key()

    pipe.scheduler = RectifiedFlowAB2Scheduler(
        sigma_min=config.timestamps.t_min,
        sigma_max=config.timestamps.t_max,
        order=config.timestamps.order,
        t_scaling_factor=config.rectified_flow_t_scaling_factor,
    )
    pipe.scaling = RectifiedFlowScaling(
        pipe.sigma_data,
        config.rectified_flow_t_scaling_factor,
        config.rectified_flow_loss_weight_uniform,
    )

    pipe.tokenizer = instantiate(config.tokenizer)
    assert pipe.tokenizer.latent_ch == pipe.config.state_ch, (
        f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"
    )

    pipe.text_encoder = None
    pipe.conditioner = instantiate(config.conditioner)
    assert sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0, (
        "conditioner should not have learnable parameters"
    )

    pipe.prompt_refiner = CosmosReason1(
        checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
        offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
        enabled=config.prompt_refiner_config.enabled,
    )
    if config.guardrail_config.enabled:
        from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

        pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
            config.guardrail_config.checkpoint_dir,
            config.guardrail_config.offload_model_to_cpu,
        )
        pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
            config.guardrail_config.checkpoint_dir,
            config.guardrail_config.offload_model_to_cpu,
        )
    else:
        pipe.text_guardrail_runner = None
        pipe.video_guardrail_runner = None

    log.info("Initializing DiT model...")
    with init_weights_on_device():
        dit_config = config.net
        pipe.dit = instantiate(dit_config).eval()

    log.info("Adding LoRA adapters to DiT model...")
    pipe.dit = add_lora_to_model(
        pipe.dit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
    )

    if config.ema.enabled:
        log.info("Setting up EMA model with LoRA adapters...")
        pipe.dit_ema = instantiate(dit_config).eval()
        pipe.dit_ema.requires_grad_(False)
        pipe.dit_ema = add_lora_to_model(
            pipe.dit_ema,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            init_lora_weights=args.init_lora_weights,
        )
        pipe.dit_ema_worker = FastEmaModelUpdater()
        s = config.ema.rate
        pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
        pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

    if dit_path:
        log.info(f"Loading LoRA checkpoint from {dit_path}")
        state_dict = load_state_dict(dit_path)
        state_dict_dit = {}
        state_dict_ema = {}

        def _normalize_dit_key(key: str, prefix: str) -> str:
            key = key[len(prefix) :]
            if key.startswith("module."):
                return key[len("module.") :]
            if key.startswith("module_ema."):
                return key[len("module_ema.") :]
            return key

        for k, v in state_dict.items():
            if k.startswith("net."):
                state_dict_dit[_normalize_dit_key(k, "net.")] = v
            elif k.startswith("net_ema."):
                state_dict_ema[_normalize_dit_key(k, "net_ema.")] = v

        log.info(
            f"Checkpoint tensors split into {len(state_dict_dit)} regular and {len(state_dict_ema)} EMA entries"
        )
        if not state_dict_dit:
            log.warning("No regular DiT weights were found in the checkpoint; base model will remain uninitialized")
        missing = pipe.dit.load_state_dict(state_dict_dit, strict=False, assign=True)
        if missing.missing_keys:
            log.warning(f"Missing keys in regular model: {missing.missing_keys}")
        else:
            log.info("All regular DiT parameters were loaded from the checkpoint")
        if missing.unexpected_keys:
            log.warning(f"Unexpected keys in regular model: {missing.unexpected_keys}")

        if config.ema.enabled and state_dict_ema:
            missing_ema = pipe.dit_ema.load_state_dict(state_dict_ema, strict=False, assign=True)
            if missing_ema.missing_keys:
                log.warning(f"Missing keys in EMA model: {missing_ema.missing_keys}")
            else:
                log.info("All EMA DiT parameters were loaded from the checkpoint")
            if missing_ema.unexpected_keys:
                log.warning(f"Unexpected keys in EMA model: {missing_ema.unexpected_keys}")
        elif config.ema.enabled:
            log.warning("EMA is enabled but no EMA weights were found in the checkpoint")
        del state_dict, state_dict_dit, state_dict_ema
        log.success(f"Successfully loaded LoRA checkpoint from {dit_path}")
    else:
        log.warning("No checkpoint path provided, using random weights")

    pipe.dit = pipe.dit.to(device="cuda", dtype=torch.bfloat16)
    if config.ema.enabled:
        pipe.dit_ema = pipe.dit_ema.to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.empty_cache()

    if parallel_state.is_initialized():
        pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
    else:
        pipe.data_parallel_size = 1

    total_params = sum(p.numel() for p in pipe.dit.parameters())
    trainable_params = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable LoRA parameters: {trainable_params:,}")
    log.info(f"LoRA parameter ratio: {trainable_params / total_params * 100:.2f}%")

    return pipe


def get_action_sequence(annotation_path):
    with open(annotation_path) as file:
        data = json.load(file)

    # rescale the action to the original scale
    # action_ee = np.array(data["action"])[:, :6] * 20
    action_ee = np.array(data["action"])[:, :6]
    gripper = np.array(data["continuous_gripper_state"])[1:, None]

    # concatenate the end-effector displacement and gripper width
    action = np.concatenate([action_ee, gripper], axis=1)
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=CosmosPredict2ActionConditionedModelSize.__args__,
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--resolution",
        choices=CosmosPredict2ActionConditionedResolution.__args__,
        default="480",
        help="Resolution of the base model checkpoint to use",
    )
    parser.add_argument(
        "--fps",
        type=int,
        # choices=CosmosPredict2ActionConditionedFPS.__args__,
        default=CosmosPredict2ActionConditionedFPS.__args__[0],
        help="Frame rate of the base model checkpoint to use",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=CosmosPredict2Video2WorldAspectRatio.__args__,
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    # LoRA-specific arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA inference mode",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank of the LoRA adaptation",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument(
        "--init_lora_weights",
        action="store_true",
        default=True,
        help="Whether to initialize LoRA weights",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )
    parser.add_argument(
        "--use_cuda_graphs",
        action="store_true",
        help="Use CUDA Graphs for the action-conditioned inference",
    )
    parser.add_argument(
        "--input_videos",
        nargs="+",
        type=str,
        required=True,
        help="List of input videos for conditioning",
    )
    parser.add_argument(
        "--input_annotations",
        nargs="+",
        type=str,
        required=True,
        help="List of annotation files corresponding to input_videos",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=12,
        help="Chunk size",
    )
    parser.add_argument(
        "--use_teacher_forcing",
        action="store_true",
        help="Enable teacher forcing by grounding chunks on the ground-truth video",
    )
    parser.add_argument(
        "--teacher_forcing_interval",
        type=int,
        default=12,
        help="Number of action steps per chunk when using teacher forcing",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive mode")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of random seeds for reproducibility",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of generations for the input",
    )
    parser.add_argument(
        "--pipeline_seed",
        type=int,
        default=0,
        help="Seed used for pipeline initialization",
    )
    parser.add_argument(
        "--save_paths",
        nargs="+",
        type=str,
        required=True,
        help="List of paths to save the generated videos",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--negative_prompt", default=""
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    config = get_cosmos_predict2_action_conditioned_pipeline(
        model_size=args.model_size, resolution=args.resolution, fps=args.fps
    )
    config.resolution = args.resolution
    config.net.action_dim=7*args.chunk_size
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = get_cosmos_predict2_action_conditioned_checkpoint(
            model_size=args.model_size, resolution=args.resolution, fps=args.fps
        )

    misc.set_random_seed(seed=args.pipeline_seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    if args.use_lora:
        log.info("LoRA inference mode detected - using LoRA pipeline")
        pipe = setup_lora_pipeline(config=config, dit_path=dit_path, args=args)
    else:
        pipe = Video2WorldActionConditionedPipeline.from_config(
            config=config,
            dit_path=dit_path,
            use_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=args.load_ema,
            load_prompt_refiner=True,
        )

    return pipe, dit_path


def read_first_frame(video_path):
    video = mp.read_video(video_path)  # Returns (T, H, W, C) numpy array
    return video[0]  # Return first frame as numpy array


def _extract_action_dim_from_config(net_cfg) -> Optional[int]:
    if net_cfg is None:
        return None
    try:
        net_dict = OmegaConf.to_container(net_cfg, resolve=True)
    except Exception:  # noqa: BLE001 - best effort
        return None

    if isinstance(net_dict, dict) and "action_dim" in net_dict:
        try:
            return int(net_dict["action_dim"])
        except (TypeError, ValueError):
            return None
    return None


def determine_expected_action_dim(
    pipe: Video2WorldActionConditionedPipeline, dit_path: str
) -> int:
    """Infer the flattened action dimension expected by the pipeline."""

    action_dim = _extract_action_dim_from_config(getattr(pipe.config, "net", None))
    if action_dim is not None:
        return action_dim

    if hasattr(pipe, "dit") and hasattr(pipe.dit, "action_embedder_B_D"):
        embedder = pipe.dit.action_embedder_B_D
        if hasattr(embedder, "fc1") and hasattr(embedder.fc1, "in_features"):
            return int(embedder.fc1.in_features)

    raise RuntimeError("Unable to determine the expected action_dim for the pipeline")


def tensor_last_frame_to_numpy(video: torch.Tensor) -> np.ndarray:
    frame = video[:, :, -1].permute(0, 2, 3, 1).cpu().numpy()
    frame = ((frame / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return frame


def highlight_ground_truth_frames(
    video: torch.Tensor, frame_indices: Iterable[int], border_thickness: int = 2
) -> torch.Tensor:
    """Highlight selected frames with a green border to indicate teacher forcing."""

    if video is None or not isinstance(video, torch.Tensor):
        return video

    total_frames = video.shape[2]
    if total_frames == 0:
        return video

    # Normalize indices and keep only those that fall within the video length.
    unique_indices = sorted({int(i) for i in frame_indices if 0 <= int(i) < total_frames})
    if not unique_indices:
        return video

    height, width = video.shape[-2:]
    thickness = max(1, min(border_thickness, height, width))

    border_color = torch.tensor([-1.0, 1.0, -1.0], dtype=video.dtype, device=video.device).view(
        1, -1, 1, 1
    )

    for frame_idx in unique_indices:
        frame = video[:, :, frame_idx]
        frame[:, :, :thickness, :] = border_color
        frame[:, :, -thickness:, :] = border_color
        frame[:, :, :, :thickness] = border_color
        frame[:, :, :, -thickness:] = border_color

    return video


def process_single_generation(
    pipe,
    input_path,
    input_annotation,
    output_path,
    negative_prompt,
    guidance,
    seed,
    chunk_size,
    autoregressive,
    use_teacher_forcing,
    teacher_forcing_interval,
    expected_action_dim,
    *,
    batch_size: int,
    num_conditional_frames: int,
    aspect_ratio: str,
    use_cuda_graphs: bool,
    output_fps: int,
):
    actions = get_action_sequence(input_annotation)

    ground_truth_video = None
    if use_teacher_forcing:
        ground_truth_video = mp.read_video(input_path)
        if ground_truth_video.ndim != 4 or ground_truth_video.shape[0] == 0:
            raise ValueError("Unable to read ground-truth video for teacher forcing")
        first_frame = ground_truth_video[0]
    else:
        first_frame = read_first_frame(input_path)

    first_frame = np.broadcast_to(first_frame, (batch_size, *first_frame.shape))
    actions = np.broadcast_to(actions, (batch_size, *actions.shape))

    per_action_dim = actions.shape[-1]
    if expected_action_dim % per_action_dim != 0:
        raise ValueError(
            "Action feature dimension %d does not divide expected action_dim %d"
            % (per_action_dim, expected_action_dim)
        )

    valid_chunk_size = expected_action_dim // per_action_dim
    if chunk_size != valid_chunk_size:
        raise ValueError(
            (
                "chunk_size=%d is incompatible with model action_dim=%d; expected %d actions per chunk"
            )
            % (chunk_size, expected_action_dim, valid_chunk_size)
        )

    chunk_size = valid_chunk_size

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}")

    if use_teacher_forcing:
        if teacher_forcing_interval <= 0:
            raise ValueError("teacher_forcing_interval must be a positive integer")

        if teacher_forcing_interval % chunk_size != 0:
            raise ValueError(
                (
                    "teacher_forcing_interval=%d must be a positive multiple of chunk_size=%d to respect action_dim=%d"
                )
                % (teacher_forcing_interval, chunk_size, expected_action_dim)
            )

        log.info(
            f"Using teacher forcing with interval size of {teacher_forcing_interval} action steps"
        )

        video_chunks = []
        chunk_grounding_flags: list[bool] = []
        num_action_steps = actions.shape[1]
        if num_action_steps < chunk_size:
            # raise ValueError(
            #     "Not enough action steps (%d) to form a single chunk of size %d"
            #     % (num_action_steps, chunk_size)
            # )
            log.warning(
                "Not enough action steps (%d) to form a single chunk of size %d"
                % (num_action_steps, chunk_size)
            )
            return

        remainder = num_action_steps % chunk_size
        if remainder != 0:
            log.warning(f"Dropping {remainder} trailing action steps that do not fill a complete chunk")

        last_generated_frame: Optional[np.ndarray] = None
        for start in range(0, num_action_steps - chunk_size + 1, chunk_size):
            end = start + chunk_size
            if end > num_action_steps:
                log.info("Reached end of actions")
                break

            chunk_actions = actions[:, start:end]

            use_ground_truth = (start % teacher_forcing_interval) == 0
            if use_ground_truth:
                frame_idx = min(start, ground_truth_video.shape[0] - 1)
                if frame_idx != start:
                    log.warning(f"Teacher forcing start index {start} exceeds "
                    f"available frames ({ground_truth_video.shape[0]}); using frame {frame_idx}")
                chunk_first_frame = np.broadcast_to(
                    ground_truth_video[frame_idx], first_frame.shape
                )
            else:
                if last_generated_frame is None:
                    raise RuntimeError(
                        "Missing generated frame to seed teacher forcing continuation"
                    )
                chunk_first_frame = last_generated_frame

            start_time = time.time()
            chunk_video = pipe(
                chunk_first_frame,
                chunk_actions,
                num_conditional_frames=num_conditional_frames,
                guidance=guidance,
                seed=seed + start,
                prompt="",
                negative_prompt=negative_prompt,
                fps=output_fps,
                aspect_ratio=aspect_ratio,
                use_cuda_graphs=use_cuda_graphs,
            ) # B x C x T x H x W
            print(f"Inference time: {time.time() - start_time:.2f} s")
            video_chunks.append(chunk_video)
            chunk_grounding_flags.append(use_ground_truth)
            last_generated_frame = tensor_last_frame_to_numpy(chunk_video)

        if video_chunks:
            video = video_chunks[0]
            total_frames = video.shape[2]
            teacher_frame_indices = [0] if chunk_grounding_flags[0] else []

            for idx, chunk in enumerate(video_chunks[1:], start=1):
                grounded = chunk_grounding_flags[idx]
                if grounded:
                    teacher_frame_indices.append(total_frames-1)
                    video = torch.cat([video[:, :, :-1], chunk], dim=2)
                    total_frames += chunk.shape[2] - 1
                else:
                    video = torch.cat([video, chunk[:, :, 1:]], dim=2)
                    total_frames += chunk.shape[2] - 1

            video = highlight_ground_truth_frames(video, teacher_frame_indices)
        else:
            video = None

    elif autoregressive:
        log.info("Using autoregressive mode")
        video_chunks = []
        for i in range(0, actions.shape[1], chunk_size):
            if actions.shape[1] - i < chunk_size:
                log.info("Reached end of actions")
                break
            start_time = time.time()
            video = pipe(
                first_frame,
                actions[:, i : i + chunk_size],
                num_conditional_frames=num_conditional_frames,
                guidance=guidance,
                seed=seed+i,
                prompt="",
                negative_prompt=negative_prompt,
                fps=output_fps,
                aspect_ratio=aspect_ratio,
                use_cuda_graphs=use_cuda_graphs,
            ) # B x C x T x H x W
            print(f"Inference time: {time.time() - start_time:.2f} s")
            first_frame = tensor_last_frame_to_numpy(video)
            video_chunks.append(video)
        video = torch.cat([video_chunks[0]] + [chunk[:, :, 1:] for chunk in video_chunks[1:]], dim=2)
    else:
        start_time = time.time()
        video = pipe(
            first_frame,
            actions[:, :chunk_size],
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            prompt="",
            negative_prompt=negative_prompt,
            fps=output_fps,
            aspect_ratio=aspect_ratio,
            use_cuda_graphs=use_cuda_graphs,
        )
        print(f"Inference time: {time.time() - start_time:.2f} s")

    if video is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        log.info(f"Saving generated videos to: {output_path}")
        if len(video) == 1:
            save_image_or_video(video[0], output_path, fps=output_fps)
        else:
            base = Path(output_path).with_suffix("")
            for i, v in enumerate(video):
                save_image_or_video(v, f"{base}_{i}.mp4", fps=output_fps)
        log.success(f"Successfully saved videos to: {output_path}")
        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {"negative_prompt": negative_prompt, "prompt": ""}
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")
        config_path = os.path.splitext(output_path)[0] + ".yaml"
        LazyConfig.save_yaml(pipe.config, config_path)
        log.success(f"Successfully saved config file to: {config_path}")
        return True
    return False


def generate_video(
    args: argparse.Namespace,
    pipe: Video2WorldActionConditionedPipeline,
    input_video: str,
    input_annotation: str,
    save_path: str,
    seed: int = 0,
    expected_action_dim: int | None = None,
) -> None:
    if expected_action_dim is None:
        raise ValueError("expected_action_dim must be provided")

    process_single_generation(
        pipe=pipe,
        input_path=input_video,
        input_annotation=input_annotation,
        output_path=save_path,
        negative_prompt=args.negative_prompt,
        guidance=args.guidance,
        seed=seed,
        chunk_size=args.chunk_size,
        autoregressive=args.autoregressive,
        use_teacher_forcing=args.use_teacher_forcing,
        teacher_forcing_interval=args.teacher_forcing_interval,
        expected_action_dim=expected_action_dim,
        batch_size=args.batch_size,
        num_conditional_frames=args.num_conditional_frames,
        aspect_ratio=args.aspect_ratio,
        use_cuda_graphs=args.use_cuda_graphs,
        output_fps=args.fps,
    )
    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe, dit_path = setup_pipeline(args)
        expected_action_dim = determine_expected_action_dim(pipe, dit_path)
        input_videos = args.input_videos
        input_annotations = args.input_annotations
        save_paths = args.save_paths
        if not (
            len(input_videos) == len(save_paths) == len(input_annotations)
        ):
            raise ValueError(
                "input_videos, input_annotations, and save_paths must have the same length"
            )
        seeds = args.seeds if args.seeds else list(range(args.num_generations))
        for seed in seeds:
            for j in range(len(input_videos)):
                output_path = (
                    f"{Path(save_paths[j]).with_suffix('')}_pipeline_{args.pipeline_seed}_seed_{seed}.mp4"
                )
                generate_video(
                    args,
                    pipe,
                    input_video=input_videos[j],
                    input_annotation=input_annotations[j],
                    save_path=output_path,
                    seed=seed,
                    expected_action_dim=expected_action_dim,
                )
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
