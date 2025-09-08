# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration helpers for action conditioned multi-view training."""

from __future__ import annotations

import dataclasses

from cosmos_predict2.conditioner import (
    BooleanFlag,
    ConditionLocation,
    MultiViewActionConditioner,
    ReMapkey,
    TextAttr,
)
from cosmos_predict2.configs.base.config_multiview import (
    ConditioningStrategy,
    MultiviewPipelineConfig,
)
from cosmos_predict2.configs.base.config_video2world import (
    CosmosGuardrailConfig,
    CosmosReason1Config,
    EMAConfig,
    TokenizerInterface,
)
from cosmos_predict2.models.multiview_action_dit import MultiViewActionDiT
from cosmos_predict2.models.text2image_dit import SACConfig
from imaginaire.constants import (
    CHECKPOINTS_DIR,
    COSMOS_REASON1_MODEL_DIR,
    get_cosmos_predict2_video2world_tokenizer,
)
from imaginaire.lazy_config import LazyCall as L

_PREDICT2_MULTIVIEW_ACTION_NET_2B = L(MultiViewActionDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,  # >= num_views * state_t
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    pos_emb_cls="rope3d",
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,  # no temporal rescaling
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(every_n_blocks=1, mode="predict2_2b_720"),
    state_t=4,  # frames per view processed by the model
    n_cameras_emb=4,  # number of camera views in the dataset
    view_condition_dim=4,  # must match n_cameras_emb
    concat_view_embedding=True,
    action_dim=7 * 12,  # 7D per-step action * 12 time steps
)

_PREDICT2_MULTIVIEW_ACTION_PIPELINE_2B = MultiviewPipelineConfig(
    adjust_video_noise=False,
    conditioner=L(MultiViewActionConditioner)(
        fps=L(ReMapkey)(dropout_rate=0.0, dtype=None, input_key="fps", output_key="fps"),
        padding_mask=L(ReMapkey)(dropout_rate=0.0, dtype=None, input_key="padding_mask", output_key="padding_mask"),
        text=L(TextAttr)(dropout_rate=0.0, input_key=["t5_text_embeddings"]),
        use_video_condition=L(BooleanFlag)(dropout_rate=0.0, input_key="fps", output_key="use_video_condition"),
        view_indices_B_T=L(ReMapkey)(
            input_key="latent_view_indices_B_T", output_key="view_indices_B_T", dropout_rate=0.0, dtype=None
        ),
        ref_cam_view_idx_sample_position=L(ReMapkey)(
            input_key="ref_cam_view_idx_sample_position",
            output_key="ref_cam_view_idx_sample_position",
            dropout_rate=0.0,
            dtype=None,
        ),
        action=L(ReMapkey)(input_key="action", output_key="action", dropout_rate=0.0, dtype=None),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames_per_view=0,
    max_num_conditional_frames_per_view=1,
    condition_locations=[ConditionLocation.FIRST_RANDOM_N],
    net=_PREDICT2_MULTIVIEW_ACTION_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="480",  # per-view height; change if dataset uses another resolution
    ema=L(EMAConfig)(enabled=False),
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=4,  # must match dataset.state_t and net.state_t
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B"),
    ),
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir=COSMOS_REASON1_MODEL_DIR,
        offload_model_to_cpu=True,
        enabled=False,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=False,
    ),
)


@dataclasses.dataclass(frozen=True)
class _Key:
    model_size: str


_PIPELINES: dict[_Key, MultiviewPipelineConfig] = {
    _Key("2B"): _PREDICT2_MULTIVIEW_ACTION_PIPELINE_2B,
}


def get_cosmos_predict2_multiview_action_pipeline(*, model_size: str = "2B") -> MultiviewPipelineConfig:
    key = _Key(model_size)
    return _PIPELINES[key]