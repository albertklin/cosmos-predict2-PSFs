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

"""
Tower Building 16fps Video2World experiment configuration (short videos).

For shorter trajectories (e.g., single block placement) that don't have 93+ frames.
Uses state_t=16 (num_frames=61) instead of state_t=24 (num_frames=93).

Dataset: 2x2 composite view (480p) of tower building trajectories.
Model: Cosmos-Predict2-14B-Video2World with LoRA fine-tuning.

Checkpoint: Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID/model-480p-16fps.pt (robotics-adapted)

Usage:
    torchrun --nproc_per_node=8 --master_port=12341 \
        -m scripts.train --config=cosmos_predict2/configs/base/config.py -- \
        experiment=finetune_cosmos_cv_v2w_14b_tower_building_16fps_short \
        model.config.train_architecture=lora
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L

# Dataset path can be overridden via environment variable
# Set COSMOS_DATASET_DIR to override the default path
_DATASET_DIR = os.environ.get("COSMOS_DATASET_DIR", "datasets/tower_building_16fps/train")


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# Tower building 16fps dataset (short videos)
# Single block placement trajectories: ~3-6 seconds (48-96 frames at 16fps)
# 2x2 composite view at 480p (480x640)
# num_frames=61 corresponds to state_t=16
# Requires videos with at least 61 frames
tower_building_16fps_short_video_dataset = L(Dataset)(
    dataset_dir=_DATASET_DIR,
    num_frames=61,
    video_size=(480, 640),
)

dataloader_tower_building_16fps_short = L(DataLoader)(
    dataset=tower_building_16fps_short_video_dataset,
    sampler=L(get_sampler)(dataset=tower_building_16fps_short_video_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# 14B LoRA fine-tuning for tower building (short videos)
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
#   --config=cosmos_predict2/configs/base/config.py -- \
#   experiment=finetune_cosmos_cv_v2w_14b_tower_building_16fps_short \
#   model.config.train_architecture=lora
finetune_cosmos_cv_v2w_14b_tower_building_16fps_short = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b_480p_16fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="14b_tower_building_16fps_short",
    ),
    model=dict(
        config=dict(
            # LoRA configuration (enabled via command line: model.config.train_architecture=lora)
            lora_rank=32,
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            fsdp_shard_size=8,
            pipe_config=dict(
                state_t=16,  # Override: use shorter temporal context (61 frames instead of 93)
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_tower_building_16fps_short,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=5000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[100],
        cycle_lengths=[10_000],  # Must be >= max_iter (validated at training start)
        f_max=[0.25],
        f_min=[0.0],
    ),
)

for _item in [
    finetune_cosmos_cv_v2w_14b_tower_building_16fps_short,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
