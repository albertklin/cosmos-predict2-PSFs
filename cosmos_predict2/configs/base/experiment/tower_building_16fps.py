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
Tower Building 16fps Video2World experiment configuration.

Dataset: 2x2 composite view (480p) of full tower building trajectories.
Each trajectory shows a robot stacking 5 colored cubes on a white plate.
Model: Cosmos-Predict2-14B-Video2World with LoRA fine-tuning.

Checkpoint: Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID/model-480p-16fps.pt (robotics-adapted)

Usage:
    torchrun --nproc_per_node=8 --master_port=12341 \
        -m scripts.train --config=cosmos_predict2/configs/base/config.py -- \
        experiment=finetune_cosmos_cv_v2w_14b_tower_building_16fps \
        model.config.train_architecture=lora
"""

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# Tower building 16fps dataset
# Full trajectories: robot stacking 5 cubes (~10-15s each, 160-240 frames at 16fps)
# 2x2 composite view at 480p (480x640)
# num_frames=93 corresponds to state_t=24 (Cosmos 16fps default)
# Training samples random 93-frame clips from each full trajectory
tower_building_16fps_video_dataset = L(Dataset)(
    dataset_dir="datasets/tower_building_16fps/train",
    num_frames=93,
    video_size=(480, 640),
)

dataloader_tower_building_16fps = L(DataLoader)(
    dataset=tower_building_16fps_video_dataset,
    sampler=L(get_sampler)(dataset=tower_building_16fps_video_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# 14B LoRA fine-tuning for tower building
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
#   --config=cosmos_predict2/configs/base/config.py -- \
#   experiment=finetune_cosmos_cv_v2w_14b_tower_building_16fps \
#   model.config.train_architecture=lora
finetune_cosmos_cv_v2w_14b_tower_building_16fps = dict(
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
        name="14b_tower_building_16fps",
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
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_tower_building_16fps,
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
        cycle_lengths=[5_000],
        f_max=[0.25],
        f_min=[0.0],
    ),
)

for _item in [
    finetune_cosmos_cv_v2w_14b_tower_building_16fps,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
