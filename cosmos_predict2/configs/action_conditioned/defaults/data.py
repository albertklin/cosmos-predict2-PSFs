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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.action_conditioned.action_conditioned_dataset import ActionConditionedDataset
from imaginaire.lazy_config import LazyCall as L

base_path = "./datasets/bridge/"
train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


bridge_train_dataset = L(ActionConditionedDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=13,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
    output_fps=4,
)

bridge_val_dataset = L(ActionConditionedDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=13,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
    output_fps=4,
)


my16fps_base_path = "./datasets/my_16fps_action/"
my16fps_train_dataset = L(ActionConditionedDataset)(
    train_annotation_path=os.path.join(my16fps_base_path, "annotation/train"),
    val_annotation_path=os.path.join(my16fps_base_path, "annotation/val"),
    test_annotation_path=os.path.join(my16fps_base_path, "annotation/test"),
    video_path=my16fps_base_path,
    sequence_interval=1,
    num_frames=61, # 4*(state_t-1)+1=num_frames; num_frames=61 => state_t=16 (state_t is set in ../experiment/exp.py)
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
    output_fps=16,
)

my16fps_val_dataset = L(ActionConditionedDataset)(
    train_annotation_path=os.path.join(my16fps_base_path, "annotation/train"),
    val_annotation_path=os.path.join(my16fps_base_path, "annotation/val"),
    test_annotation_path=os.path.join(my16fps_base_path, "annotation/test"),
    video_path=my16fps_base_path,
    sequence_interval=1,
    num_frames=61, # 4*(state_t-1)+1=num_frames; num_frames=61 => state_t=16 (state_t is set in ../experiment/exp.py)
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
    output_fps=16,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


bridge_train_dataloader = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)

bridge_val_dataloader = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
)

my16fps_train_dataloader = L(DataLoader)(
    dataset=my16fps_train_dataset,
    sampler=L(get_sampler)(dataset=my16fps_train_dataset),
    batch_size=1,
    drop_last=True,
)

my16fps_val_dataloader = L(DataLoader)(
    dataset=my16fps_val_dataset,
    sampler=L(get_sampler)(dataset=my16fps_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data_action_conditioned():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="my16fps_action_train",
        node=my16fps_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="my16fps_action_val",
        node=my16fps_val_dataloader,
    )
