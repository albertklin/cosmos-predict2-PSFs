# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example dataloader registration for multi-view action dataset."""


from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.multiview_action_conditioned_dataset import (
    MultiviewActionConditionedDataset,
)
from imaginaire.lazy_config import LazyCall as L

# Path to the root of your dataset.
base_path = "./datasets/multiview_action/"

train_dataset = L(MultiviewActionConditionedDataset)(
    dataset_dir=base_path,
    sequence_interval=1,  # sample every frame; >1 down-samples FPS
    num_frames=13,  # total frames per clip = FPS * clip_seconds
    camera_keys=["cam0", "cam1", "cam2", "cam3"],  # names of cameras in dataset
    video_size=[480, 640],  # [H, W] of each view; 480p per view recommended
    camera_to_view_id={"cam0": 0, "cam1": 1, "cam2": 2, "cam3": 3},  # map camera key -> view index
    front_camera_key="cam0",  # camera used for captions (if any)
    state_t=4,  # frames per view fed to model; adjust if num_frames changes
    accumulate_action=False,
    is_train=True,
)

val_dataset = L(MultiviewActionConditionedDataset)(
    dataset_dir=base_path,
    sequence_interval=1,  # match train_dataset
    num_frames=13,  # set to same length as training clips
    camera_keys=["cam0", "cam1", "cam2", "cam3"],
    video_size=[480, 640],
    camera_to_view_id={"cam0": 0, "cam1": 1, "cam2": 2, "cam3": 3},
    front_camera_key="cam0",
    state_t=4,
    accumulate_action=False,
    is_train=False,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


train_loader = L(DataLoader)(
    dataset=train_dataset,
    sampler=L(get_sampler)(dataset=train_dataset),
    batch_size=1,
    drop_last=True,
)

val_loader = L(DataLoader)(
    dataset=val_dataset,
    sampler=L(get_sampler)(dataset=val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data_multiview_action() -> None:
    cs = ConfigStore.instance()
    cs.store(group="dataloader_train", package="dataloader_train", name="multiview_action_train", node=train_loader)
    cs.store(group="dataloader_val", package="dataloader_val", name="multiview_action_val", node=val_loader)