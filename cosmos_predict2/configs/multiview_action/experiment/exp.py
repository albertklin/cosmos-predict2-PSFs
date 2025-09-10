# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experiment configs for multiview action-conditioned post-training."""

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    --experiment=predict2_multiview_action_2b_training
"""
predict2_multiview_action_2b_training = dict(
    defaults=[
        {"override /model": "predict2_multiview_action_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "multiview_action_train"},
        {"override /dataloader_val": "multiview_action_val"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
        )
    ),
    job=dict(
        group="multiview_action",
        name="predict2_multiview_action_2b_training_${now:%Y-%m-%d}_${now:%H-%M-%S}",
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    dataloader_train=dict(
        batch_size=1,
    ),
)


for _item in [
    predict2_multiview_action_2b_training,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

