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

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

"""
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment="predict2_video2world_2b_action_conditioned_training"
"""
predict2_video2world_2b_action_conditioned_training = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_action_conditioned_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "bridge_train"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
        )
    ),
    job=dict(group="debug", name="predict2_video2world_2b_action_conditioned_training_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=2,
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
)

predict2_video2world_14b_action_conditioned_training_my16fps = dict(
    defaults=[
        {"override /model": "predict2_v2w_14b_action_conditioned_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "my16fps_action_train"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=8,
            pipe_config=dict(
                state_t=16, # corresponds to num_frames=61; (16-1)*4 + 1 = 61; (num_frames is set in ../defaults/data.py)
                resize_online=False,
                net=dict(
                    action_dim=7 * 60, # 7 * (num_frames-1)
                ),
            ),
        )
    ),
    job=dict(
        group="action_conditioned",
        name="predict2_video2world_14b_action_conditioned_my16fps_${now:%Y-%m-%d}_${now:%H-%M-%S}",
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dict(
        batch_size=1,
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    checkpoint=dict(
        save_iter=1000,
    ),
)


for _item in [
    # predict2_video2world_2b
    predict2_video2world_2b_action_conditioned_training,
    predict2_video2world_14b_action_conditioned_training_my16fps,
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
