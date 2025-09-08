# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hydra registration for multi-view action conditioned model."""

from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.multiview_action.config import (
    get_cosmos_predict2_multiview_action_pipeline,
)
from cosmos_predict2.models.multiview_action_model import (
    Predict2MultiviewActionConditionedModel,
    Predict2MultiviewModelConfig,
)
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig
from imaginaire.constants import get_cosmos_predict2_multiview_checkpoint
from imaginaire.lazy_config import LazyCall as L

_PREDICT2_MULTIVIEW_ACTION_MODEL = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(Predict2MultiviewActionConditionedModel)(
        config=Predict2MultiviewModelConfig(
            pipe_config=get_cosmos_predict2_multiview_action_pipeline(model_size="2B"),
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path=get_cosmos_predict2_multiview_checkpoint(model_size="2B"),
                text_encoder_path="",
            ),
            fsdp_shard_size=-1,
        ),
        _recursive_=False,
    ),
)


def register_model_multiview_action() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="predict2_multiview_action_2b_fsdp",
        node=_PREDICT2_MULTIVIEW_ACTION_MODEL,
    )