# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example for running the multi-view action conditioned pipeline."""

import torch

from cosmos_predict2.configs.multiview_action.config import (
    get_cosmos_predict2_multiview_action_pipeline,
)
from cosmos_predict2.pipelines.multiview_action import MultiviewActionPipeline


def main() -> None:
    pipe = MultiviewActionPipeline.from_config(
        get_cosmos_predict2_multiview_action_pipeline(model_size="2B"), use_text_encoder=False
    )
    # dummy four-view clip with four frames per view
    video = torch.zeros(1, 3, 16, 64, 64)
    actions = torch.zeros(1, 15, 7)
    view_indices = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    latent_view_indices = torch.tensor([0, 1, 2, 3])
    batch = pipe._get_data_batch_input(video, actions, view_indices, latent_view_indices)
    print(batch.keys())


if __name__ == "__main__":
    main()