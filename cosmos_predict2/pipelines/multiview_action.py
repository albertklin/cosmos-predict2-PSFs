# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action conditioned variant of :mod:`multiview` pipeline."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_predict2.pipelines.multiview import MultiviewPipeline
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig


class MultiviewActionPipeline(MultiviewPipeline):
    """Pipeline that accepts multi-view video and corresponding actions."""

    @staticmethod
    def from_config(*args, **kwargs):
        pipe = MultiviewPipeline.from_config(*args, **kwargs)
        pipe.__class__ = MultiviewActionPipeline  # reuse parent initialisation
        return pipe

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
        view_indices: torch.Tensor,
        latent_view_indices: torch.Tensor,
        num_latent_conditional_frames: int = 1,
    ) -> dict[str, Any]:
        """Create a data batch similar to training samples.

        Parameters closely follow
        :meth:`Video2WorldActionConditionedPipeline._get_data_batch_input`
        but additionally require view index tensors used by the
        multi-view model.
        """

        B, C, T, H, W = video.shape
        data_batch: dict[str, Any] = {
            "dataset_name": "video_data",
            "video": video,
            "t5_text_embeddings": torch.zeros(
                B,
                CosmosTextEncoderConfig.NUM_TOKENS,
                CosmosTextEncoderConfig.EMBED_DIM,
                dtype=torch.bfloat16,
                device=video.device,
            ),
            "fps": torch.randint(16, 32, (B,), device=video.device),
            "padding_mask": torch.zeros(B, 1, H, W, device=video.device),
            "num_conditional_frames": num_latent_conditional_frames,
            "action": actions,
            "latent_view_indices_B_T": latent_view_indices,
            "view_indices": view_indices,
        }
        return data_batch