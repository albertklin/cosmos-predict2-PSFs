# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-view dataset with action annotations.

This dataset merges the logic of :mod:`dataset_multiview` and the
:mod:`action_conditioned_dataset`. It loads synchronized frames from
multiple camera views and computes per-step actions from the robot
state annotations. The resulting sample mirrors the structure returned
by the individual datasets so that existing training scripts can be
re-used.

The implementation is intentionally close to the reference datasets to
make verification easier. ``MultiviewDataset`` is responsible for
stacking frames from different cameras and returning ``view_indices``
while ``ActionConditionedDataset`` handles reading the robot states and
converting them into relative actions. Here we simply combine the two.
"""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig


class MultiviewActionConditionedDataset(Dataset):
    """Dataset returning multi-view clips and per-frame actions."""

    def __init__(
        self,
        dataset_dir: str,
        sequence_interval: int,
        num_frames: int,
        camera_keys: list[str],
        video_size: list[int],
        start_frame_interval: int = 1,
        state_t: int = 8,
        camera_to_view_id: dict[str, int] | None = None,
        front_camera_key: str | None = None,
        front_view_caption_only: bool = False,
        accumulate_action: bool = False,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.start_frame_interval = start_frame_interval
        self.sequence_interval = sequence_interval
        self.sequence_length = num_frames
        self.camera_keys = camera_keys
        self.state_t = state_t
        self.H, self.W = video_size
        self.front_view_caption_only = front_view_caption_only
        self.camera_to_view_id = camera_to_view_id or {}
        self.front_camera_key = front_camera_key
        self.accumulate_action = accumulate_action

        video_dir = os.path.join(self.dataset_dir, "videos", camera_keys[0])
        self.ann_dir = os.path.join(self.dataset_dir, "annotations")
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_paths = sorted(self.video_paths)

        cutoff = int(len(self.video_paths) * 0.1) + 1
        if is_train:
            self.video_paths = self.video_paths[:-cutoff]
        else:
            self.video_paths = self.video_paths[-cutoff:]

        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")
        self.samples = self._init_samples(self.video_paths)
        self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))

        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])
        self.wrong_number = 0

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------
    def _init_samples(self, video_paths: list[str]) -> list[dict[str, Any]]:
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_video_path = {
                executor.submit(self._load_and_process_video_path, video_path): video_path
                for video_path in video_paths
            }
            for future in tqdm(as_completed(future_to_video_path), total=len(video_paths)):
                samples.extend(future.result())
        return samples

    def _load_and_process_video_path(self, video_path: str) -> list[dict[str, Any]]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)
        base = os.path.splitext(os.path.basename(video_path))[0]
        ann_path = os.path.join(self.ann_dir, base + ".json")

        samples = []
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = {"video_path": video_path, "ann_file": ann_path, "frame_ids": []}
            curr = frame_i
            while True:
                if curr > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr += self.sequence_interval
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    # ------------------------------------------------------------------
    # video helpers
    # ------------------------------------------------------------------
    def _load_video(self, video_path: str, frame_ids: list[int]) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # pragma: no cover - decord may fail to read fps
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path: str, frame_ids: list[int]) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path, frame_ids)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    # ------------------------------------------------------------------
    # action helpers copied from ActionConditionedDataset
    # ------------------------------------------------------------------
    def _get_robot_states(self, label: dict[str, Any], frame_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        all_states = np.array(label["state"])
        all_cont_gripper_states = np.array(label["continuous_gripper_state"])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        return arm_states, cont_gripper_states

    def _get_actions(
        self, arm_states: np.ndarray, gripper_states: np.ndarray, accumulate_action: bool
    ) -> torch.Tensor:
        action_dim = 7
        action = np.zeros((self.sequence_length - 1, action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = gripper_states[k]
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = gripper_states[k]
        return torch.from_numpy(action)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        video_path = sample["video_path"]
        frame_ids = sample["frame_ids"]
        ann_file = sample["ann_file"]

        with open(ann_file) as f:
            label = json.load(f)
        arm_states, gripper_states = self._get_robot_states(label, frame_ids)
        actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
        actions *= 20.0  # scale matches ActionConditionedDataset.c_act_scaler

        videos = []
        t5_embeddings = []
        t5_masks = []
        camera_keys_selection = self.camera_keys
        view_indices_selection = [self.camera_to_view_id[c] for c in camera_keys_selection]
        view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.sequence_length)
        latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)

        for camera_key in camera_keys_selection:
            vpath = os.path.join(os.path.dirname(os.path.dirname(video_path)), camera_key, os.path.basename(video_path))
            video, fps = self._get_frames(vpath, frame_ids)
            video = video.permute(1, 0, 2, 3)
            videos.append(video)

            if camera_key == self.front_camera_key or not self.front_view_caption_only:
                t5_path = os.path.join(self.t5_dir, camera_key, os.path.basename(video_path).replace(".mp4", ".pkl"))
                if os.path.exists(t5_path):
                    with open(t5_path, "rb") as f:
                        t5_embedding = torch.from_numpy(pickle.load(f)[0])
                else:
                    t5_embedding = torch.zeros(1, CosmosTextEncoderConfig.EMBED_DIM)
            else:
                t5_embedding = torch.zeros(1, CosmosTextEncoderConfig.EMBED_DIM)
            t5_mask = torch.ones(t5_embedding.shape[0], dtype=torch.int64)
            if t5_embedding.shape[0] < CosmosTextEncoderConfig.NUM_TOKENS:
                pad = CosmosTextEncoderConfig.NUM_TOKENS - t5_embedding.shape[0]
                t5_embedding = torch.cat(
                    [t5_embedding, torch.zeros(pad, CosmosTextEncoderConfig.EMBED_DIM)], dim=0
                )
                t5_mask = torch.cat([t5_mask, torch.zeros(pad)], dim=0)
            else:
                t5_embedding = t5_embedding[: CosmosTextEncoderConfig.NUM_TOKENS]
                t5_mask = t5_mask[: CosmosTextEncoderConfig.NUM_TOKENS]
            t5_embeddings.append(t5_embedding)
            t5_masks.append(t5_mask)

        video = torch.cat(videos, dim=1)
        t5_embedding = torch.cat(t5_embeddings, dim=0)

        data: dict[str, Any] = {}
        data["video"] = video
        data["video_name"] = {"video_path": video_path, "start_frame_id": str(frame_ids[0])}
        data["t5_text_embeddings"] = t5_embedding
        data["t5_text_mask"] = torch.cat(t5_masks)
        data["fps"] = fps
        data["image_size"] = torch.tensor([self.H, self.W, self.H, self.W])
        data["num_frames"] = self.sequence_length
        data["sample_n_views"] = len(camera_keys_selection)
        data["padding_mask"] = torch.zeros(1, self.H, self.W)
        data["view_indices"] = view_indices_t.contiguous()
        data["latent_view_indices_B_T"] = latent_view_indices_t.contiguous()
        data["ref_cam_view_idx_sample_position"] = torch.ones(1, dtype=torch.int64) * (-1)
        data["action"] = actions.float()
        return data