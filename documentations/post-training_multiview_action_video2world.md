# Multiview Action-Conditioned Video2World Post-training

This guide describes how to fine-tune a **Cosmos-Predict2** model that takes
multi-view video along with robot actions and predicts future frames. It
combines the data preparation steps from the multiview tutorial
([`post-training_multiview_waymo.md`](post-training_multiview_waymo.md)) and the
action-conditioned tutorial
([`post-training_video2world_action.md`](post-training_video2world_action.md)).

## Prerequisites

Before starting ensure that:

1. The environment is set up following [setup.md](setup.md) and the required
   checkpoints have been downloaded.
2. You have multi-camera videos with corresponding action annotations. The
   dataset folder layout expected by
   `MultiviewActionConditionedDataset` is shown below.

```
datasets/multiview_action/
├── annotations/           # one <video>.json per clip with state and gripper data
├── videos/
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   └── cam3/
```

Each annotation file stores the robot end-effector pose, gripper width and the
per-step actions, identical to the format used for the Bridge dataset in
[`post-training_video2world_action.md`](post-training_video2world_action.md).

This pipeline ignores text prompts, so no T5 embeddings need to be
pre-computed.

## Post-training

The project registers a training configuration for multiview action-conditioned
fine-tuning in
`cosmos_predict2/configs/multiview_action/experiment/exp.py`. Launch training
with:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --experiment=predict2_multiview_action_2b_training
```

Checkpoints are saved to `checkpoints/PROJECT/GROUP/NAME` where `GROUP` is
`multiview_action` by default.

## Inference

Use the provided example script to sample from a fine-tuned checkpoint:

```bash
python examples/multiview_action.py \
  --model_size 2B \
  --dit_path checkpoints/posttraining/multiview_action/2b/checkpoints/model/iter_000001000.pt \
  --input_path datasets/multiview_action/videos/cam0/example.mp4 \
  --annotation datasets/multiview_action/annotations/example.json \
  --num_conditional_frames 1 \
  --save_path output/multiview_action_result.mp4 \
  --disable_guardrail
```

Adjust the paths, number of GPUs and other flags as necessary. For additional
inference options see the documentation in
[`inference_multiview.md`](inference_multiview.md).

