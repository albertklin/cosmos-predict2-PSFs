### Marlowe

`salloc -N 1 -A marlowe-m000154 -p preempt --gpus 8 --mem=256G`

```
apptainer exec --nv \
  --bind /projects/m000154/albert/repos/cosmos-predict2-PSFs:/workspace \
  --bind /projects/m000154/albert/repos/cosmos-predict2-PSFs/datasets:/workspace/datasets \
  --bind /projects/m000154/albert/repos/cosmos-predict2-PSFs/checkpoints:/workspace/checkpoints \
  --bind /projects/m000154/albert:/projects/m000154/albert:rw \
  --env TRITON_CACHE_DIR=/projects/m000154/albert/.triton_cache \
  --env PS1="\[\033[1;34m\]Apptainer>\[\033[0m\] \[\033[1;32m\]$PS1\[\033[0m\]" \
  cosmos-predict2.sif bash
```

`export PYTHONPATH=$(pwd)`

`python /workspace/scripts/test_environment.py`

`PROMPT_="A robot arm picks up the red vegetable and places it in the metal pot."`

```
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/robot.jpg \
    --num_conditional_frames 1 \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_robot.mp4 \
    --natten
```

`PROMPT_="A robot arm knocks over the cereal box. The camera angle is static."`

`PROMPT_="A robotic arm is positioned above a wooden table in a simulated scene, where a white plate and cereal box sit. The arm, equipped with a gripper, moves quickly toward the cereal box and knocks it over. The scene captures a moment of automation in a virtual setting. A medium shot from a slightly elevated angle."`


```
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/cereal.jpg \
    --num_conditional_frames 1 \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_cereal_prompt.mp4 \
    --disable_prompt_refiner \
    --disable_guardrail \
    --natten
```

TEMP

# Set the number of GPUs to use
export NUM_GPUS=8

# Run video2world generation with context parallelism using torchrun
torchrun --nproc_per_node=${NUM_GPUS} examples/video2world.py \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_${NUM_GPUS}gpu.mp4 \
    --num_gpus ${NUM_GPUS} \
    --natten