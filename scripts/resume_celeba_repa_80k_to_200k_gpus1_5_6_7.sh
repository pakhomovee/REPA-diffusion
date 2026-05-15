#!/usr/bin/env bash
# Resume CelebA REPA from step 80k to 200k on GPUs 1,5,6,7.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/seankopylov/projects/repa_diffusion_celeba}"
ACCELERATE="${ACCELERATE:-/home/seankopylov/.venv/bin/accelerate}"

cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="1,5,6,7"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"$ACCELERATE" launch \
  --num_processes 4 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29682}" \
  train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes=16 \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="logs" \
  --exp-name="celeba_sit_b2_repa_dinov2b_gpus4-7" \
  --data-dir="../data/celeba256" \
  --resolution=256 \
  --batch-size=128 \
  --max-train-steps=200000 \
  --checkpointing-steps=10000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --resume-step=80000 \
  --num-workers=4
