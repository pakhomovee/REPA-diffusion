#!/usr/bin/env bash
# Resume CelebA baseline from step 40k to 200k on GPUs 2,3,4.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCELERATE="${ACCELERATE:-accelerate}"

cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="2,3,4"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"$ACCELERATE" launch \
  --num_processes 3 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29681}" \
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
  --enc-type="none" \
  --proj-coeff=0.0 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="logs" \
  --exp-name="celeba_sit_b2_baseline_gpus2-7_40k" \
  --data-dir="../data/celeba256" \
  --resolution=256 \
  --batch-size=126 \
  --max-train-steps=200000 \
  --checkpointing-steps=10000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --resume-step=40000 \
  --num-workers=4
