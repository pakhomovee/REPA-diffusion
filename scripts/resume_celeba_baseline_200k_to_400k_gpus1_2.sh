#!/usr/bin/env bash
# Resume CelebA baseline from step 200k to 400k on GPUs 1,2.
#
# Phase 3 of baseline training:
#   phase 1: 0→40k (original)
#   phase 2: 40k→200k (resume_celeba_baseline_40k_to_200k_gpus2-4.sh)
#   phase 3: 200k→400k (this script)
#
# Args are an exact match to the 200k args.json — only --resume-step and
# --max-train-steps change. batch_size=126 preserved so per-step gradient
# noise matches the original (63 per GPU on 2 GPUs vs 21 per GPU on 6 GPUs
# at original launch — math identical because DDP all-reduces gradients).
#
# Scientific motivation: testing Theorem 1(i)'s bias claim — baseline must
# eventually catch up with or overtake biased REPA-const.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCELERATE="${ACCELERATE:-accelerate}"

cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="1,2"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"$ACCELERATE" launch \
  --num_processes 2 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29695}" \
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
  --max-train-steps=400000 \
  --checkpointing-steps=20000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --resume-step=200000 \
  --num-workers=4
