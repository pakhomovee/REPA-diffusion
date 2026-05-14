#!/usr/bin/env bash
# Single-GPU baseline training script for RTX 3080 (10–12 GB VRAM).
#
# The default global batch is 256 achieved via gradient accumulation:
#   local_batch=32 * accum_steps=8 = effective 256.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_imagenette_baseline_1gpu.sh
#
# Override key knobs via env vars:
#   MAX_TRAIN_STEPS=30000 EXP_NAME=my_run ./scripts/run_imagenette_baseline_1gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# 3080 has 10-12 GB; fp16 SiT-B/2 at bs=32 uses ~7-8 GB leaving headroom.
# Gradient accumulation of 8 gives effective batch = 256.
LOCAL_BATCH="${LOCAL_BATCH:-32}"
ACCUM_STEPS="${ACCUM_STEPS:-8}"

${ACCELERATE:-accelerate} launch \
  --num_processes 1 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29701}" \
  train.py \
  --report-to="${REPORT_TO:-tensorboard}" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes=10 \
  --enc-type="none" \
  --proj-coeff=0.0 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="${LOGGING_DIR:-logs}" \
  --exp-name="${EXP_NAME:-imagenette_sit_b2_baseline_1gpu}" \
  --data-dir="../data/imagenette256-train" \
  --resolution=256 \
  --batch-size="${LOCAL_BATCH}" \
  --gradient-accumulation-steps="${ACCUM_STEPS}" \
  --max-train-steps="${MAX_TRAIN_STEPS:-30000}" \
  --checkpointing-steps="${CHECKPOINTING_STEPS:-5000}" \
  --sampling-steps="${SAMPLING_STEPS:-1000000}" \
  --no-sample-at-step-one \
  --num-workers="${NUM_WORKERS:-2}"
