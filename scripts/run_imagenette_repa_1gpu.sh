#!/usr/bin/env bash
# Single-GPU REPA training script for RTX 3080 (10–12 GB VRAM).
#
# DINOv2-B teacher adds ~1.5 GB VRAM on top of the diffusion model, so the
# local batch is reduced to 16 with gradient accumulation of 16 to keep the
# effective global batch at 256.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_imagenette_repa_1gpu.sh
#
# Override key knobs via env vars:
#   MAX_TRAIN_STEPS=100000 EXP_NAME=my_repa_run ./scripts/run_imagenette_repa_1gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# DINOv2-B takes ~1.5 GB; use bs=16 + accum=16 to keep effective batch = 256.
# If OOM, reduce LOCAL_BATCH to 8 and increase ACCUM_STEPS to 32.
LOCAL_BATCH="${LOCAL_BATCH:-16}"
ACCUM_STEPS="${ACCUM_STEPS:-16}"

${ACCELERATE:-accelerate} launch \
  --num_processes 1 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29702}" \
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
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="${LOGGING_DIR:-logs}" \
  --exp-name="${EXP_NAME:-imagenette_sit_b2_repa_dinov2b_1gpu}" \
  --data-dir="../data/imagenette256-train" \
  --resolution=256 \
  --batch-size="${LOCAL_BATCH}" \
  --gradient-accumulation-steps="${ACCUM_STEPS}" \
  --max-train-steps="${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing-steps="${CHECKPOINTING_STEPS:-10000}" \
  --sampling-steps="${SAMPLING_STEPS:-1000000}" \
  --no-sample-at-step-one \
  --num-workers="${NUM_WORKERS:-2}"
