#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

${ACCELERATE:-accelerate} launch \
  --num_processes "${NUM_PROCESSES:-6}" \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29521}" \
  train.py \
  --report-to="${REPORT_TO:-tensorboard}" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes=196 \
  --enc-type="none" \
  --proj-coeff=0.0 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="${LOGGING_DIR:-logs}" \
  --exp-name="${EXP_NAME:-stanford_cars_sit_b2_baseline}" \
  --data-dir="../data/stanford_cars256-train" \
  --resolution=256 \
  --batch-size="${BATCH_SIZE:-256}" \
  --max-train-steps="${MAX_TRAIN_STEPS:-1000}" \
  --checkpointing-steps="${CHECKPOINTING_STEPS:-500}" \
  --sampling-steps="${SAMPLING_STEPS:-1000000}" \
  --no-sample-at-step-one \
  --num-workers="${NUM_WORKERS:-4}"
