#!/usr/bin/env bash
# Run CompCars baseline (no REPA encoder alignment) training.
# Prerequisites:
#   1. Export images:
#        python scripts/export_compcars_for_repa.py
#      Note the printed --num-classes value and set NUM_CLASSES below.
#   2. Encode VAE latents:
#        cd REPA/preprocessing
#        python dataset_tools.py encode \
#            --source ../data/compcars256 \
#            --dest ../data/compcars256/vae-sd \
#            --model-url stabilityai/sd-vae-ft-mse
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Set to the number printed by export_compcars_for_repa.py
# (make x model classes; ~1600 with default --min-class-size=10).
NUM_CLASSES="${NUM_CLASSES:-1600}"

${ACCELERATE:-accelerate} launch \
  --num_processes "${NUM_PROCESSES:-6}" \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29525}" \
  train.py \
  --report-to="${REPORT_TO:-tensorboard}" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes="${NUM_CLASSES}" \
  --enc-type="none" \
  --proj-coeff=0.0 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="${LOGGING_DIR:-logs}" \
  --exp-name="${EXP_NAME:-compcars_sit_b2_baseline}" \
  --data-dir="../data/compcars256" \
  --resolution=256 \
  --batch-size="${BATCH_SIZE:-256}" \
  --max-train-steps="${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing-steps="${CHECKPOINTING_STEPS:-10000}" \
  --sampling-steps="${SAMPLING_STEPS:-1000000}" \
  --no-sample-at-step-one \
  --num-workers="${NUM_WORKERS:-4}"
