#!/usr/bin/env bash
# Run LSUN Church REPA (Representation Alignment) training.
# Prerequisites:
#   1. Export images:  python scripts/export_lsunchurch_for_repa.py
#   2. Encode VAE latents:
#        cd REPA/preprocessing
#        python dataset_tools.py encode \
#            --source ../data/lsun_church256 \
#            --dest ../data/lsun_church256/vae-sd \
#            --model-url stabilityai/sd-vae-ft-mse
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NUM_CLASSES="${NUM_CLASSES:-1}"

${ACCELERATE:-accelerate} launch \
  --num_processes "${NUM_PROCESSES:-6}" \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29524}" \
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
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="${LOGGING_DIR:-logs}" \
  --exp-name="${EXP_NAME:-lsunchurch_sit_b2_repa_dinov2b}" \
  --data-dir="../data/lsun_church256" \
  --resolution=256 \
  --batch-size="${BATCH_SIZE:-256}" \
  --max-train-steps="${MAX_TRAIN_STEPS:-100000}" \
  --checkpointing-steps="${CHECKPOINTING_STEPS:-10000}" \
  --sampling-steps="${SAMPLING_STEPS:-1000000}" \
  --no-sample-at-step-one \
  --num-workers="${NUM_WORKERS:-4}"
