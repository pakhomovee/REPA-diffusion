#!/usr/bin/env bash
# Single-GPU REPA training script for RTX 3080 (10–12 GB VRAM)
# UPDATED: Includes anti-memorization regularizations (Shallow model, Augmentations, Diversity Loss)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

LOCAL_BATCH="${LOCAL_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-32}"

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
--model="SiT-B-shallow/2" \
--num-classes=10 \
--enc-type="dinov2-vit-b" \
--proj-coeff=0.5 \
--encoder-depth=4 \
--output-dir="../runs" \
--logging-dir="${LOGGING_DIR:-logs}" \
--exp-name="${EXP_NAME:-imagenette_repa_anti_mem_run}" \
--data-dir="../data/imagenette256-train" \
--resolution=256 \
--batch-size="${LOCAL_BATCH}" \
--gradient-accumulation-steps="${ACCUM_STEPS}" \
--max-train-steps="${MAX_TRAIN_STEPS:-30000}" \
--checkpointing-steps="${CHECKPOINTING_STEPS:-10000}" \
--sampling-steps="${SAMPLING_STEPS:-10000}" \
--no-sample-at-step-one \
--num-workers="${NUM_WORKERS:-2}" \
--augment \
--div-coeff=0.1 \
--sample-cfg-scale=2.0