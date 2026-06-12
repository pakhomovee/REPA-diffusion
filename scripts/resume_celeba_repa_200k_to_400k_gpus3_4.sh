#!/usr/bin/env bash
# Resume CelebA REPA const-λ (DINOv2-B teacher) from step 200k to 400k on GPUs 3,4.
#
# Phase 3 of REPA-const training:
#   phase 1: 0→80k (original)
#   phase 2: 80k→200k (resume_celeba_repa_80k_to_200k_gpus1_5_6_7.sh)
#   phase 3: 200k→400k (this script)
#
# Args are an exact match to the 200k args.json — only --resume-step and
# --max-train-steps change. batch_size=128 preserved (64 per GPU on 2 GPUs).
# DINOv2-B teacher adds ~6 GB memory per rank; at 64-sample per-rank batch we
# expect ~22 GB peak, comfortably within 40 GB.
#
# Scientific motivation: testing Theorem 1(i)'s bias claim — biased REPA-const
# should plateau or be overtaken by unbiased baseline / REPA-Σ at long horizon.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCELERATE="${ACCELERATE:-accelerate}"

cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="3,4"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"$ACCELERATE" launch \
  --num_processes 2 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29696}" \
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
  --max-train-steps=400000 \
  --checkpointing-steps=20000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --resume-step=200000 \
  --num-workers=4
