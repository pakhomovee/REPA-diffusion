#!/usr/bin/env bash
# Resume REPA-Σ bloop (EMA-stabilized gradient surgery) on CelebA SiT-B/2
# from step 200k to 400k. GPUs 5-7 (3× A100-40GB).
#
# Phase 2 of bloop training:
#   phase 1: 0→200k (train_repa_sigma_bloop_gpus5_7.sh) — FID=5.89 @ 200k
#   phase 2: 200k→400k (this script) — tests Theorem 1(ii): EMA REPA-Σ
#       should stay competitive vs unbiased baseline at long horizon.
#
# Args are an exact match to the 200k args.json — only --resume-step,
# --max-train-steps and --checkpointing-steps change. batch_size=128
# preserved (43 per GPU on 3 GPUs). sigma-mode=bloop with β=0.99 is the
# same EMA-stabilized variant that yielded 5.89 FID at 200k.
#
# Output: runs/celeba_sit_b2_repa_sigma_bloop_gpus5-7/
#         + new checkpoints/ every 20k steps (220k, 240k, …, 400k)
#         + surgery_stats.csv continued
#
# Note: the bloop EMA tensor g_d_ema is RUNTIME-only (not saved in ckpt) →
# at resume it reinitialises from the first batch's g_d. With β=0.99 the
# effective window is ~100 steps so this stabilises within the first few
# hundred steps — same behaviour as at training start. fp16 inf-poisoning
# guard (isfinite check) and alpha clip remain active per our REPA fixes.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCELERATE="${ACCELERATE:-accelerate}"

cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="5,6,7"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"$ACCELERATE" launch \
  --num_processes 3 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29697}" \
  train_sigma.py \
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
  --exp-name="celeba_sit_b2_repa_sigma_bloop_gpus5-7" \
  --data-dir="../data/celeba256" \
  --resolution=256 \
  --batch-size=128 \
  --max-train-steps=400000 \
  --checkpointing-steps=20000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --resume-step=200000 \
  --num-workers=4 \
  --sigma-mode="bloop" \
  --sigma-bloop-beta=0.99 \
  --sigma-log-every=10
