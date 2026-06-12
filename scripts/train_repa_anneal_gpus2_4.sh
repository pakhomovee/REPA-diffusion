#!/usr/bin/env bash
# REPA + λ-anneal (NO surgery) on CelebA SiT-B/2, from scratch to 200k.
# GPUs 2,3,4. λ linearly annealed proj_coeff(0.5) → 0 over all 200k steps.
#
# This is the "+REPA lambda-anneal" arm of the ablation (gray curve in the toy
# simulation). Tests whether removing the orthogonal teacher bias by annealing
# λ→0 improves final FID over constant-λ REPA.
#
# Effective batch = 128 (3 GPUs × 42 ≈ 126), matching the constant-λ baseline.
# Output: runs/celeba_sit_b2_repa_anneal_gpus2-4/
#         + surgery_stats.csv (per-step lambda_eff + losses)
#         + checkpoints/ every 10k steps
#
# See experiments/repa_sigma/lambda_anneal_analysis.md for rationale.

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
  --main_process_port "${MAIN_PROCESS_PORT:-29692}" \
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
  --exp-name="celeba_sit_b2_repa_anneal_gpus2-4" \
  --data-dir="../data/celeba256" \
  --resolution=256 \
  --batch-size=128 \
  --max-train-steps=200000 \
  --checkpointing-steps=10000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --num-workers=4 \
  --sigma-mode="off" \
  --lambda-anneal \
  --lambda-final=0.0 \
  --sigma-log-every=10
