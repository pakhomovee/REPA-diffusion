#!/usr/bin/env bash
# Train REPA-Σ (EMA-stabilized gradient surgery, "bloop" mode) on CelebA SiT-B/2
# from scratch to 200k. GPUs 5-7 (3× A100-40GB). Effective batch size kept at 128
# (same as the constant-λ REPA hard run at
# runs/celeba_sit_b2_repa_sigma_hard_gpus2-7) so the FID comparison is
# apples-to-apples — only the reference direction for projection differs.
#
# Surgery: --sigma-mode=bloop with --sigma-bloop-beta=0.99
#   Maintains an exponential moving average of the diffusion gradient
#       v_k = β·v_{k-1} + (1−β)·g_d(B_k),  β = 0.99
#   and projects g_r onto the half-space ⟨·, v_k⟩ ≥ 0 (PCGrad against the
#   smoothed direction). This matches the EMA-stabilized variant analyzed in
#   §5.1.1 of the report (Theorem 1) — companion to the per-batch hard run.
#
# Output: runs/celeba_sit_b2_repa_sigma_bloop_gpus5-7/
#         + surgery_stats.csv (per-step telemetry)
#         + checkpoints/ every 10k steps

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
  --main_process_port "${MAIN_PROCESS_PORT:-29694}" \
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
  --max-train-steps=200000 \
  --checkpointing-steps=10000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --num-workers=4 \
  --sigma-mode="bloop" \
  --sigma-bloop-beta=0.99 \
  --sigma-log-every=10
