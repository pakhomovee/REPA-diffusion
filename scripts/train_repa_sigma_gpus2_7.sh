#!/usr/bin/env bash
# Train REPA-Σ (per-batch t-conditional gradient surgery) on CelebA SiT-B/2
# from scratch to 200k. GPUs 2-7 (6× A100-40GB). Effective batch size kept at
# 128 (same as the constant-λ REPA baseline at
# runs/celeba_sit_b2_repa_dinov2b_gpus4-7) so the FID comparison is
# apples-to-apples; only the distribution across GPUs changes (128//6 = 21
# per rank, total 126 ≈ 128).
#
# Surgery: --sigma-mode=hard
#   At every step, after computing g_d (denoising) and g_r (REPA), project
#   away the component of g_r anti-parallel to g_d on REPA's parameter
#   support (embedders + blocks [0, encoder_depth)). PCGrad applied per
#   batch with the diff direction as the primary axis.
#
# Output: runs/celeba_sit_b2_repa_sigma_hard_gpus2-7/
#         + surgery_stats.csv (per-step telemetry)
#         + checkpoints/ every 10k steps
#
# See experiments/repa_sigma/PLAN.md for the full design rationale.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/seankopylov/projects/repa_diffusion_celeba}"
ACCELERATE="${ACCELERATE:-/home/seankopylov/.venv/bin/accelerate}"

cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

"$ACCELERATE" launch \
  --num_processes 6 \
  --mixed_precision "fp16" \
  --main_process_port "${MAIN_PROCESS_PORT:-29684}" \
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
  --exp-name="celeba_sit_b2_repa_sigma_hard_gpus2-7" \
  --data-dir="../data/celeba256" \
  --resolution=256 \
  --batch-size=128 \
  --max-train-steps=200000 \
  --checkpointing-steps=10000 \
  --sampling-steps=1000000 \
  --no-sample-at-step-one \
  --num-workers=4 \
  --sigma-mode="hard" \
  --sigma-log-every=10
