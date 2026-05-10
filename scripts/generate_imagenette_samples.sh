#!/usr/bin/env bash
set -euo pipefail

kind="${1:-repa}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

num_samples="${NUM_SAMPLES:-64}"
num_steps="${NUM_STEPS:-250}"

case "$kind" in
  baseline)
    ckpt="${CKPT:-../runs/tb_imagenette_sit_b2_baseline_1k/checkpoints/0001000.pt}"
    projector_embed_dims="${PROJECTOR_EMBED_DIMS:-none}"
    sample_dir="${SAMPLE_DIR:-../samples/imagenette_baseline_1k_${num_samples}s_${num_steps}steps}"
    ;;
  repa)
    ckpt="${CKPT:-../runs/tb_imagenette_sit_b2_repa_dinov2b_1k/checkpoints/0001000.pt}"
    projector_embed_dims="${PROJECTOR_EMBED_DIMS:-768}"
    sample_dir="${SAMPLE_DIR:-../samples/imagenette_repa_dinov2b_1k_${num_samples}s_${num_steps}steps}"
    ;;
  *)
    echo "Usage: $0 [baseline|repa]" >&2
    exit 2
    ;;
esac

${TORCHRUN:-torchrun} \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_PROCESSES:-1}" \
  generate.py \
  --model="SiT-B/2" \
  --num-classes=10 \
  --ckpt="$ckpt" \
  --weights="${WEIGHTS:-ema}" \
  --sample-dir="$sample_dir" \
  --resolution=256 \
  --vae=mse \
  --path-type=linear \
  --encoder-depth=4 \
  --projector-embed-dims="$projector_embed_dims" \
  --per-proc-batch-size="${PER_PROC_BATCH_SIZE:-16}" \
  --num-fid-samples="$num_samples" \
  --mode="${MODE:-sde}" \
  --num-steps="$num_steps" \
  --cfg-scale="${CFG_SCALE:-1.0}" \
  --guidance-high="${GUIDANCE_HIGH:-1.0}"
