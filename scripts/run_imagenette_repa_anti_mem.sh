#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export WANDB_MODE="disabled"

NUM_PROCESSES="2"
LOCAL_BATCH="64"
ACCUM_STEPS="2"

# ---------------------------------------------------------------------------
# Augmentation mode — choose one:
#   "vae"       correct: re-encodes with VAE after crop+flip+jitter (default)
#   "flip_only" fast fallback: horizontal flip only, no extra VAE pass
#               → use on Colab free tier if VAE re-encoding is too slow
# ---------------------------------------------------------------------------
AUGMENT_MODE="vae"

# Pre-download DINOv2-Small so multi-GPU doesn't race on extract
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"

accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  --mixed_precision "fp16" \
  --main_process_port "29502" \
  train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B-shallow/2" \
  --num-classes=10 \
  --enc-type="dinov2-vit-s" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="logs" \
  --exp-name="imagenette_repa_smart_anti_overfit" \
  --data-dir="../data/imagenette256-train" \
  --resolution=256 \
  --batch-size=$(( LOCAL_BATCH * NUM_PROCESSES )) \
  --gradient-accumulation-steps="${ACCUM_STEPS}" \
  --max-train-steps=15000 \
  --checkpointing-steps=3000 \
  --sampling-steps=3000 \
  --no-sample-at-step-one \
  --num-workers=4 \
  --augment \
  --augment-mode="${AUGMENT_MODE}" \
  --div-coeff=0.01 \
  --sample-cfg-scale=2.0 \
  --adam-weight-decay=0.03 \
  --cfg-prob=0.2