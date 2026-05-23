#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/REPA"

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS="1"   
export MKL_NUM_THREADS="1"   
export WANDB_MODE="disabled"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

NUM_PROCESSES="2"
LOCAL_BATCH="128"
ACCUM_STEPS="1"
AUGMENT_MODE="flip_only"

# Warm the teacher weights once to avoid multi-process hub races.
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
  --encoder-depth=4 \
  --output-dir="../runs" \
  --logging-dir="logs" \
  --exp-name="imagenette_repa_stable" \
  --data-dir="../data/imagenette256-train" \
  --resolution=256 \
  --batch-size=$(( LOCAL_BATCH * NUM_PROCESSES )) \
  --gradient-accumulation-steps="${ACCUM_STEPS}" \
  --learning-rate=1e-4 \
  --adam-weight-decay=0.05 \
  --max-grad-norm=1.0 \
  --max-train-steps=30000 \
  --checkpointing-steps=10000 \
  --sampling-steps=10000 \
  --num-workers=2 \
  --cfg-prob=0.2 \
  --proj-coeff=0.8 \
  --div-coeff=0.0 \
  --augment \
  --augment-mode="${AUGMENT_MODE}" \
  --sample-cfg-scale=2.0 \
  --no-sample-at-step-one \
  --log-every=100