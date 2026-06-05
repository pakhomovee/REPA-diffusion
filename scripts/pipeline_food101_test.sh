#!/usr/bin/env bash
# Smoke-test pipeline for Food-101.
# Runs every step (export → encode → train) on a tiny subset so you can
# verify the full pipeline works before committing to a full run.
#
# What this does differently from pipeline_food101.sh:
#   - Exports only the first MAX_IMAGES_PER_CLASS images per class (default 5)
#   - Encodes on a single GPU
#   - Trains for only 100 steps with no sampling
#   - Writes to data/food101_256_test/ and runs/food101_test_* so it does
#     not overwrite any real experiment data
#
# Usage:
#   bash scripts/pipeline_food101_test.sh
#
# Expected runtime: ~3-5 minutes on a single GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
PREPROC="$REPO_ROOT/REPA/preprocessing"

# ── test knobs (intentionally tiny) ───────────────────────────────────────
RAW_DIR="${RAW_DIR:-$DATA/food101}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA/food101_256_test}"
RESOLUTION="${RESOLUTION:-256}"
SPLIT="${SPLIT:-train}"
MAX_IMAGES_PER_CLASS="${MAX_IMAGES_PER_CLASS:-5}"   # 5 × 101 = 505 images total
ENCODE_GPU="${ENCODE_GPU:-0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-8}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# ──────────────────────────────────────────────────────────────────────────

DEST="$OUTPUT_DIR/vae-sd"

echo "╔══════════════════════════════════════════════════╗"
echo "║  Food-101 SMOKE TEST                             ║"
echo "║  ${MAX_IMAGES_PER_CLASS} images/class × 101 classes = $(( MAX_IMAGES_PER_CLASS * 101 )) images  ║"
echo "║  Train steps : $TRAIN_STEPS                             ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Step 1: Export (tiny subset) ──────────────────────────────────────────
echo ""
echo "=== [1/3] Exporting Food-101 test subset ==="
python "$SCRIPTS/export_food101_for_repa.py" \
    --root-dir   "$RAW_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --resolution "$RESOLUTION" \
    --split      "$SPLIT" \
    --max-images-per-class "$MAX_IMAGES_PER_CLASS"

# ── Step 2: Encode VAE latents (single GPU) ────────────────────────────────
echo ""
echo "=== [2/3] Encoding VAE latents (single GPU: cuda:$ENCODE_GPU) ==="
python "$PREPROC/dataset_tools.py" encode \
    --source     "$OUTPUT_DIR" \
    --dest       "$DEST" \
    --model-url  "$VAE_MODEL" \
    --batch-size "$ENCODE_BATCH_SIZE" \
    --device     "cuda:$ENCODE_GPU"

# ── Step 3: Train (minimal steps, both baseline and REPA) ─────────────────
echo ""
echo "=== [3/3] Training (smoke test: $TRAIN_STEPS steps) ==="

cd "$REPO_ROOT/REPA"

echo "--- Baseline ---"
${ACCELERATE:-accelerate} launch \
  --num_processes 1 \
  --mixed_precision "fp16" \
  --main_process_port 29531 \
  train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes=101 \
  --enc-type="none" \
  --proj-coeff=0.0 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --exp-name="food101_test_baseline" \
  --data-dir="$DEST" \
  --resolution=256 \
  --batch-size="$BATCH_SIZE" \
  --max-train-steps="$TRAIN_STEPS" \
  --checkpointing-steps="$TRAIN_STEPS" \
  --sampling-steps=999999 \
  --no-sample-at-step-one \
  --num-workers=2

echo "--- REPA ---"
${ACCELERATE:-accelerate} launch \
  --num_processes 1 \
  --mixed_precision "fp16" \
  --main_process_port 29532 \
  train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes=101 \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="../runs" \
  --exp-name="food101_test_repa" \
  --data-dir="$DEST" \
  --resolution=256 \
  --batch-size="$BATCH_SIZE" \
  --max-train-steps="$TRAIN_STEPS" \
  --checkpointing-steps="$TRAIN_STEPS" \
  --sampling-steps=999999 \
  --no-sample-at-step-one \
  --num-workers=2

echo ""
echo "=== Smoke test PASSED ==="
echo "Outputs:"
echo "  Images  : $OUTPUT_DIR"
echo "  Latents : $DEST"
echo "  Runs    : $REPO_ROOT/runs/food101_test_*"
