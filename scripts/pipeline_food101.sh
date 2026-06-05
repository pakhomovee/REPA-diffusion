#!/usr/bin/env bash
# End-to-end pipeline: Food-101 (auto-download) → VAE latents (multi-GPU)
#                       → train baseline + REPA
#
# Food-101 is downloaded automatically via torchvision — no Kaggle account needed.
# 101 classes, 750 training images per class (75,750 total).
#
# Usage:
#   bash scripts/pipeline_food101.sh
#
# Common overrides:
#   # Use both train+test splits for a larger pool (101,000 images)
#   SPLIT=both bash scripts/pipeline_food101.sh
#
#   # Fewer encode GPUs, quick test run
#   NUM_ENCODE_GPUS=1 MAX_TRAIN_STEPS=5000 bash scripts/pipeline_food101.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
PREPROC="$REPO_ROOT/REPA/preprocessing"

# ── tuneable knobs ─────────────────────────────────────────────────────────
RAW_DIR="${RAW_DIR:-$DATA/food101}"           # torchvision download destination
OUTPUT_DIR="${OUTPUT_DIR:-$DATA/food101_256}" # REPA image folder
RESOLUTION="${RESOLUTION:-256}"
SPLIT="${SPLIT:-train}"                       # train | test | both
NUM_ENCODE_GPUS="${NUM_ENCODE_GPUS:-4}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
# ──────────────────────────────────────────────────────────────────────────

DEST="$OUTPUT_DIR/vae-sd"

# ── Step 1: Download + export ──────────────────────────────────────────────
echo "=== [1/3] Exporting Food-101 (split=$SPLIT, resolution=$RESOLUTION) ==="
python "$SCRIPTS/export_food101_for_repa.py" \
    --root-dir    "$RAW_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --resolution  "$RESOLUTION" \
    --split       "$SPLIT"

# ── Step 2: Encode VAE latents (parallel across GPUs) ──────────────────────
echo "=== [2/3] Encoding VAE latents on $NUM_ENCODE_GPUS GPU(s) ==="

pids=()
for i in $(seq 0 $((NUM_ENCODE_GPUS - 1))); do
    python "$PREPROC/dataset_tools.py" encode \
        --source     "$OUTPUT_DIR/images" \
        --dest       "$DEST" \
        --model-url  "$VAE_MODEL" \
        --batch-size "$ENCODE_BATCH_SIZE" \
        --device     "cuda:$i" \
        --rank       "$i" \
        --world-size "$NUM_ENCODE_GPUS" &
    pids+=($!)
    echo "  launched rank $i (PID ${pids[-1]})"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: worker PID $pid failed"
        failed=1
    fi
done
[ "$failed" -eq 0 ] || { echo "One or more encoding workers failed. Aborting."; exit 1; }

if [ "$NUM_ENCODE_GPUS" -gt 1 ]; then
    echo "=== Merging VAE shards ==="
    python "$SCRIPTS/merge_vae_shards.py" \
        --dest       "$DEST" \
        --world-size "$NUM_ENCODE_GPUS"
fi

# ── Step 3: Train ──────────────────────────────────────────────────────────
echo "=== [3/3] Training ==="

echo "--- Baseline (no REPA) ---"
DATA_DIR=$OUTPUT_DIR \
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=food101_sit_b2_baseline \
bash "$SCRIPTS/run_food101_baseline.sh"

echo "--- REPA (DINOv2-B) ---"
DATA_DIR=$OUTPUT_DIR \
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=food101_sit_b2_repa_dinov2b \
bash "$SCRIPTS/run_food101_repa.sh"

echo "=== Pipeline complete ==="
echo "Classes : 101"
echo "Data    : $OUTPUT_DIR"
echo "Runs    : $REPO_ROOT/runs/"
