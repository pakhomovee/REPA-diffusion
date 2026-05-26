#!/usr/bin/env bash
# End-to-end pipeline: LSUN Church .npy → VAE latents (multi-GPU) → train baseline + REPA
#
# Prerequisites:
#   Download church_outdoor_train_lmdb_color_64.npy from Kaggle and place it in
#   data/lsun_church/  (or set LSUN_ROOT to the directory containing it).
#
# Usage:
#   bash scripts/pipeline_lsunchurch.sh
#
# Override at call-time:
#   NUM_ENCODE_GPUS=2 MAX_IMAGES=50000 bash scripts/pipeline_lsunchurch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
PREPROC="$REPO_ROOT/REPA/preprocessing"

NUM_ENCODE_GPUS="${NUM_ENCODE_GPUS:-4}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
MAX_IMAGES="${MAX_IMAGES:-}"          # leave empty to export all ~126k images
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
LSUN_ROOT="${LSUN_ROOT:-$DATA/lsun_church}"

SOURCE="$DATA/lsun_church256"
DEST="$DATA/lsun_church256/vae-sd"

echo "=== [1/3] Exporting LSUN Church images from .npy ==="
MAX_IMAGES_ARG=""
if [ -n "$MAX_IMAGES" ]; then
    MAX_IMAGES_ARG="--max-images $MAX_IMAGES"
fi
# shellcheck disable=SC2086
python "$SCRIPTS/export_lsunchurch_for_repa.py" \
    --root-dir "$LSUN_ROOT" \
    --output-dir "$SOURCE" \
    --resolution 256 \
    $MAX_IMAGES_ARG

echo "=== [2/3] Encoding VAE latents on $NUM_ENCODE_GPUS GPU(s) ==="
pids=()
for i in $(seq 0 $((NUM_ENCODE_GPUS - 1))); do
    python "$PREPROC/dataset_tools.py" encode \
        --source "$SOURCE" \
        --dest   "$DEST" \
        --model-url "$VAE_MODEL" \
        --batch-size "$ENCODE_BATCH_SIZE" \
        --device "cuda:$i" \
        --rank "$i" \
        --world-size "$NUM_ENCODE_GPUS" &
    pids+=($!)
    echo "  launched rank $i (PID ${pids[-1]})"
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [ "$NUM_ENCODE_GPUS" -gt 1 ]; then
    echo "=== Merging VAE shards ==="
    python "$SCRIPTS/merge_vae_shards.py" --dest "$DEST" --world-size "$NUM_ENCODE_GPUS"
fi

echo "=== [3/3] Training ==="
echo "--- Baseline ---"
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=lsunchurch_sit_b2_baseline \
bash "$SCRIPTS/run_lsunchurch_baseline.sh"

echo "--- REPA ---"
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=lsunchurch_sit_b2_repa_dinov2b \
bash "$SCRIPTS/run_lsunchurch_repa.sh"

echo "=== Pipeline complete ==="
