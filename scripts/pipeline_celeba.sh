#!/usr/bin/env bash
# End-to-end pipeline: CelebA download → VAE latents (multi-GPU) → train baseline + REPA
#
# Usage:
#   bash scripts/pipeline_celeba.sh
#
# Override at call-time:
#   NUM_ENCODE_GPUS=2 SELECTED_ATTRS="Male Smiling" bash scripts/pipeline_celeba.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
PREPROC="$REPO_ROOT/REPA/preprocessing"

NUM_ENCODE_GPUS="${NUM_ENCODE_GPUS:-4}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
SELECTED_ATTRS="${SELECTED_ATTRS:-Male Smiling Young Attractive}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"

SOURCE="$DATA/celeba256"
DEST="$DATA/celeba256/vae-sd"

# Derive NUM_CLASSES = 2 ^ (number of selected attributes)
ATTR_COUNT=$(echo "$SELECTED_ATTRS" | wc -w | tr -d ' ')
NUM_CLASSES=$(python3 -c "print(2**$ATTR_COUNT)")
echo "Selected attrs: $SELECTED_ATTRS  →  $NUM_CLASSES classes"

echo "=== [1/3] Exporting CelebA images ==="
# shellcheck disable=SC2086
python "$SCRIPTS/export_celeba_for_repa.py" \
    --root-dir "$DATA/celeba" \
    --output-dir "$SOURCE" \
    --selected-attrs $SELECTED_ATTRS \
    --resolution 256

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
NUM_CLASSES=$NUM_CLASSES \
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=celeba_sit_b2_baseline \
bash "$SCRIPTS/run_celeba_baseline.sh"

echo "--- REPA ---"
NUM_CLASSES=$NUM_CLASSES \
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=celeba_sit_b2_repa_dinov2b \
bash "$SCRIPTS/run_celeba_repa.sh"

echo "=== Pipeline complete ==="
