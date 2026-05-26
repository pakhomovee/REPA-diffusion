#!/usr/bin/env bash
# End-to-end pipeline: CompCars download → VAE latents (multi-GPU) → train baseline + REPA
#
# Prerequisites:
#   Download CompCars from Kaggle (renancostaalencar/compcars) and extract into
#   data/compcars/  (or set COMPCARS_ROOT).
#
# Usage:
#   bash scripts/pipeline_compcars.sh
#
# Override at call-time:
#   NUM_ENCODE_GPUS=2 MIN_CLASS_SIZE=20 bash scripts/pipeline_compcars.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
PREPROC="$REPO_ROOT/REPA/preprocessing"

NUM_ENCODE_GPUS="${NUM_ENCODE_GPUS:-4}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
MIN_CLASS_SIZE="${MIN_CLASS_SIZE:-10}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
COMPCARS_ROOT="${COMPCARS_ROOT:-$DATA/compcars}"

SOURCE="$DATA/compcars256"
DEST="$DATA/compcars256/vae-sd"

echo "=== [1/3] Exporting CompCars images ==="
python "$SCRIPTS/export_compcars_for_repa.py" \
    --root-dir "$COMPCARS_ROOT" \
    --output-dir "$SOURCE" \
    --resolution 256 \
    --min-class-size "$MIN_CLASS_SIZE"

# Read the number of classes from classes.json written by the export script
NUM_CLASSES=$(python3 -c "
import json; d=json.load(open('$SOURCE/classes.json'))
print(len(d))
")
echo "NUM_CLASSES=$NUM_CLASSES"

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
EXP_NAME=compcars_sit_b2_baseline \
bash "$SCRIPTS/run_compcars_baseline.sh"

echo "--- REPA ---"
NUM_CLASSES=$NUM_CLASSES \
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=compcars_sit_b2_repa_dinov2b \
bash "$SCRIPTS/run_compcars_repa.sh"

echo "=== Pipeline complete ==="
