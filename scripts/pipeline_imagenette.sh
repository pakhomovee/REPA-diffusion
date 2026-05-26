#!/usr/bin/env bash
# End-to-end pipeline: Imagenette → VAE latents (multi-GPU) → train baseline + REPA
#
# Usage:
#   bash scripts/pipeline_imagenette.sh
#
# Override any variable at call-time:
#   NUM_ENCODE_GPUS=2 MAX_TRAIN_STEPS=50000 bash scripts/pipeline_imagenette.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
PREPROC="$REPO_ROOT/REPA/preprocessing"

# ── tuneable knobs ─────────────────────────────────────────────────────────────
NUM_ENCODE_GPUS="${NUM_ENCODE_GPUS:-4}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
# ──────────────────────────────────────────────────────────────────────────────

SOURCE="$DATA/imagenette256-train"
DEST="$DATA/imagenette256-train/vae-sd"

echo "=== [1/4] Downloading Imagenette ==="
if [ ! -d "$DATA/raw/imagenette2-320" ]; then
    mkdir -p "$DATA/downloads"
    wget -q --show-progress -P "$DATA/downloads" \
        https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    mkdir -p "$DATA/raw"
    tar -xzf "$DATA/downloads/imagenette2-320.tgz" -C "$DATA/raw/"
fi

echo "=== [2/4] Exporting images to REPA folder format ==="
python "$PREPROC/encoders.py" \
    --data_path "$DATA/raw/imagenette2-320/train" \
    --output_path "$SOURCE" \
    --resolution 256

echo "=== [3/4] Encoding VAE latents on $NUM_ENCODE_GPUS GPU(s) ==="
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

echo "=== [4/4] Training ==="
echo "--- Baseline ---"
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=imagenette_sit_b2_baseline \
bash "$SCRIPTS/run_imagenette_baseline.sh"

echo "--- REPA ---"
MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
EXP_NAME=imagenette_sit_b2_repa_dinov2b \
bash "$SCRIPTS/run_imagenette_repa.sh"

echo "=== Pipeline complete ==="
