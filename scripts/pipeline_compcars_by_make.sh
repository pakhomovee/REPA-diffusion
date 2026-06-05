    #!/usr/bin/env bash
    # End-to-end pipeline: CompCars → manufacturer-level classes
    #                       → VAE latents (multi-GPU) → train baseline + REPA
    #
    # Classes are assigned at the make (manufacturer) level, e.g. Toyota, BMW.
    # This gives ~100–163 classes depending on --min-class-size.
    #
    # Prerequisites:
    #   Download CompCars from Kaggle (renancostaalencar/compcars) and extract
    #   into data/compcars/  (or set COMPCARS_ROOT).
    #   The archive must include the misc/ folder for human-readable make names.
    #
    # Usage:
    #   bash scripts/pipeline_compcars_by_make.sh
    #
    # Common overrides:
    #   # Only specific manufacturers
    #   MAKES="Toyota Honda BMW Audi" bash scripts/pipeline_compcars_by_make.sh
    #
    #   # Stricter class size filter
    #   MIN_CLASS_SIZE=500 bash scripts/pipeline_compcars_by_make.sh
    #
    #   # Fewer encode GPUs, longer training
    #   NUM_ENCODE_GPUS=2 MAX_TRAIN_STEPS=200000 bash scripts/pipeline_compcars_by_make.sh
    set -euo pipefail

    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    SCRIPTS="$REPO_ROOT/scripts"
    DATA="$REPO_ROOT/data"
    PREPROC="$REPO_ROOT/REPA/preprocessing"

    # ── tuneable knobs ─────────────────────────────────────────────────────────
    COMPCARS_ROOT="${COMPCARS_ROOT:-$DATA/compcars}"
    MISC_DIR="${MISC_DIR:-$COMPCARS_ROOT/misc}"
    OUTPUT_DIR="${OUTPUT_DIR:-$DATA/compcars256_by_make}"
    RESOLUTION="${RESOLUTION:-256}"
    MIN_CLASS_SIZE="${MIN_CLASS_SIZE:-100}"   # drop makes with fewer images
    MAKES="${MAKES:-}"                        # space-separated whitelist, e.g. "Toyota BMW"
                                              # leave empty to keep all makes
    NUM_ENCODE_GPUS="${NUM_ENCODE_GPUS:-4}"
    ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"
    VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
    CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
    # ──────────────────────────────────────────────────────────────────────────

    DEST="$OUTPUT_DIR/vae-sd"

    # ── Step 1: Export images with make-level classes ──────────────────────────
    echo "=== [1/3] Exporting CompCars images (manufacturer-level classes) ==="

    MAKES_ARG=""
    if [ -n "$MAKES" ]; then
        # Convert space-separated string into --makes arg1 arg2 ...
        # shellcheck disable=SC2086
        MAKES_ARG="--makes $MAKES"
    fi

    # shellcheck disable=SC2086
    python "$SCRIPTS/export_compcars_by_make.py" \
        --root-dir   "$COMPCARS_ROOT" \
        --misc-dir   "$MISC_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --resolution "$RESOLUTION" \
        --min-class-size "$MIN_CLASS_SIZE" \
        $MAKES_ARG

    # Read NUM_CLASSES from the classes.json written by the export script
    NUM_CLASSES=$(python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/classes.json'))
print(len(d))
")
    echo "NUM_CLASSES = $NUM_CLASSES"

    # ── Step 2: Encode VAE latents (parallel across GPUs) ──────────────────────
    echo "=== [2/3] Encoding VAE latents on $NUM_ENCODE_GPUS GPU(s) ==="

    pids=()
    for i in $(seq 0 $((NUM_ENCODE_GPUS - 1))); do
        python "$PREPROC/dataset_tools.py" encode \
            --source     "$OUTPUT_DIR" \
            --dest       "$DEST" \
            --model-url  "$VAE_MODEL" \
            --batch-size "$ENCODE_BATCH_SIZE" \
            --device     "cuda:$i" \
            --rank       "$i" \
            --world-size "$NUM_ENCODE_GPUS" &
        pids+=($!)
        echo "  launched rank $i (PID ${pids[-1]})"
    done

    # Wait for all workers to finish
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: worker PID $pid failed"
            failed=1
        fi
    done
    [ "$failed" -eq 0 ] || { echo "One or more encoding workers failed. Aborting."; exit 1; }

    # Merge shards if more than one GPU was used
    if [ "$NUM_ENCODE_GPUS" -gt 1 ]; then
        echo "=== Merging VAE shards ==="
        python "$SCRIPTS/merge_vae_shards.py" \
            --dest       "$DEST" \
            --world-size "$NUM_ENCODE_GPUS"
    fi

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    echo "=== [3/3] Training ==="

    echo "--- Baseline (no REPA) ---"
    NUM_CLASSES=$NUM_CLASSES \
    MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
    CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
    EXP_NAME=compcars_by_make_sit_b2_baseline \
    bash "$SCRIPTS/run_compcars_by_make_baseline.sh"

    echo "--- REPA (DINOv2-B) ---"
    NUM_CLASSES=$NUM_CLASSES \
    MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS \
    CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS \
    EXP_NAME=compcars_by_make_sit_b2_repa_dinov2b \
    bash "$SCRIPTS/run_compcars_by_make_repa.sh"

    echo "=== Pipeline complete ==="
    echo "Classes : $NUM_CLASSES"
    echo "Data    : $OUTPUT_DIR"
    echo "Runs    : $REPO_ROOT/runs/"
