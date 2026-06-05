    #!/usr/bin/env bash
    # Smoke-test pipeline for CompCars (manufacturer-level classes).
    # Runs every step on a tiny subset to verify the full pipeline works.
    #
    # What this does differently from pipeline_compcars_by_make.sh:
    #   - Uses only 2-3 specific makes (Toyota + BMW) to keep class count small
    #   - Keeps only makes with ≥1 image (no filtering)
    #   - Exports only MAX_IMAGES_PER_CLASS images per make (default 5)
    #   - Encodes on a single GPU
    #   - Trains for only 100 steps, no sampling
    #   - Writes to data/compcars256_by_make_test/ and runs/compcars_make_test_*
    #
    # Usage:
    #   bash scripts/pipeline_compcars_by_make_test.sh
    #
    # Expected runtime: ~3-5 minutes on a single GPU.
    set -euo pipefail

    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    SCRIPTS="$REPO_ROOT/scripts"
    DATA="$REPO_ROOT/data"
    PREPROC="$REPO_ROOT/REPA/preprocessing"

    # ── test knobs ─────────────────────────────────────────────────────────────
    COMPCARS_ROOT="${COMPCARS_ROOT:-$DATA/compcars}"
    MISC_DIR="${MISC_DIR:-$COMPCARS_ROOT/misc}"
    OUTPUT_DIR="${OUTPUT_DIR:-$DATA/compcars256_by_make_test}"
    RESOLUTION="${RESOLUTION:-256}"
    TEST_MAKES="${TEST_MAKES:-Toyota BMW Audi}"   # small whitelist → ~3 classes
    MAX_IMAGES_PER_CLASS="${MAX_IMAGES_PER_CLASS:-20}"  # 20 × 3 = ~60 images
    ENCODE_GPU="${ENCODE_GPU:-0}"
    ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-8}"
    VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"
    TRAIN_STEPS="${TRAIN_STEPS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    # ──────────────────────────────────────────────────────────────────────────

    DEST="$OUTPUT_DIR/vae-sd"

    echo "╔══════════════════════════════════════════════════╗"
    echo "║  CompCars (by make) SMOKE TEST                   ║"
    echo "║  Makes       : $TEST_MAKES"
    echo "║  Max img/cls : $MAX_IMAGES_PER_CLASS (must be > batch_size/classes)    ║"
    echo "║  Train steps : $TRAIN_STEPS                             ║"
    echo "╚══════════════════════════════════════════════════╝"

    # ── Step 1: Export tiny subset ─────────────────────────────────────────────
    echo ""
    echo "=== [1/3] Exporting CompCars test subset ==="
    # shellcheck disable=SC2086
    python "$SCRIPTS/export_compcars_by_make.py" \
        --root-dir        "$COMPCARS_ROOT" \
        --misc-dir        "$MISC_DIR" \
        --output-dir      "$OUTPUT_DIR" \
        --resolution      "$RESOLUTION" \
        --min-class-size  1 \
        --max-images-per-class "$MAX_IMAGES_PER_CLASS" \
        --makes           $TEST_MAKES

    NUM_CLASSES=$(python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/classes.json'))
print(len(d))
")
    echo "NUM_CLASSES = $NUM_CLASSES"

    # ── Step 2: Encode VAE latents (single GPU) ────────────────────────────────
    echo ""
    echo "=== [2/3] Encoding VAE latents (single GPU: cuda:$ENCODE_GPU) ==="
    python "$PREPROC/dataset_tools.py" encode \
        --source     "$OUTPUT_DIR/images" \
        --dest       "$DEST" \
        --model-url  "$VAE_MODEL" \
        --batch-size "$ENCODE_BATCH_SIZE" \
        --device     "cuda:$ENCODE_GPU"

    # ── Step 3: Train (minimal steps) ─────────────────────────────────────────
    echo ""
    echo "=== [3/3] Training (smoke test: $TRAIN_STEPS steps) ==="

    cd "$REPO_ROOT/REPA"

    echo "--- Baseline ---"
    ${ACCELERATE:-accelerate} launch \
      --num_processes 1 \
      --mixed_precision "fp16" \
      --main_process_port 29533 \
      train.py \
      --report-to="tensorboard" \
      --allow-tf32 \
      --mixed-precision="fp16" \
      --seed=0 \
      --path-type="linear" \
      --prediction="v" \
      --weighting="uniform" \
      --model="SiT-B/2" \
      --num-classes="$NUM_CLASSES" \
      --enc-type="none" \
      --proj-coeff=0.0 \
      --encoder-depth=4 \
      --output-dir="../runs" \
      --exp-name="compcars_make_test_baseline" \
      --data-dir="$OUTPUT_DIR" \
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
      --main_process_port 29534 \
      train.py \
      --report-to="tensorboard" \
      --allow-tf32 \
      --mixed-precision="fp16" \
      --seed=0 \
      --path-type="linear" \
      --prediction="v" \
      --weighting="uniform" \
      --model="SiT-B/2" \
      --num-classes="$NUM_CLASSES" \
      --enc-type="dinov2-vit-b" \
      --proj-coeff=0.5 \
      --encoder-depth=4 \
      --output-dir="../runs" \
      --exp-name="compcars_make_test_repa" \
      --data-dir="$OUTPUT_DIR" \
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
    echo "  Runs    : $REPO_ROOT/runs/compcars_make_test_*"
