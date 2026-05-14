#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-6006}"
HOST="${HOST:-127.0.0.1}"
BASELINE_EXP="${BASELINE_EXP:-tb_imagenette_sit_b2_baseline_1k}"
REPA_EXP="${REPA_EXP:-tb_imagenette_sit_b2_repa_dinov2b_1k}"

BASELINE_LOGDIR="runs/${BASELINE_EXP}/logs/REPA"
REPA_LOGDIR="runs/${REPA_EXP}/logs/REPA"
LOGDIR_SPEC="baseline_sit_b2_no_repa_1k:${BASELINE_LOGDIR},repa_sit_b2_dinov2b_1k:${REPA_LOGDIR}"

echo "TensorBoard will read:"
echo "  baseline_sit_b2_no_repa_1k -> ${BASELINE_LOGDIR}"
echo "  repa_sit_b2_dinov2b_1k    -> ${REPA_LOGDIR}"
echo
echo "From your laptop, open a tunnel in another terminal:"
echo "  ssh -L ${PORT}:127.0.0.1:${PORT} <your-server>"
echo
echo "Then open:"
echo "  http://localhost:${PORT}"
echo

exec ${TENSORBOARD:-tensorboard} \
  --logdir_spec "${LOGDIR_SPEC}" \
  --host "${HOST}" \
  --port "${PORT}"
