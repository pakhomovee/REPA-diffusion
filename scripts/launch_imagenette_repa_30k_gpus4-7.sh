#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

exp_name="${EXP_NAME:-imagenette_sit_b2_repa_dinov2b_30k_gpus4-7}"
log_path="${LOG_PATH:-logs/train_repa_30k_gpus4-7.log}"
pid_path="${PID_PATH:-logs/repa_30k.pid}"

if [[ -s "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
  echo "REPA PID from ${pid_path} is still running: $(cat "${pid_path}")" >&2
  exit 1
fi

if pgrep -f "EXP_NAME=${exp_name}" >/dev/null; then
  echo "REPA run already appears to be running: ${exp_name}" >&2
  exit 1
fi

echo "Launching REPA on GPUs 4,5,6,7 -> ${exp_name}"
nohup env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  NUM_PROCESSES=4 \
  BATCH_SIZE=256 \
  MAX_TRAIN_STEPS=30000 \
  CHECKPOINTING_STEPS=5000 \
  SAMPLING_STEPS=1000000 \
  EXP_NAME="${exp_name}" \
  REPORT_TO=tensorboard \
  MAIN_PROCESS_PORT=29602 \
  ./scripts/run_imagenette_repa.sh \
  > "${log_path}" 2>&1 &

pid=$!
echo "${pid}" > "${pid_path}"

cat <<EOF
Started REPA:
  pid:  ${pid}
  exp:  ${exp_name}
  log:  ${log_path}

Progress:
  ./scripts/show_imagenette_30k_progress.sh
  tail -f ${log_path}
EOF
