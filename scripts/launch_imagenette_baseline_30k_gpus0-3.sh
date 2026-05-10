#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

exp_name="${EXP_NAME:-imagenette_sit_b2_baseline_30k_gpus0-3}"
log_path="${LOG_PATH:-logs/train_baseline_30k_gpus0-3.log}"
pid_path="${PID_PATH:-logs/baseline_30k.pid}"

if [[ -s "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
  echo "Baseline PID from ${pid_path} is still running: $(cat "${pid_path}")" >&2
  exit 1
fi

if pgrep -f "EXP_NAME=${exp_name}" >/dev/null; then
  echo "Baseline run already appears to be running: ${exp_name}" >&2
  exit 1
fi

echo "Launching baseline on GPUs 0,1,2,3 -> ${exp_name}"
nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_PROCESSES=4 \
  BATCH_SIZE=256 \
  MAX_TRAIN_STEPS=30000 \
  CHECKPOINTING_STEPS=5000 \
  SAMPLING_STEPS=1000000 \
  EXP_NAME="${exp_name}" \
  REPORT_TO=tensorboard \
  MAIN_PROCESS_PORT=29601 \
  ./scripts/run_imagenette_baseline.sh \
  > "${log_path}" 2>&1 &

pid=$!
echo "${pid}" > "${pid_path}"

cat <<EOF
Started baseline:
  pid:  ${pid}
  exp:  ${exp_name}
  log:  ${log_path}

Progress:
  ./scripts/show_imagenette_30k_progress.sh
  tail -f ${log_path}
EOF
