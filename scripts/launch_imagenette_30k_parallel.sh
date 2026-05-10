#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

baseline_exp="${BASELINE_EXP:-imagenette_sit_b2_baseline_30k_gpus0-3}"
repa_exp="${REPA_EXP:-imagenette_sit_b2_repa_dinov2b_30k_gpus4-7}"

baseline_log="logs/train_baseline_30k_gpus0-3.log"
repa_log="logs/train_repa_30k_gpus4-7.log"

if [[ -s logs/baseline_30k.pid ]] && kill -0 "$(cat logs/baseline_30k.pid)" 2>/dev/null; then
  echo "Baseline PID from logs/baseline_30k.pid is still running: $(cat logs/baseline_30k.pid)" >&2
  exit 1
fi

if [[ -s logs/repa_30k.pid ]] && kill -0 "$(cat logs/repa_30k.pid)" 2>/dev/null; then
  echo "REPA PID from logs/repa_30k.pid is still running: $(cat logs/repa_30k.pid)" >&2
  exit 1
fi

if pgrep -f "EXP_NAME=${baseline_exp}" >/dev/null; then
  echo "Baseline run already appears to be running: ${baseline_exp}" >&2
  exit 1
fi

if pgrep -f "EXP_NAME=${repa_exp}" >/dev/null; then
  echo "REPA run already appears to be running: ${repa_exp}" >&2
  exit 1
fi

echo "Launching baseline on GPUs 0,1,2,3 -> ${baseline_exp}"
nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_PROCESSES=4 \
  BATCH_SIZE=256 \
  MAX_TRAIN_STEPS=30000 \
  CHECKPOINTING_STEPS=5000 \
  SAMPLING_STEPS=1000000 \
  EXP_NAME="${baseline_exp}" \
  REPORT_TO=tensorboard \
  MAIN_PROCESS_PORT=29601 \
  ./scripts/run_imagenette_baseline.sh \
  > "${baseline_log}" 2>&1 &
baseline_pid=$!
echo "${baseline_pid}" > logs/baseline_30k.pid

echo "Launching REPA on GPUs 4,5,6,7 -> ${repa_exp}"
nohup env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  NUM_PROCESSES=4 \
  BATCH_SIZE=256 \
  MAX_TRAIN_STEPS=30000 \
  CHECKPOINTING_STEPS=5000 \
  SAMPLING_STEPS=1000000 \
  EXP_NAME="${repa_exp}" \
  REPORT_TO=tensorboard \
  MAIN_PROCESS_PORT=29602 \
  ./scripts/run_imagenette_repa.sh \
  > "${repa_log}" 2>&1 &
repa_pid=$!
echo "${repa_pid}" > logs/repa_30k.pid

cat <<EOF

Started:
  baseline pid: ${baseline_pid}
  REPA pid:     ${repa_pid}

Logs:
  ${baseline_log}
  ${repa_log}

Progress:
  ./scripts/show_imagenette_30k_progress.sh
  watch -n 30 ./scripts/show_imagenette_30k_progress.sh

Raw tqdm logs:
  tail -f ${baseline_log}
  tail -f ${repa_log}

TensorBoard:
  BASELINE_EXP=${baseline_exp} REPA_EXP=${repa_exp} ./scripts/start_tensorboard.sh

EOF
