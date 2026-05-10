#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

exp_name="${EXP_NAME:-imagenette_sit_b2_repa_dinov2b_30k_gpus4-7}"
resume_step="${RESUME_STEP:-50000}"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
checkpointing_steps="${CHECKPOINTING_STEPS:-10000}"
num_processes="${NUM_PROCESSES:-6}"
main_process_port="${MAIN_PROCESS_PORT:-29612}"
log_file="${LOG_FILE:-logs/train_repa_continue_50k_to_100k_gpus0-5.log}"
pid_file="${PID_FILE:-logs/repa_continue_50k_to_100k.pid}"

resume_ckpt="$(printf "runs/%s/checkpoints/%07d.pt" "$exp_name" "$resume_step")"
if [[ ! -f "$resume_ckpt" ]]; then
  echo "Missing resume checkpoint: $resume_ckpt" >&2
  exit 1
fi

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "A run from $pid_file is already alive: PID $(cat "$pid_file")" >&2
  exit 1
fi

mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export WANDB_MODE="${WANDB_MODE:-offline}"

nohup ${ACCELERATE:-accelerate} launch \
  --num_processes "$num_processes" \
  --mixed_precision "fp16" \
  --main_process_port "$main_process_port" \
  REPA/train.py \
  --report-to="${REPORT_TO:-tensorboard}" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --num-classes=10 \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="runs" \
  --logging-dir="logs" \
  --exp-name="$exp_name" \
  --data-dir="data/imagenette256-train" \
  --resolution=256 \
  --batch-size="${BATCH_SIZE:-256}" \
  --max-train-steps="$max_train_steps" \
  --checkpointing-steps="$checkpointing_steps" \
  --sampling-steps="${SAMPLING_STEPS:-1000000}" \
  --no-sample-at-step-one \
  --num-workers="${NUM_WORKERS:-4}" \
  --resume-step="$resume_step" \
  > "$log_file" 2>&1 &

echo $! > "$pid_file"
echo "Started REPA continuation ${resume_step}->${max_train_steps}"
echo "PID: $(cat "$pid_file")"
echo "Log: $log_file"
echo "Resume checkpoint: $resume_ckpt"
