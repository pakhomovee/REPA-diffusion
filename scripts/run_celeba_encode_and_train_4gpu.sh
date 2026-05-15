#!/usr/bin/env bash
# Encode CelebA VAE latents in parallel on GPUs 4,5,6,7, then start CelebA REPA training.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/seankopylov/projects/repa_diffusion_celeba}"
PYTHON="${PYTHON:-/home/seankopylov/.venv/bin/python}"
ACCELERATE="${ACCELERATE:-/home/seankopylov/.venv/bin/accelerate}"

ENC_GPUS="${ENC_GPUS:-4,5,6,7}"
ENC_BATCH="${ENC_BATCH:-128}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-1000000}"
EXP_NAME="${EXP_NAME:-celeba_sit_b2_repa_dinov2b_gpus4-7}"
REPORT_TO="${REPORT_TO:-tensorboard}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29677}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run_celeba_repa.sh}"

cd "$REPO_ROOT"
mkdir -p logs

if [[ ! -f data/celeba256/dataset.json ]]; then
  echo "[setup] data/celeba256/dataset.json missing; exporting CelebA first"
  "$PYTHON" scripts/export_celeba_for_repa.py 2>&1 | tee logs/export_celeba_for_repa.log
fi

echo "[encode] checking/creating VAE latents with GPUs ${ENC_GPUS}, batch ${ENC_BATCH}"
REPO_ROOT="$REPO_ROOT" FORCE_REENCODE="${FORCE_REENCODE:-0}" ENC_GPUS="$ENC_GPUS" ENC_BATCH="$ENC_BATCH" "$PYTHON" - <<'PY' 2>&1 | tee logs/encode_celeba256_vae_sd_parallel_gpus4-7.log
import os
import sys
import io
import json
import time
import shutil
import multiprocessing as mp
from pathlib import Path

ROOT = Path(os.environ.get("REPO_ROOT", "/home/seankopylov/projects/repa_diffusion_celeba"))
SRC = ROOT / "data/celeba256"
IMG_DIR = SRC / "images"
DEST = SRC / "vae-sd"
GPUS = [gpu.strip() for gpu in os.environ["ENC_GPUS"].split(",") if gpu.strip()]
BATCH = int(os.environ.get("ENC_BATCH", "128"))
FORCE_REENCODE = os.environ.get("FORCE_REENCODE", "0") == "1"

sys.path.insert(0, str(ROOT / "REPA/preprocessing"))

labels_src = dict(json.loads((SRC / "dataset.json").read_text())["labels"])
image_files = sorted(p.name for p in IMG_DIR.glob("*.jpg"))
n = len(image_files)

def feat_name(i):
    s = f"{i:08d}"
    return f"{s[:5]}/img-mean-std-{s}.npy"

def is_complete():
    if not (DEST / "dataset.json").exists():
        return False
    count = sum(1 for _ in DEST.glob("*/*.npy"))
    if count != n:
        print(f"[encode] existing latent count is {count}, expected {n}")
        return False
    try:
        labels = json.loads((DEST / "dataset.json").read_text())["labels"]
    except Exception as exc:
        print(f"[encode] existing dataset.json is unreadable: {exc}")
        return False
    if len(labels) != n:
        print(f"[encode] existing label count is {len(labels)}, expected {n}")
        return False
    return True

def flush(indices, enc, dev):
    import numpy as np
    import torch
    from PIL import Image

    arrs = []
    for i in indices:
        img = Image.open(IMG_DIR / image_files[i]).convert("RGB")
        arrs.append(np.asarray(img, dtype=np.uint8))

    x = torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2)
    with torch.inference_mode():
        z = enc.encode_pixels(x).cpu().numpy()

    for j, i in enumerate(indices):
        out = DEST / feat_name(i)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, z[j])
        os.replace(tmp, out)

def worker(rank, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from encoders import StabilityVAEEncoder

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dev = torch.device("cuda")
    enc = StabilityVAEEncoder("stabilityai/sd-vae-ft-mse", batch_size=BATCH)
    enc.init(dev)

    done = 0
    batch = []
    for i in range(rank, n, len(GPUS)):
        batch.append(i)
        if len(batch) < BATCH:
            continue
        flush(batch, enc, dev)
        done += len(batch)
        print(f"[encode][gpu {gpu}] encoded {done}", flush=True)
        batch = []
    if batch:
        flush(batch, enc, dev)
        done += len(batch)
        print(f"[encode][gpu {gpu}] encoded final {done}", flush=True)

if __name__ == "__main__":
    if not image_files:
        raise SystemExit(f"No images found in {IMG_DIR}")
    if not GPUS:
        raise SystemExit("ENC_GPUS is empty")

    if not FORCE_REENCODE and is_complete():
        print(f"[encode] complete latent dataset already exists at {DEST}; skipping")
        raise SystemExit(0)

    if DEST.exists():
        print(f"[encode] removing incomplete latent directory: {DEST}")
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    procs = [mp.Process(target=worker, args=(rank, gpu)) for rank, gpu in enumerate(GPUS)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    failures = [proc.exitcode for proc in procs if proc.exitcode != 0]
    if failures:
        raise SystemExit(f"encoder worker failed with exit codes: {failures}")

    missing = [feat_name(i) for i in range(n) if not (DEST / feat_name(i)).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} latents; first missing: {missing[:3]}")

    labels = [[feat_name(i), int(labels_src[image_files[i]])] for i in range(n)]
    (DEST / "dataset.json").write_text(json.dumps({"labels": labels}))
    print(f"[encode] encoded {n} latents to {DEST} in {(time.time() - t0) / 60:.1f} min")
PY

echo "[train] starting ${TRAIN_SCRIPT} on GPUs ${TRAIN_GPUS}"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
NUM_PROCESSES="$NUM_PROCESSES" \
BATCH_SIZE="$TRAIN_BATCH_SIZE" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
CHECKPOINTING_STEPS="$CHECKPOINTING_STEPS" \
SAMPLING_STEPS="$SAMPLING_STEPS" \
EXP_NAME="$EXP_NAME" \
REPORT_TO="$REPORT_TO" \
MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
NUM_WORKERS="$NUM_WORKERS" \
ACCELERATE="$ACCELERATE" \
bash "$TRAIN_SCRIPT" 2>&1 | tee "logs/run_${EXP_NAME}.log"
