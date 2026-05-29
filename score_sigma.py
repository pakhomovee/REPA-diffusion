#!/usr/bin/env python3
"""
score_sigma.py — Generate 50k samples at REPA-Σ checkpoints 100k and 200k
(EMA weights) and compute FID / KID against CelebA-256 real stats.

Mirrors the hyperparameter set used by fid_kid.py main() to produce the
existing Baseline/REPA scores in scores.csv:
  model=SiT-B/2, num_classes=16, encoder_depth=4, projector_embed_dims=768
  num_fid_samples=50_000, mode=ode, num_steps=50, cfg_scale=2.0,
  guidance_high=0.7, weights=ema, vae=mse

Output:
  /home/seankopylov/projects/repa_diffusion_celeba/fid_samples/REPA-Sigma/
    step_0100000/*.npz
    step_0200000/*.npz
  scores_sigma.csv  (run, step, fid, kid_e3)
"""
import os, sys, csv
sys.path.insert(0, "/home/seankopylov/projects/repa_diffusion_celeba")
os.chdir("/home/seankopylov/projects/repa_diffusion_celeba")

from pathlib import Path

from fid_kid import (
    generate_samples, compute_fid_kid, ensure_real_stats_cached,
    PROJECT_ROOT, REPA_ROOT, REAL_DATA, RUNS_ROOT,
)

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_NAME = "REPA-Sigma (DINOv2-B)"
RUN_DIR  = RUNS_ROOT / "celeba_sit_b2_repa_sigma_hard_gpus2-7"
STEPS    = [100_000, 200_000]

NUM_FID_SAMPLES     = 50_000
PER_PROC_BATCH_SIZE = 512
DEVICE_COUNT        = int(os.environ.get("NUM_GPUS", "8"))

samples_root = PROJECT_ROOT / "fid_samples"
samples_root.mkdir(exist_ok=True)

# ── 1. Generate samples (per checkpoint, EMA weights) ─────────────────────────
npz_paths = {RUN_NAME: []}
for step in STEPS:
    ckpt = str(RUN_DIR / "checkpoints" / f"{step:07d}.pt")
    assert Path(ckpt).exists(), f"Missing checkpoint: {ckpt}"
    out_dir = str(samples_root / RUN_NAME / f"step_{step:07d}")
    print(f"\n{'#'*70}\n# Generating {RUN_NAME} step={step} ({NUM_FID_SAMPLES} samples)\n{'#'*70}", flush=True)
    npz = generate_samples(
        ckpt_path=ckpt,
        output_dir=out_dir,
        model="SiT-B/2",
        num_classes=16,
        num_fid_samples=NUM_FID_SAMPLES,
        per_proc_batch_size=PER_PROC_BATCH_SIZE,
        encoder_depth=4,
        projector_embed_dims="768",
        path_type="linear",
        mode="ode",
        num_steps=50,
        cfg_scale=2.0,
        guidance_high=0.7,
        weights="ema",
        repa_root=str(REPA_ROOT),
        device_count=DEVICE_COUNT,
    )
    npz_paths[RUN_NAME].append(npz)

# ── 2. Compute FID / KID against cached CelebA-256 real stats ─────────────────
ensure_real_stats_cached(str(REAL_DATA))

fid_scores, kid_scores = compute_fid_kid(
    npz_paths=npz_paths,
    checkpoint_steps=STEPS,
)

# ── 3. Write scores_sigma.csv ─────────────────────────────────────────────────
out_csv = PROJECT_ROOT / "scores_sigma.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "step", "fid", "kid_e3"])
    for step, fid, kid in zip(STEPS, fid_scores[RUN_NAME], kid_scores[RUN_NAME]):
        w.writerow([RUN_NAME, step, f"{fid:.4f}", f"{kid * 1e3:.6f}"])

print(f"\n{'#'*70}")
print(f"# Done. Results:")
for step, fid, kid in zip(STEPS, fid_scores[RUN_NAME], kid_scores[RUN_NAME]):
    print(f"#   {RUN_NAME} step={step}: FID={fid:.2f}  KID×10³={kid * 1e3:.3f}")
print(f"# Written → {out_csv}")
print(f"{'#'*70}")
