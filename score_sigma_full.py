#!/usr/bin/env python3
"""
score_sigma_full.py — FID/KID for REPA-Σ at every 20k checkpoint (20k…200k).

Crash-safe: progressively writes scores_sigma.csv after each checkpoint, and
skips both sample generation and FID computation for any (run, step) already
present in the CSV. Re-running picks up where it left off.

InceptionV3 is built once and reused across all scorings (saves ~30s × 10
checkpoints of model construction).

Hyperparameters identical to score_sigma.py (the 100k/200k initial run):
  model=SiT-B/2, num_classes=16, encoder_depth=4, projector_embed_dims=768
  num_fid_samples=50000, per_proc_batch_size=512, mode=ode, num_steps=50,
  cfg_scale=2.0, guidance_high=0.7, weights=ema, vae=ema (generate.py default)
"""
import os, sys, csv, time, traceback
sys.path.insert(0, "/home/seankopylov/projects/repa_diffusion_celeba")
os.chdir("/home/seankopylov/projects/repa_diffusion_celeba")

from pathlib import Path

import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, kernel_distance

from fid_kid import (
    generate_samples, ensure_real_stats_cached,
    _load_npz_fast, _NpDataset,
    PROJECT_ROOT, REPA_ROOT, REAL_DATA, RUNS_ROOT,
)

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_NAME = "REPA-Sigma (DINOv2-B)"
RUN_DIR  = RUNS_ROOT / "celeba_sit_b2_repa_sigma_hard_gpus2-7"
STEPS    = list(range(20_000, 201_000, 20_000))   # 20k, 40k, …, 200k = 10 ckpts

NUM_FID_SAMPLES     = 50_000
PER_PROC_BATCH_SIZE = 512
DEVICE_COUNT        = int(os.environ.get("NUM_GPUS", "8"))
FID_BATCH_SIZE      = 512
FID_NUM_WORKERS     = 48

samples_root = PROJECT_ROOT / "fid_samples"
samples_root.mkdir(exist_ok=True)

OUT_CSV = PROJECT_ROOT / "scores_sigma.csv"

# ── Load existing rows so we can resume ───────────────────────────────────────
existing: set[tuple[str, int]] = set()
if OUT_CSV.exists():
    with open(OUT_CSV, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            existing.add((row['run'], int(row['step'])))
    print(f"[resume] Found {len(existing)} existing rows in {OUT_CSV}:", flush=True)
    for k in sorted(existing):
        print(f"          {k}", flush=True)

# Open CSV append, write header if file is empty/new
write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
csv_file = open(OUT_CSV, 'a', newline='')
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["run", "step", "fid", "kid_e3"])
    csv_file.flush()

def append_row(run, step, fid, kid):
    csv_writer.writerow([run, step, f"{fid:.4f}", f"{kid*1e3:.6f}"])
    csv_file.flush()
    os.fsync(csv_file.fileno())

# ── Helper to score one npz against pre-loaded real stats ─────────────────────
def score_one(npz_path, feat_model, device, mu_real, sigma_real, real_kid_feats):
    """Load npz, run InceptionV3, compute FID and KID. Returns (fid, kid)."""
    t0 = time.perf_counter()
    samples = _load_npz_fast(npz_path)
    print(f"    [score] loaded {len(samples)} samples in {time.perf_counter()-t0:.1f}s", flush=True)

    dataset = _NpDataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=FID_BATCH_SIZE, num_workers=FID_NUM_WORKERS,
        shuffle=False, drop_last=False, pin_memory=True,
    )

    t1 = time.perf_counter()
    l_feats = []
    for batch in loader:
        with torch.no_grad():
            l_feats.append(feat_model(batch.to(device)).detach().cpu().numpy())
    gen_feats = np.concatenate(l_feats)   # (50000, 2048)
    print(f"    [score] InceptionV3 forward in {time.perf_counter()-t1:.1f}s", flush=True)

    t2 = time.perf_counter()
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    fid_val = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"    [score] FID computed in {time.perf_counter()-t2:.1f}s", flush=True)

    t3 = time.perf_counter()
    kid_val = kernel_distance(real_kid_feats, gen_feats)
    print(f"    [score] KID computed in {time.perf_counter()-t3:.1f}s", flush=True)

    return float(fid_val), float(kid_val)


# ── Phase 0: Pre-warm real stats cache ────────────────────────────────────────
print("\n" + "#"*70 + "\n# PHASE 0: ensure real stats cached\n" + "#"*70, flush=True)
ensure_real_stats_cached(str(REAL_DATA))

# ── Build InceptionV3 once, load real stats once ──────────────────────────────
print("\n" + "#"*70 + "\n# PHASE 0: build InceptionV3 + load real stats\n" + "#"*70, flush=True)
device = torch.device("cuda")
feat_model = build_feature_extractor("clean", device, use_dataparallel=True)

import cleanfid as _cfd_pkg
stats_dir = Path(_cfd_pkg.__file__).parent / "stats"
fid_stats = np.load(stats_dir / "celeba256_clean_custom_na.npz")
mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
print("[setup] loaded mu/sigma for FID", flush=True)
real_kid_feats = np.load(stats_dir / "celeba256_clean_custom_na_kid.npz",
                          mmap_mode="r")["feats"]
print(f"[setup] loaded real_kid_feats with shape {real_kid_feats.shape} (mmap)", flush=True)


# ── Main loop: one checkpoint at a time, gen + score, append CSV row ──────────
for step in STEPS:
    key = (RUN_NAME, step)

    ckpt_path = RUN_DIR / "checkpoints" / f"{step:07d}.pt"
    if not ckpt_path.exists():
        print(f"\n!! MISSING CHECKPOINT: {ckpt_path}  — skipping step {step}", flush=True)
        continue

    out_dir = samples_root / RUN_NAME / f"step_{step:07d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── A. sample generation (skipped if cached npz with enough samples) ─────
    print(f"\n{'#'*70}\n# step={step}: PHASE A — sample generation\n{'#'*70}", flush=True)
    try:
        npz = generate_samples(
            ckpt_path=str(ckpt_path),
            output_dir=str(out_dir),
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
    except Exception:
        print(f"!! GENERATION FAILED for step={step}:")
        traceback.print_exc()
        continue

    # ── B. FID/KID scoring (skipped if row already in CSV) ────────────────────
    if key in existing:
        print(f"[skip-score] step={step} already in {OUT_CSV.name}", flush=True)
        continue

    print(f"\n{'#'*70}\n# step={step}: PHASE B — FID/KID scoring\n{'#'*70}", flush=True)
    try:
        t_start = time.perf_counter()
        fid, kid = score_one(npz, feat_model, device, mu_real, sigma_real, real_kid_feats)
        elapsed = time.perf_counter() - t_start
        print(f"\n  RESULT step={step}: FID={fid:.4f}  KID×10³={kid*1e3:.4f}  ({elapsed:.1f}s)\n", flush=True)
        append_row(RUN_NAME, step, fid, kid)
        existing.add(key)
    except Exception:
        print(f"!! SCORING FAILED for step={step}:")
        traceback.print_exc()
        continue


csv_file.close()
print(f"\n{'#'*70}\n# Done. {OUT_CSV} contains:")
with open(OUT_CSV) as f:
    print(f.read())
print(f"{'#'*70}", flush=True)
