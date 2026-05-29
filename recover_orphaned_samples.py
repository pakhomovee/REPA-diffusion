#!/usr/bin/env python3
"""
recover_orphaned_samples.py — Recover the 5 orphaned PNG sample dirs left by
the previous run that timed out at NCCL barriers.

For each step in {80k, 120k, 140k, 160k, 180k}:
  1. Read 50,000 PNGs from REPA/samples/SiT-B-2-XXX.../ and build an .npz
     (no DDP, no NCCL — single process)
  2. Move the .npz to fid_samples/REPA-Sigma (DINOv2-B)/step_XXXXXXX/
  3. Compute FID + KID against cached CelebA real stats
  4. Append row to scores_sigma.csv

Uses only the visible GPUs (set via CUDA_VISIBLE_DEVICES). Recommend GPUs 4-7.
"""
import os, sys, csv, time, traceback
sys.path.insert(0, "/home/seankopylov/projects/repa_diffusion_celeba")
os.chdir("/home/seankopylov/projects/repa_diffusion_celeba")

from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/seankopylov/projects/repa_diffusion_celeba/REPA")
from generate import create_npz_from_sample_folder

from fid_kid import (
    _load_npz_fast, _NpDataset,
    ensure_real_stats_cached,
    PROJECT_ROOT, REAL_DATA,
)
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, kernel_distance

RUN_NAME = "REPA-Sigma (DINOv2-B)"
STEPS = [80_000, 120_000, 140_000, 160_000, 180_000]

REPA_SAMPLES = Path("/home/seankopylov/projects/repa_diffusion_celeba/REPA/samples")
TARGET_ROOT  = PROJECT_ROOT / "fid_samples" / RUN_NAME
OUT_CSV      = PROJECT_ROOT / "scores_sigma.csv"

# ── Phase A: Build npz for each step ──────────────────────────────────────────
for step in STEPS:
    src_dir = REPA_SAMPLES / f"SiT-B-2-{step:07d}-size-256-vae-ema-cfg-2.0-seed-0-ode"
    target_dir = TARGET_ROOT / f"step_{step:07d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_npz = target_dir / f"{src_dir.name}.npz"

    if target_npz.exists():
        print(f"[skip-npz] step={step}: {target_npz.name} already in target ({target_npz.stat().st_size/1e9:.1f} GB)", flush=True)
        continue

    print(f"\n[build-npz] step={step}: reading PNGs from {src_dir.name}", flush=True)
    assert src_dir.exists(), f"missing PNG dir: {src_dir}"
    n_pngs = len(list(src_dir.glob("*.png")))
    print(f"           found {n_pngs} PNGs", flush=True)
    assert n_pngs >= 50_000, f"only {n_pngs} PNGs in {src_dir}, need 50000"

    t0 = time.perf_counter()
    npz_path = create_npz_from_sample_folder(str(src_dir), num=50_000)
    print(f"[build-npz] step={step}: created {npz_path} in {time.perf_counter()-t0:.1f}s", flush=True)

    Path(npz_path).rename(target_npz)
    print(f"[build-npz] step={step}: moved → {target_npz}", flush=True)

    # Free disk: delete the source PNG directory (4.5 GB). The npz is the
    # canonical artifact; PNGs were intermediate only.
    import shutil
    shutil.rmtree(src_dir)
    print(f"[build-npz] step={step}: deleted source PNG dir ({src_dir.name})", flush=True)


# ── Phase B: Score each new npz ───────────────────────────────────────────────
existing = set()
with open(OUT_CSV, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        existing.add((row['run'], int(row['step'])))
print(f"\n[csv] {len(existing)} existing rows in {OUT_CSV}", flush=True)

print("\n[setup] ensure real stats cached", flush=True)
ensure_real_stats_cached(str(REAL_DATA))

print("[setup] build InceptionV3 + load real stats", flush=True)
device = torch.device("cuda")
feat_model = build_feature_extractor("clean", device, use_dataparallel=True)

import cleanfid as _cfd_pkg
stats_dir = Path(_cfd_pkg.__file__).parent / "stats"
fid_stats = np.load(stats_dir / "celeba256_clean_custom_na.npz")
mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
real_kid_feats = np.load(stats_dir / "celeba256_clean_custom_na_kid.npz", mmap_mode="r")["feats"]
print(f"[setup] real_kid_feats shape={real_kid_feats.shape}", flush=True)

csv_file = open(OUT_CSV, 'a', newline='')
csv_writer = csv.writer(csv_file)

for step in STEPS:
    if (RUN_NAME, step) in existing:
        print(f"[skip-score] step={step}: already in CSV", flush=True)
        continue

    target_npz = TARGET_ROOT / f"step_{step:07d}" / f"SiT-B-2-{step:07d}-size-256-vae-ema-cfg-2.0-seed-0-ode.npz"
    if not target_npz.exists():
        print(f"!! [missing-npz] step={step}: {target_npz} — skipping", flush=True)
        continue

    print(f"\n[score] step={step}", flush=True)
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    samples = _load_npz_fast(str(target_npz))
    print(f"  [score] loaded {len(samples)} samples in {time.perf_counter()-t0:.1f}s", flush=True)

    loader = torch.utils.data.DataLoader(
        _NpDataset(samples), batch_size=512, num_workers=48,
        shuffle=False, drop_last=False, pin_memory=True,
    )
    t1 = time.perf_counter()
    l_feats = []
    for batch in loader:
        with torch.no_grad():
            l_feats.append(feat_model(batch.to(device)).detach().cpu().numpy())
    gen_feats = np.concatenate(l_feats)
    print(f"  [score] InceptionV3 forward in {time.perf_counter()-t1:.1f}s", flush=True)

    t2 = time.perf_counter()
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    fid_val = float(frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen))
    print(f"  [score] FID computed in {time.perf_counter()-t2:.1f}s", flush=True)

    t3 = time.perf_counter()
    kid_val = float(kernel_distance(real_kid_feats, gen_feats))
    print(f"  [score] KID computed in {time.perf_counter()-t3:.1f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"  RESULT step={step}: FID={fid_val:.4f}  KID×10³={kid_val*1e3:.4f}  ({elapsed:.1f}s)\n", flush=True)
    csv_writer.writerow([RUN_NAME, step, f"{fid_val:.4f}", f"{kid_val*1e3:.6f}"])
    csv_file.flush()
    os.fsync(csv_file.fileno())

csv_file.close()
print("\n######################################################################")
print(f"# Done. {OUT_CSV} now contains:")
with open(OUT_CSV) as f:
    print(f.read())
print("######################################################################")
