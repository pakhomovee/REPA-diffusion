#!/usr/bin/env python3
"""
score_anneal_robust.py — robust FID/KID for two λ-anneal runs at every 20k.

Improvements over score_anneal_full.py (which suffered repeated NCCL barrier
timeouts on 6-GPU torchrun):

  1. PRIORITY ORDER for acquiring each checkpoint's npz:
     (a) target_dir/*.npz already complete → skip everything
     (b) REPA/samples/SiT-B-2-XXX-…/ has 50k+ PNGs → recover via
         create_npz_from_sample_folder (no DDP, ~85 sec)
     (c) DDP generate.py via torchrun
     (d) DDP failed but left a complete pre-npz → use it
     (e) DDP failed but left enough PNGs → recover from them
     (f) genuine failure → log + continue
  2. ALWAYS cleans up REPA/samples/<basename>/ after each iteration → avoids
     PNG-name collision between the two runs (REPA-anneal 40k vs
     REPA-Sigma-anneal 40k both wrote to the same folder otherwise).
  3. Cleans up partial pre_npz at start of each iteration.
  4. Crash-safe: progressive CSV writes + skip-if-already-done resume.
  5. InceptionV3 built once.
"""
import os, sys, csv, time, shutil, traceback
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "REPA"))
os.chdir(_PROJECT_ROOT)

from pathlib import Path

import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, kernel_distance

from fid_kid import (
    generate_samples, ensure_real_stats_cached,
    _load_npz_fast, _NpDataset, _npz_sample_count,
    PROJECT_ROOT, REPA_ROOT, REAL_DATA, RUNS_ROOT,
)
from generate import create_npz_from_sample_folder

# ── Configuration ─────────────────────────────────────────────────────────────
RUNS = [
    ("REPA-anneal (DINOv2-B)",       "celeba_sit_b2_repa_anneal_gpus2-4"),
    ("REPA-Sigma-anneal (DINOv2-B)", "celeba_sit_b2_repa_sigma_anneal_gpus5-7"),
]
STEPS = list(range(20_000, 201_000, 20_000))   # 10 ckpts

NUM_FID_SAMPLES     = 50_000
PER_PROC_BATCH_SIZE = 512
DEVICE_COUNT        = int(os.environ.get("NUM_GPUS", "6"))
FID_BATCH_SIZE      = 512
FID_NUM_WORKERS     = 48

samples_root = PROJECT_ROOT / "fid_samples"
samples_root.mkdir(exist_ok=True)
OUT_CSV = PROJECT_ROOT / "scores_anneal.csv"

# ── Resume ────────────────────────────────────────────────────────────────────
existing: set[tuple[str, int]] = set()
if OUT_CSV.exists():
    with open(OUT_CSV, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            existing.add((row['run'], int(row['step'])))
    print(f"[resume] {len(existing)} rows already in {OUT_CSV}:", flush=True)
    for k in sorted(existing):
        print(f"          {k}", flush=True)

write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
csv_file = open(OUT_CSV, 'a', newline='')
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["run", "step", "fid", "kid_e3"])
    csv_file.flush()


def append_row(run, step, fid, kid):
    csv_writer.writerow([run, step, f"{fid:.4f}", f"{kid * 1e3:.6f}"])
    csv_file.flush()
    os.fsync(csv_file.fileno())


# ── Acquire npz: target / PNG-recovery / DDP-gen / post-fail recovery ─────────
def acquire_npz(run_name, step, ckpt_path, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # (a) already in target
    existing_npz = list(target_dir.glob("*.npz"))
    if existing_npz:
        n = _npz_sample_count(str(existing_npz[0]))
        if n >= NUM_FID_SAMPLES:
            print(f"  [acquire] target has {n} samples — using {existing_npz[0].name}", flush=True)
            return str(existing_npz[0])
        print(f"  [acquire] target has only {n} samples — re-acquiring", flush=True)
        existing_npz[0].unlink()

    basename = f"SiT-B-2-{step:07d}-size-256-vae-ema-cfg-2.0-seed-0-ode"
    repa_samples = Path(REPA_ROOT) / "samples"
    pre_npz = repa_samples / f"{basename}.npz"
    png_dir = repa_samples / basename

    # (b) full PNG dir from a prior interrupted run → recover, no DDP
    if png_dir.exists():
        n_pngs = sum(1 for p in png_dir.iterdir() if p.name.endswith(".png"))
        if n_pngs >= NUM_FID_SAMPLES:
            print(f"  [acquire] recovering from existing PNG dir ({n_pngs} files)", flush=True)
            if pre_npz.exists():
                pre_npz.unlink()  # discard partial
            t0 = time.perf_counter()
            built = create_npz_from_sample_folder(str(png_dir), num=NUM_FID_SAMPLES)
            target_npz = target_dir / Path(built).name
            Path(built).rename(target_npz)
            shutil.rmtree(png_dir)
            print(f"  [acquire] recovered in {time.perf_counter()-t0:.1f}s → {target_npz.name}", flush=True)
            return str(target_npz)
        else:
            print(f"  [acquire] partial PNG dir ({n_pngs} files) — discarding", flush=True)
            shutil.rmtree(png_dir)

    if pre_npz.exists():
        pre_npz.unlink()  # stale / partial from prior crash

    # (c) DDP generate via torchrun
    try:
        npz = generate_samples(
            ckpt_path=str(ckpt_path),
            output_dir=str(target_dir),
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
        # success — cleanup any leftover PNG dir
        if png_dir.exists():
            shutil.rmtree(png_dir)
        return npz
    except Exception as e:
        print(f"  [acquire] DDP gen failed: {type(e).__name__}", flush=True)

    # (d) DDP left a complete pre_npz → use it (barrier hung after savez)
    if pre_npz.exists():
        n = _npz_sample_count(str(pre_npz))
        if n >= NUM_FID_SAMPLES:
            target_npz = target_dir / pre_npz.name
            print(f"  [acquire] DDP left complete npz ({n} samples) — moving", flush=True)
            pre_npz.rename(target_npz)
            if png_dir.exists():
                shutil.rmtree(png_dir)
            return str(target_npz)
        print(f"  [acquire] DDP left partial npz ({n} samples) — discarding", flush=True)
        pre_npz.unlink()

    # (e) DDP left enough PNGs → recover
    if png_dir.exists():
        n_pngs = sum(1 for p in png_dir.iterdir() if p.name.endswith(".png"))
        if n_pngs >= NUM_FID_SAMPLES:
            print(f"  [acquire] DDP left {n_pngs} PNGs — recovering", flush=True)
            t0 = time.perf_counter()
            built = create_npz_from_sample_folder(str(png_dir), num=NUM_FID_SAMPLES)
            target_npz = target_dir / Path(built).name
            Path(built).rename(target_npz)
            shutil.rmtree(png_dir)
            print(f"  [acquire] recovered in {time.perf_counter()-t0:.1f}s", flush=True)
            return str(target_npz)
        if png_dir.exists():
            shutil.rmtree(png_dir)

    raise RuntimeError(f"could not acquire npz for {run_name} step={step}")


# ── Score one npz against pre-loaded real stats ───────────────────────────────
def score_one(npz_path, feat_model, device, mu_real, sigma_real, real_kid_feats):
    t0 = time.perf_counter()
    samples = _load_npz_fast(npz_path)
    print(f"    [score] loaded {len(samples)} samples in {time.perf_counter()-t0:.1f}s", flush=True)
    loader = torch.utils.data.DataLoader(
        _NpDataset(samples), batch_size=FID_BATCH_SIZE, num_workers=FID_NUM_WORKERS,
        shuffle=False, drop_last=False, pin_memory=True,
    )
    t1 = time.perf_counter()
    l_feats = []
    for batch in loader:
        with torch.no_grad():
            l_feats.append(feat_model(batch.to(device)).detach().cpu().numpy())
    gen_feats = np.concatenate(l_feats)
    print(f"    [score] InceptionV3 forward in {time.perf_counter()-t1:.1f}s", flush=True)
    t2 = time.perf_counter()
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    fid_val = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"    [score] FID in {time.perf_counter()-t2:.1f}s", flush=True)
    t3 = time.perf_counter()
    kid_val = kernel_distance(real_kid_feats, gen_feats)
    print(f"    [score] KID in {time.perf_counter()-t3:.1f}s", flush=True)
    return float(fid_val), float(kid_val)


# ── Setup ─────────────────────────────────────────────────────────────────────
print("\n" + "#" * 70 + "\n# PHASE 0: setup\n" + "#" * 70, flush=True)
ensure_real_stats_cached(str(REAL_DATA))

device = torch.device("cuda")
feat_model = build_feature_extractor("clean", device, use_dataparallel=True)

import cleanfid as _cfd_pkg
stats_dir = Path(_cfd_pkg.__file__).parent / "stats"
fid_stats = np.load(stats_dir / "celeba256_clean_custom_na.npz")
mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
print("[setup] mu/sigma loaded", flush=True)
real_kid_feats = np.load(stats_dir / "celeba256_clean_custom_na_kid.npz", mmap_mode="r")["feats"]
print(f"[setup] real_kid_feats {real_kid_feats.shape}", flush=True)


def _clean_repa_samples_for_step(step: int):
    """Delete REPA/samples/SiT-B-2-{step:07d}-... and matching .npz, to prevent
    stale PNG dirs from a different run being mis-recovered."""
    basename = f"SiT-B-2-{step:07d}-size-256-vae-ema-cfg-2.0-seed-0-ode"
    png_dir = Path(REPA_ROOT) / "samples" / basename
    pre_npz = Path(REPA_ROOT) / "samples" / f"{basename}.npz"
    if png_dir.exists():
        shutil.rmtree(png_dir)
        print(f"  [cleanup] removed stale PNG dir {basename}", flush=True)
    if pre_npz.exists():
        pre_npz.unlink()
        print(f"  [cleanup] removed stale .npz {basename}.npz", flush=True)


def _clean_all_repa_samples():
    """Wipe everything in REPA/samples/ that matches our basename pattern.
    Called between runs (run_idx > 0) so that PNG dirs left over from the
    previous run are not mis-recovered for the current run."""
    samples = Path(REPA_ROOT) / "samples"
    if not samples.exists():
        return
    n = 0
    for p in samples.iterdir():
        if p.name.startswith("SiT-B-2-") and p.name.endswith("-vae-ema-cfg-2.0-seed-0-ode"):
            shutil.rmtree(p); n += 1
        elif p.name.startswith("SiT-B-2-") and p.name.endswith("-vae-ema-cfg-2.0-seed-0-ode.npz"):
            p.unlink(); n += 1
    if n:
        print(f"[cleanup] wiped {n} entries from REPA/samples/", flush=True)


# ── Main loop ─────────────────────────────────────────────────────────────────
total = len(RUNS) * len(STEPS)
done_count = 0
for run_idx, (run_name, run_subdir) in enumerate(RUNS):
    run_dir = RUNS_ROOT / run_subdir
    print(f"\n{'=' * 70}\n=== RUN {run_idx+1}/{len(RUNS)}: {run_name}\n=== from: {run_dir}\n{'=' * 70}", flush=True)

    # Cleanup between runs: PNG dirs are named by checkpoint step only (no run name),
    # so anything left in REPA/samples/ from the previous run would be silently
    # mis-recovered for the current run. Wipe before starting a new run.
    if run_idx > 0:
        _clean_all_repa_samples()

    for step in STEPS:
        done_count += 1
        key = (run_name, step)
        if key in existing:
            print(f"\n[{done_count}/{total}] {run_name} step={step}: SKIP (in CSV)", flush=True)
            # Even on skip, kill any stale PNG dir for this step — otherwise the
            # same step in the NEXT run would mis-recover from it before the
            # inter-run wipe (which happens earlier than next iteration here).
            _clean_repa_samples_for_step(step)
            continue

        ckpt = run_dir / "checkpoints" / f"{step:07d}.pt"
        if not ckpt.exists():
            print(f"\n[{done_count}/{total}] {run_name} step={step}: MISSING CHECKPOINT {ckpt}", flush=True)
            continue

        target_dir = samples_root / run_name / f"step_{step:07d}"
        print(f"\n{'#' * 70}\n# [{done_count}/{total}] {run_name} step={step}\n{'#' * 70}", flush=True)

        # 1. Acquire npz
        try:
            t_acq = time.perf_counter()
            npz = acquire_npz(run_name, step, ckpt, target_dir)
            print(f"  [acquire] done in {time.perf_counter()-t_acq:.1f}s", flush=True)
        except Exception:
            print(f"!! ACQUIRE FAILED for {run_name} step={step}:")
            traceback.print_exc()
            continue

        # 2. Score
        try:
            t_score = time.perf_counter()
            fid, kid = score_one(npz, feat_model, device, mu_real, sigma_real, real_kid_feats)
            print(f"\n  RESULT {run_name} step={step}: FID={fid:.4f}  KID×10³={kid * 1e3:.4f}  ({time.perf_counter()-t_score:.1f}s)\n", flush=True)
            append_row(run_name, step, fid, kid)
            existing.add(key)
        except Exception:
            print(f"!! SCORE FAILED for {run_name} step={step}:")
            traceback.print_exc()
            continue


csv_file.close()
print(f"\n{'#' * 70}\n# Done. {OUT_CSV}:\n")
with open(OUT_CSV) as f:
    print(f.read())
print(f"{'#' * 70}", flush=True)
