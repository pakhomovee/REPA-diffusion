#!/usr/bin/env python3
"""
score_bloop_extension.py — FID/KID for the 10 NEW bloop ckpts (220k→400k).

The original score_400k_and_bloop.py covered:
  - baseline 200k→400k @ 20k (10 ckpts)   — done
  - REPA-const 200k→400k @ 20k (10 ckpts) — done
  - bloop 0→200k @ 20k (10 ckpts)         — done

This script covers the bloop 200k→400k extension:
  - REPA-Σ bloop (EMA-stabilized) 220k→400k @ every 20k (10 ckpts)

Output is APPENDED to the same CSV (scores_400k_and_bloop.csv) under the
same run name "REPA-Sigma-bloop (DINOv2-B)" so the full trajectory
(20k→400k) sits in one CSV. The resume-via-CSV logic naturally skips the
30 rows already present.

All robustness fixes from the prior run are preserved:
  - FID_NUM_WORKERS = 8 (was 48)
  - gc.collect() + torch.cuda.empty_cache() after each ckpt
  - explicit del of samples/loaders/feats in score_one
  - NCCL timeout 30 min, async error handling
  - PRIORITY-ORDERED npz acquisition with PNG recovery
"""
import os, sys, csv, time, shutil, traceback, gc
# Raise NCCL barrier timeout for sample-gen DDP from 10 min → 30 min.
# Under cluster load the ALLREDUCE in barrier() can legitimately need >10 min.
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # seconds, applied to NCCL ops
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
# Only the bloop 220k→400k extension here. Run name matches the existing CSV
# entries (20k→200k), so resume-via-CSV ignores them and only processes new
# steps. Output CSV path unchanged.
RUNS = [
    (
        "REPA-Sigma-bloop (DINOv2-B)",
        "celeba_sit_b2_repa_sigma_bloop_gpus5-7",
        list(range(220_000, 401_000, 20_000)),  # 220k..400k @ 20k (10 ckpts)
    ),
]

NUM_FID_SAMPLES     = 50_000
PER_PROC_BATCH_SIZE = 512
DEVICE_COUNT        = int(os.environ.get("NUM_GPUS", "6"))
FID_BATCH_SIZE      = 512
# Reduced from 48 to 8: with InceptionV3 forward being the bottleneck (not data
# loading), 48 worker processes were just creating pinned-memory pressure +
# inter-process overhead. 8 workers comfortably saturate the 6-GPU DataParallel
# pipeline and avoid the resource leaks that may have caused the monotonic
# slow-down we observed (score time grew 80 s → 26 000 s over 26 ckpts).
FID_NUM_WORKERS     = 8

samples_root = PROJECT_ROOT / "fid_samples"
samples_root.mkdir(exist_ok=True)
OUT_CSV = PROJECT_ROOT / "scores_400k_and_bloop.csv"


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


# ── Acquire npz (priority-ordered, robust against NCCL hangs) ─────────────────
def acquire_npz(run_name, step, ckpt_path, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # (a) already in target → use
    existing_npz = list(target_dir.glob("*.npz"))
    if existing_npz:
        n = _npz_sample_count(str(existing_npz[0]))
        if n >= NUM_FID_SAMPLES:
            print(f"  [acquire] target has {n} samples — using {existing_npz[0].name}", flush=True)
            return str(existing_npz[0])
        print(f"  [acquire] target has only {n} — re-acquiring", flush=True)
        existing_npz[0].unlink()

    basename = f"SiT-B-2-{step:07d}-size-256-vae-ema-cfg-2.0-seed-0-ode"
    repa_samples = Path(REPA_ROOT) / "samples"
    pre_npz = repa_samples / f"{basename}.npz"
    png_dir = repa_samples / basename

    # (b) full PNG dir already present → recover, no DDP
    if png_dir.exists():
        n_pngs = sum(1 for p in png_dir.iterdir() if p.name.endswith(".png"))
        if n_pngs >= NUM_FID_SAMPLES:
            print(f"  [acquire] recovering from existing PNG dir ({n_pngs} files)", flush=True)
            if pre_npz.exists():
                pre_npz.unlink()
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
        pre_npz.unlink()  # stale/partial

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
        if png_dir.exists():
            shutil.rmtree(png_dir)
        return npz
    except Exception as e:
        print(f"  [acquire] DDP gen failed: {type(e).__name__}", flush=True)

    # (d) DDP left a complete pre_npz — use it
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

    # (e) DDP left enough PNGs — recover
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
    try:
        for batch in loader:
            with torch.no_grad():
                l_feats.append(feat_model(batch.to(device)).detach().cpu().numpy())
    finally:
        # Explicitly drop DataLoader so its worker processes are torn down
        # before we move on to the next ckpt (avoids worker accumulation).
        del loader
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
    # Drop large arrays explicitly so the next iteration starts from a clean
    # heap. Without this, samples (≈9.4 GB) + gen_feats (≈400 MB) + the list
    # of per-batch feats stay reachable until the next assignment, increasing
    # memory pressure under cluster contention.
    del samples, l_feats, gen_feats, mu_gen, sigma_gen
    return float(fid_val), float(kid_val)


# ── Cleanup helpers ───────────────────────────────────────────────────────────
def _clean_repa_samples_for_step(step: int):
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


def _delete_target_npz(npz_path: str, target_dir: Path):
    """After scoring, drop the npz and the (now-empty) step_NNNNNNN dir to keep
    disk pressure ≤ ~9 GB peak. The CSV row is the durable artifact."""
    try:
        size_bytes = Path(npz_path).stat().st_size
        Path(npz_path).unlink()
        print(f"  [cleanup] removed npz {Path(npz_path).name} (~{size_bytes/1024**3:.1f} GB freed)", flush=True)
    except FileNotFoundError:
        pass
    try:
        # remove the per-step dir only if empty (paranoid: don't nuke unexpected siblings)
        target_dir.rmdir()
    except OSError:
        pass


# ── Setup (run once) ──────────────────────────────────────────────────────────
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


# ── Main loop ─────────────────────────────────────────────────────────────────
total = sum(len(steps) for _, _, steps in RUNS)
done_count = 0
t_overall = time.perf_counter()
for run_idx, (run_name, run_subdir, STEPS) in enumerate(RUNS):
    run_dir = RUNS_ROOT / run_subdir
    print(f"\n{'=' * 70}\n=== RUN {run_idx+1}/{len(RUNS)}: {run_name}\n=== from: {run_dir}\n=== steps: {STEPS[0]}..{STEPS[-1]} ({len(STEPS)} ckpts)\n{'=' * 70}", flush=True)

    # Wipe REPA/samples/ between runs so PNG dirs from previous run aren't
    # mis-recovered (their basenames depend only on step, not run name).
    if run_idx > 0:
        _clean_all_repa_samples()

    for step in STEPS:
        done_count += 1
        key = (run_name, step)
        if key in existing:
            print(f"\n[{done_count}/{total}] {run_name} step={step}: SKIP (in CSV)", flush=True)
            _clean_repa_samples_for_step(step)
            continue

        ckpt = run_dir / "checkpoints" / f"{step:07d}.pt"
        if not ckpt.exists():
            print(f"\n[{done_count}/{total}] {run_name} step={step}: MISSING CHECKPOINT {ckpt}", flush=True)
            continue

        target_dir = samples_root / run_name / f"step_{step:07d}"
        elapsed_h = (time.perf_counter() - t_overall) / 3600
        print(f"\n{'#' * 70}\n# [{done_count}/{total}] {run_name} step={step}  (t={elapsed_h:.2f}h)\n{'#' * 70}", flush=True)

        # 1. Acquire npz
        npz = None
        try:
            t_acq = time.perf_counter()
            npz = acquire_npz(run_name, step, ckpt, target_dir)
            print(f"  [acquire] done in {time.perf_counter()-t_acq:.1f}s", flush=True)
        except Exception:
            print(f"!! ACQUIRE FAILED for {run_name} step={step}:")
            traceback.print_exc()
            continue

        # 2+3. Score + append to CSV
        try:
            t_score = time.perf_counter()
            fid, kid = score_one(npz, feat_model, device, mu_real, sigma_real, real_kid_feats)
            print(f"\n  RESULT {run_name} step={step}: FID={fid:.4f}  KID×10³={kid * 1e3:.4f}  ({time.perf_counter()-t_score:.1f}s)\n", flush=True)
            append_row(run_name, step, fid, kid)
            existing.add(key)
        except Exception:
            print(f"!! SCORE FAILED for {run_name} step={step}:")
            traceback.print_exc()
            # Keep the npz around so a re-run can resume from it; don't delete on failure.
            continue

        # 4. Cleanup npz to keep disk bounded
        _delete_target_npz(npz, target_dir)

        # 5. Heap + CUDA cleanup. Without this, GPU memory fragments steadily
        #    across iterations: DataParallel scatter/gather slows down, and
        #    the score time grew 80 s → 26 000 s over 26 ckpts in the prior
        #    run. gc.collect() forces release of Python-side references
        #    (samples, l_feats, gen_feats); empty_cache() returns cached
        #    allocator blocks to the driver.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Final cleanup of REPA/samples/ leftovers
_clean_all_repa_samples()

csv_file.close()
total_h = (time.perf_counter() - t_overall) / 3600
print(f"\n{'#' * 70}\n# Done in {total_h:.2f}h. {OUT_CSV}:\n")
with open(OUT_CSV) as f:
    print(f.read())
print(f"{'#' * 70}", flush=True)
