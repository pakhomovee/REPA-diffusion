#!/usr/bin/env python3
"""
FID / KID / LPIPS diversity evaluation across training checkpoints.

Produces:
  fid_results/fid_vs_steps.png       — FID curve vs training steps
  fid_results/kid_vs_steps.png       — KID curve vs training steps
  fid_results/fid_vs_wallclock.png   — FID curve vs cumulative training time (min)
  fid_results/scores.csv             — raw numbers

Usage:
    # Quick test (3 checkpoints, 2 000 samples, ~10 min):
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python fid_kid.py --quick

    # Full run (all 20 checkpoints, 10 000 samples, ~1.5 h):
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python fid_kid.py > logs/fid_kid.log 2>&1 &
"""
import argparse
import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from cleanfid import fid as cleanfid
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, kernel_distance
from cleanfid.resize import build_resizer
import lpips as lpips_lib
from torchvision import transforms


import zipfile, struct as _struct

def _npz_sample_count(path):
    with zipfile.ZipFile(path) as zf:
        with zf.open("arr_0.npy") as f:
            f.read(6)
            major = _struct.unpack("BB", f.read(2))[0]
            hlen = _struct.unpack("<H", f.read(2))[0] if major == 1 else _struct.unpack("<I", f.read(4))[0]
            header = f.read(hlen).decode("latin1")
    m = re.search(r"shape.*?\((\d+)", header)
    return int(m.group(1)) if m else 0




def _load_npz_fast(path, key="arr_0"):
    """Load array from uncompressed NPZ using direct C-level file I/O."""
    import ast as _ast2
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local_hdr = f.read(30)
        fname_len, extra_len = _struct.unpack_from("<HH", local_hdr, 26)
        npy_start = info.header_offset + 30 + fname_len + extra_len
        f.seek(npy_start)
        f.read(6)
        major = f.read(1)[0]
        f.read(1)
        hdr_len = _struct.unpack("<H", f.read(2))[0] if major == 1 else _struct.unpack("<I", f.read(4))[0]
        header = f.read(hdr_len).decode("latin1")
        d = _ast2.literal_eval(header.strip())
        shape, dtype = tuple(d["shape"]), np.dtype(d["descr"])
        nbytes = int(np.prod(shape)) * dtype.itemsize
        raw = f.read(nbytes)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)

# Dataset that serves images directly from a (N,H,W,C) uint8 numpy array.
# Defined at module level so DataLoader workers can pickle it.
class _NpDataset(torch.utils.data.Dataset):
    """Serves images from a (N,H,W,C) uint8 numpy/mmap array using the same
    preprocessing pipeline as cleanfid's ResizeDataset (mode='clean')."""
    def __init__(self, arr):
        self.arr = arr
        self.fn_resize = build_resizer("clean")   # expects numpy (H,W,C), returns float32 [0,255]
        self.to_tensor = transforms.ToTensor()    # float32 (H,W,C) → (C,H,W), no scaling

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        img_np = np.array(self.arr[i])            # materialise mmap slice → (H,W,3) uint8
        img_resized = self.fn_resize(img_np)      # (299,299,3) float32 [0,255]
        return self.to_tensor(img_resized)        # (3,299,299) float32 [0,255]


# ── paths ─────────────────────────────────────────────────────────────────────
# Resolve the project root from this file's location (fid_kid.py lives at the
# repo root) so the pipeline is portable. Override any of these with env vars
# if your data / runs live elsewhere.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
REPA_ROOT    = Path(os.environ.get("REPA_ROOT", PROJECT_ROOT / "REPA"))
RUNS_ROOT    = Path(os.environ.get("RUNS_ROOT", PROJECT_ROOT / "runs"))
REAL_DATA    = Path(os.environ.get("REAL_DATA", PROJECT_ROOT / "data" / "celeba256" / "images"))
TORCHRUN     = os.environ.get("TORCHRUN", "torchrun")  # resolved from PATH

BASELINE_DIR = RUNS_ROOT / "celeba_sit_b2_baseline_gpus2-7_40k"
REPA_DIR     = RUNS_ROOT / "celeba_sit_b2_repa_dinov2b_gpus4-7"

ALL_STEPS = list(range(10_000, 201_000, 10_000))   # 10 k … 200 k


# ── wall-clock helpers ────────────────────────────────────────────────────────
def parse_wallclock_minutes(log_path: Path) -> dict[int, float]:
    """
    Parse training log.txt and return {step: cumulative_training_minutes}.
    Handles logs that span multiple resumed sessions (gaps are excluded).
    Strips ANSI escape codes before matching.
    """
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    ts_re   = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
    ckpt_re = re.compile(r'Saved checkpoint.*?(\d{7})\.pt')
    start_re = re.compile(r'Experiment directory created')
    fmt = "%Y-%m-%d %H:%M:%S"

    sessions: list[tuple[datetime, list[tuple[datetime, int]]]] = []

    with open(log_path) as f:
        session_start = None
        checkpoints = []
        for line in f:
            line = ansi_re.sub('', line)
            ts_m = ts_re.search(line)
            if not ts_m:
                continue
            ts = datetime.strptime(ts_m.group(1), fmt)
            if start_re.search(line):
                if session_start is not None:
                    sessions.append((session_start, checkpoints))
                session_start = ts
                checkpoints = []
            ck_m = ckpt_re.search(line)
            if ck_m and session_start:
                step = int(ck_m.group(1))
                checkpoints.append((ts, step))
        if session_start is not None:
            sessions.append((session_start, checkpoints))

    cumulative_min = 0.0
    result: dict[int, float] = {}
    for session_start, checkpoints in sessions:
        for ts, step in checkpoints:
            elapsed = (ts - session_start).total_seconds() / 60.0
            result[step] = cumulative_min + elapsed
        if checkpoints:
            last_ts = checkpoints[-1][0]
            cumulative_min += (last_ts - session_start).total_seconds() / 60.0

    return result


# ── 1. GENERATION ─────────────────────────────────────────────────────────────
def generate_samples(
    ckpt_path: str,
    output_dir: str,
    model: str = "SiT-B/2",
    num_classes: int = 16,
    num_fid_samples: int = 10_000,
    per_proc_batch_size: int = 64,
    encoder_depth: int = 4,
    projector_embed_dims: str = "768",   # "none" for baseline
    path_type: str = "linear",
    mode: str = "ode",
    num_steps: int = 50,
    cfg_scale: float = 2.0,    # REPA paper optimal is 1.8; user requested 2.0
    guidance_high: float = 0.7,  # Kynkäänniemi et al. NeurIPS 2024; matches REPA paper
    weights: str = "ema",
    repa_root: str = str(REPA_ROOT),
    device_count: int = 1,
) -> str:
    """Generate samples for one checkpoint. Returns path to .npz. Skips if cached."""
    os.makedirs(output_dir, exist_ok=True)

    existing = list(Path(output_dir).glob("*.npz"))
    if existing:
        cached_n = _npz_sample_count(existing[0])
        if cached_n >= num_fid_samples:
            print(f"  [cached] {Path(ckpt_path).stem} → {existing[0].name} ({cached_n} samples)")
            return str(existing[0])
        print(f"  [stale] {existing[0].name} has {cached_n} samples, need {num_fid_samples} — regenerating")
        existing[0].unlink()

    # clear stale .npz files in the generate.py output folder
    samples_dir = Path(repa_root) / "samples"
    samples_dir.mkdir(exist_ok=True)
    for old in samples_dir.glob("*.npz"):
        old.unlink()

    cmd = [
        TORCHRUN, "--nnodes=1", f"--nproc_per_node={device_count}",
        str(Path(repa_root) / "generate.py"),
        f"--model={model}",
        f"--num-classes={num_classes}",
        f"--num-fid-samples={num_fid_samples}",
        f"--ckpt={ckpt_path}",
        f"--path-type={path_type}",
        f"--encoder-depth={encoder_depth}",
        f"--projector-embed-dims={projector_embed_dims}",
        f"--per-proc-batch-size={per_proc_batch_size}",
        f"--mode={mode}",
        f"--num-steps={num_steps}",
        f"--cfg-scale={cfg_scale}",
        f"--guidance-high={guidance_high}",
        f"--weights={weights}",
        "--fused-attn",
        "--vae-chunk-size=64",
    ]

    print(f"\n{'='*60}\nGenerating: {Path(ckpt_path).name}\n{'='*60}")
    subprocess.run(cmd, cwd=repa_root, check=True)

    npz_files = sorted(samples_dir.glob("*.npz"), key=os.path.getmtime, reverse=True)
    assert npz_files, f"No .npz found in {samples_dir} after generation."

    npz_dst = Path(output_dir) / npz_files[0].name
    npz_files[0].rename(npz_dst)
    print(f"Saved: {npz_dst}")
    return str(npz_dst)


def generate_all_checkpoints(
    checkpoints: dict[str, list[str]],
    samples_root: str,
    run_kwargs: dict[str, dict] | None = None,
    **generate_kwargs,
) -> dict[str, list[str]]:
    npz_paths: dict[str, list[str]] = {}
    for run_name, ckpt_list in checkpoints.items():
        npz_paths[run_name] = []
        extra = (run_kwargs or {}).get(run_name, {})
        for ckpt_path in ckpt_list:
            step = Path(ckpt_path).stem
            out_dir = os.path.join(samples_root, run_name, f"step_{step}")
            npz = generate_samples(ckpt_path, out_dir, **{**generate_kwargs, **extra})
            npz_paths[run_name].append(npz)
    return npz_paths


# ── 2. FID / KID ──────────────────────────────────────────────────────────────
REAL_STATS_NAME = "celeba256"   # cleanfid appends _clean_custom_na to this


def ensure_real_stats_cached(real_data_dir: str) -> None:
    """
    Pre-compute and cache InceptionV3 stats for the real dataset once.
    Subsequent compute_fid/compute_kid calls use dataset_name= to skip
    re-extracting features from 202k images each time (saves ~23 min/call).
    """
    if cleanfid.test_stats_exists(REAL_STATS_NAME, mode="clean"):
        print(f"[cache] Real stats already cached as '{REAL_STATS_NAME}'.")
        return
    print(f"Pre-computing real image stats for '{REAL_STATS_NAME}' — this takes ~25 min once …")
    cleanfid.make_custom_stats(REAL_STATS_NAME, real_data_dir, mode="clean")
    print(f"[cache] Real stats saved as '{REAL_STATS_NAME}'.")


def npz_to_img_dir(npz_path: str) -> str:
    """Unpack .npz to a sibling images/ folder. Re-extracts if PNG count doesn't match."""
    img_dir = Path(npz_path).parent / "images"
    samples = np.load(npz_path)["arr_0"]   # (N, H, W, C) uint8
    existing_pngs = list(img_dir.glob("*.png")) if img_dir.exists() else []
    if len(existing_pngs) == len(samples):
        return str(img_dir)
    img_dir.mkdir(exist_ok=True)
    for p in existing_pngs:
        p.unlink()
    print(f"  Extracting {len(samples)} images …")
    from concurrent.futures import ThreadPoolExecutor
    def _save(args):
        i, img = args
        Image.fromarray(img).save(img_dir / f"{i:06d}.png")
    with ThreadPoolExecutor(max_workers=48) as ex:
        list(ex.map(_save, enumerate(samples)))
    return str(img_dir)


def compute_fid_kid(
    npz_paths: dict[str, list[str]],
    checkpoint_steps: list[int],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    device = torch.device("cuda")

    # Build InceptionV3 once, reuse across all 40 checkpoints.
    feat_model = build_feature_extractor("clean", device, use_dataparallel=True)

    # Load real stats once from disk.
    import cleanfid as _cfd_pkg
    stats_dir = Path(_cfd_pkg.__file__).parent / "stats"
    fid_stats  = np.load(stats_dir / "celeba256_clean_custom_na.npz")
    mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
    real_kid_feats = np.load(stats_dir / "celeba256_clean_custom_na_kid.npz",
                             mmap_mode="r")["feats"]   # (202599, 2048), memory-mapped

    fid_scores: dict[str, list[float]] = {}
    kid_scores: dict[str, list[float]] = {}

    for run_name, paths in npz_paths.items():
        fid_scores[run_name] = []
        kid_scores[run_name] = []
        for step, npz_path in zip(checkpoint_steps, paths):
            print(f"\nScoring {run_name} @ step {step:,} …", flush=True)

            # Memory-map the NPZ — no 9.8 GB RAM copy, workers read slices directly.
            samples = _load_npz_fast(npz_path)  # fast C-level read, bypasses Python zipfile

            dataset = _NpDataset(samples)
            loader  = torch.utils.data.DataLoader(
                dataset, batch_size=512, num_workers=48,
                shuffle=False, drop_last=False, pin_memory=True,
            )

            # Single InceptionV3 pass → features used for both FID and KID.
            l_feats = []
            for batch in loader:
                with torch.no_grad():
                    l_feats.append(feat_model(batch.to(device)).detach().cpu().numpy())
            gen_feats = np.concatenate(l_feats)   # (50000, 2048)

            mu_gen    = np.mean(gen_feats, axis=0)
            sigma_gen = np.cov(gen_feats, rowvar=False)
            fid_val   = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
            kid_val   = kernel_distance(real_kid_feats, gen_feats)

            fid_scores[run_name].append(fid_val)
            kid_scores[run_name].append(kid_val)
            print(f"  FID={fid_val:.2f}   KID×10³={kid_val*1e3:.3f}")

    return fid_scores, kid_scores


# ── 3. LPIPS DIVERSITY ────────────────────────────────────────────────────────
def compute_diversity(
    npz_path: str,
    label: str,
    n_pairs: int = 2000,
    batch_size: int = 64,
    net: str = "alex",
    device: str | None = None,
) -> float:
    """
    Mean pairwise LPIPS within one model's generated samples — the diversity
    metric from the paper. Higher = more diverse output distribution.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips_lib.LPIPS(net=net).to(device).eval()

    imgs = _load_npz_fast(npz_path)          # (N, H, W, C) uint8 — fast C-level read
    N = len(imgs)
    assert N >= 2, "Need at least 2 images."

    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # draw random pairs without replacement
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    # avoid self-pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N

    scores = []
    for i in range(0, n_pairs, batch_size):
        ia = idx_a[i:i+batch_size]
        ib = idx_b[i:i+batch_size]
        ba = torch.stack([prep(imgs[j]) for j in ia]).to(device)
        bb = torch.stack([prep(imgs[j]) for j in ib]).to(device)
        with torch.no_grad():
            out = loss_fn(ba, bb).squeeze().cpu()
        scores.extend(out.tolist() if out.dim() > 0 else [out.item()])

    mean_div = float(np.mean(scores))
    print(f"Diversity LPIPS ({label}): {mean_div:.4f}  (n_pairs={n_pairs})")
    return mean_div


# ── 4. PLOTTING ───────────────────────────────────────────────────────────────
COLORS = {"Baseline": "#2196F3", "REPA (DINOv2-B)": "#FF5722"}

def _plot(x_values: dict[str, list], y_values: dict[str, list],
          xlabel: str, ylabel: str, title: str, output_path: str, scale: float = 1.0):
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, ys in y_values.items():
        xs = x_values[run_name]
        ax.plot(xs, [y * scale for y in ys],
                marker="o", label=run_name, color=COLORS.get(run_name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="3 checkpoints, 2 000 samples (~10 min)")
    parser.add_argument("--num-gpus", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=50_000)
    args = parser.parse_args()

    if args.quick:
        STEPS = [40_000, 100_000, 200_000]
        NUM_SAMPLES = 2_000
    else:
        STEPS = ALL_STEPS
        NUM_SAMPLES = args.num_samples

    STEP_STRS = [f"{s:07d}" for s in STEPS]

    checkpoints = {
        "Baseline":        [str(BASELINE_DIR / "checkpoints" / f"{s}.pt") for s in STEP_STRS],
        "REPA (DINOv2-B)": [str(REPA_DIR    / "checkpoints" / f"{s}.pt") for s in STEP_STRS],
    }
    run_kwargs = {
        "Baseline":          {"projector_embed_dims": "none"},
        "REPA (DINOv2-B)":   {"projector_embed_dims": "768"},
    }

    # ── extract wall-clock training times from logs ───────────────────────────
    wallclock: dict[str, dict[int, float]] = {
        "Baseline":        parse_wallclock_minutes(BASELINE_DIR / "log.txt"),
        "REPA (DINOv2-B)": parse_wallclock_minutes(REPA_DIR    / "log.txt"),
    }

    # ── generate samples ──────────────────────────────────────────────────────
    samples_root = str(PROJECT_ROOT / "fid_samples")
    npz_paths = generate_all_checkpoints(
        checkpoints=checkpoints,
        samples_root=samples_root,
        run_kwargs=run_kwargs,
        num_fid_samples=NUM_SAMPLES,
        per_proc_batch_size=512,
        device_count=args.num_gpus,
        repa_root=str(REPA_ROOT),
        weights="ema",
        mode="ode",
    )

    # ── FID / KID ─────────────────────────────────────────────────────────────
    # Cache real image stats once — all 40 checkpoint pairs reuse them
    ensure_real_stats_cached(str(REAL_DATA))

    fid_scores, kid_scores = compute_fid_kid(
        npz_paths=npz_paths,
        checkpoint_steps=STEPS,
    )

    # ── diversity LPIPS at final checkpoint ───────────────────────────────────
    diversities: dict[str, float] = {}
    for run_name in npz_paths:
        diversities[run_name] = compute_diversity(npz_paths[run_name][-1], label=run_name)

    # ── plots ─────────────────────────────────────────────────────────────────
    out_root = PROJECT_ROOT / "fid_results"
    out_root.mkdir(exist_ok=True)

    # x-axis: steps (same for both runs)
    steps_x = {run: STEPS for run in fid_scores}

    _plot(steps_x, fid_scores,
          xlabel="Training Steps", ylabel="FID ↓",
          title="FID vs Training Steps — CelebA 256",
          output_path=str(out_root / "fid_vs_steps.png"))

    _plot(steps_x, kid_scores,
          xlabel="Training Steps", ylabel="KID × 10³ ↓",
          title="KID vs Training Steps — CelebA 256",
          output_path=str(out_root / "kid_vs_steps.png"),
          scale=1e3)

    # x-axis: cumulative training time (minutes) from log timestamps
    wc_x = {
        run: [wallclock[run].get(s, float("nan")) for s in STEPS]
        for run in fid_scores
    }
    _plot(wc_x, fid_scores,
          xlabel="Cumulative Training Time (min, excl. overnight gap)",
          ylabel="FID ↓",
          title="FID vs Wall-Clock Time — CelebA 256",
          output_path=str(out_root / "fid_vs_wallclock.png"))

    # ── CSV summary ───────────────────────────────────────────────────────────
    csv_path = out_root / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "wallclock_min", "fid", "kid_x1e3"])
        for run_name in fid_scores:
            for i, step in enumerate(STEPS):
                w.writerow([
                    run_name, step,
                    f"{wallclock[run_name].get(step, float('nan')):.1f}",
                    f"{fid_scores[run_name][i]:.2f}",
                    f"{kid_scores[run_name][i]*1e3:.4f}",
                ])
    print(f"\nSaved: {csv_path}")

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Step':>10}  {'Baseline FID':>14}  {'REPA FID':>10}  {'Baseline KID×10³':>18}  {'REPA KID×10³':>14}")
    print(f"{'─'*70}")
    for i, step in enumerate(STEPS):
        print(f"{step:>10,}"
              f"  {fid_scores['Baseline'][i]:>14.1f}"
              f"  {fid_scores['REPA (DINOv2-B)'][i]:>10.1f}"
              f"  {kid_scores['Baseline'][i]*1e3:>18.4f}"
              f"  {kid_scores['REPA (DINOv2-B)'][i]*1e3:>14.4f}")

    print(f"\nDiversity (mean pairwise LPIPS at 200k checkpoint):")
    for run_name, div in diversities.items():
        print(f"  {run_name}: {div:.4f}")
