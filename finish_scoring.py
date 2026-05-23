#!/usr/bin/env python3
"""
Finish scoring:
  1. Score missing REPA 200k checkpoint
  2. Compute LPIPS diversity for both runs (fast loader)
  3. Write scores.csv
  4. Generate plots
"""
import os, sys
sys.path.insert(0, "/home/seankopylov/projects/repa_diffusion_celeba")
os.chdir("/home/seankopylov/projects/repa_diffusion_celeba")

import numpy as np
import torch
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from fid_kid import (
    _load_npz_fast, compute_diversity,
    compute_fid_kid, PROJECT_ROOT,
)

# ── Known scores (from log) ───────────────────────────────────────────────────
STEPS = list(range(10_000, 201_000, 10_000))

KNOWN = {
    "Baseline": {
        10000:  (557.02, 0.878728), 20000:  (328.50, 0.374643),
        30000:  (66.36,  0.062759), 40000:  (26.72,  0.021130),
        50000:  (15.24,  0.010757), 60000:  (11.28,  0.007496),
        70000:  (9.70,   0.006393), 80000:  (8.78,   0.005648),
        90000:  (8.21,   0.005259), 100000: (7.85,   0.005174),
        110000: (7.54,   0.005036), 120000: (7.41,   0.004818),
        130000: (7.21,   0.004897), 140000: (7.00,   0.004656),
        150000: (6.89,   0.004715), 160000: (6.87,   0.004640),
        170000: (6.78,   0.004480), 180000: (6.68,   0.004662),
        190000: (6.69,   0.004742), 200000: (6.56,   0.004491),
    },
    "REPA (DINOv2-B)": {
        10000:  (542.72, 0.839128), 20000:  (334.15, 0.386226),
        30000:  (63.68,  0.059682), 40000:  (23.70,  0.018583),
        50000:  (13.96,  0.009967), 60000:  (10.51,  0.007252),
        70000:  (9.10,   0.006190), 80000:  (8.33,   0.005703),
        90000:  (7.78,   0.005364), 100000: (7.29,   0.005066),
        110000: (6.93,   0.004829), 120000: (6.78,   0.004732),
        130000: (6.66,   0.004738), 140000: (6.50,   0.004483),
        150000: (6.39,   0.004614), 160000: (6.33,   0.004602),
        170000: (6.28,   0.004459), 180000: (6.27,   0.004376),
        190000: (6.14,   0.004446),  # 200k missing — will compute now
    },
}

# ── Score missing REPA 200k ───────────────────────────────────────────────────
repa_200k_npz = list(
    (PROJECT_ROOT / "fid_samples" / "REPA (DINOv2-B)" / "step_0200000").glob("*.npz")
)
assert repa_200k_npz, "REPA 200k NPZ not found!"
repa_200k_npz = str(repa_200k_npz[0])

print("Scoring REPA 200k (the missing one)…", flush=True)

import cleanfid as _cfd
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance, kernel_distance
from fid_kid import _NpDataset

device = torch.device("cuda")
feat_model = build_feature_extractor("clean", device, use_dataparallel=True)

stats_dir = Path(_cfd.__file__).parent / "stats"
fid_stats = np.load(stats_dir / "celeba256_clean_custom_na.npz")
mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
real_kid_feats = np.load(stats_dir / "celeba256_clean_custom_na_kid.npz", mmap_mode="r")["feats"]

imgs = _load_npz_fast(repa_200k_npz)
loader = torch.utils.data.DataLoader(
    _NpDataset(imgs), batch_size=512, num_workers=16,
    shuffle=False, drop_last=False, pin_memory=True,
)
feats = []
for batch in loader:
    with torch.no_grad():
        feats.append(feat_model(batch.to(device)).detach().cpu().numpy())
gen_feats = np.concatenate(feats)

fid_200 = frechet_distance(mu_real, sigma_real,
                           np.mean(gen_feats, 0), np.cov(gen_feats, rowvar=False))
kid_200 = kernel_distance(real_kid_feats, gen_feats)
KNOWN["REPA (DINOv2-B)"][200000] = (float(fid_200), float(kid_200))
print(f"  REPA 200k: FID={fid_200:.2f}  KID×10³={kid_200*1e3:.3f}", flush=True)

del feat_model, imgs, feats, gen_feats, loader

# ── Diversity ─────────────────────────────────────────────────────────────────
FID_SAMPLES_ROOT = PROJECT_ROOT / "fid_samples"
diversities = {}
for run_name in ("Baseline", "REPA (DINOv2-B)"):
    npz_list = list((FID_SAMPLES_ROOT / run_name / "step_0200000").glob("*.npz"))
    if not npz_list:
        print(f"  WARN: no 200k NPZ for {run_name}, skipping diversity")
        continue
    print(f"\nDiversity for {run_name}…", flush=True)
    diversities[run_name] = compute_diversity(str(npz_list[0]), label=run_name)

# ── Write scores.csv ──────────────────────────────────────────────────────────
out_csv = PROJECT_ROOT / "scores.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "step", "fid", "kid_e3"])
    for run_name in ("Baseline", "REPA (DINOv2-B)"):
        for step in STEPS:
            fid, kid = KNOWN[run_name][step]
            w.writerow([run_name, step, f"{fid:.4f}", f"{kid*1e3:.6f}"])
print(f"\nScores saved → {out_csv}", flush=True)

# ── Plots ─────────────────────────────────────────────────────────────────────
out_dir = PROJECT_ROOT / "fid_results"
out_dir.mkdir(exist_ok=True)
BLUE   = "#2196F3"; ORANGE = "#FF5722"
COLORS = {"Baseline": BLUE, "REPA (DINOv2-B)": ORANGE}
MARKERS = {"Baseline": "o", "REPA (DINOv2-B)": "s"}

for metric, ylabel, fname, scale in [
    ("fid", "FID ↓",      "fid_vs_steps",     1.0),
    ("kid", "KID×10³ ↓",  "kid_vs_steps",     1e3),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name in ("Baseline", "REPA (DINOv2-B)"):
        xs = STEPS
        ys = [KNOWN[run_name][s][0 if metric=="fid" else 1] * scale for s in xs]
        ax.plot(xs, ys, color=COLORS[run_name], marker=MARKERS[run_name],
                ms=5, lw=2, label=run_name)
    ax.set_xlabel("Training Steps"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel.split()[0]} vs Training Steps — SiT-B/2 CelebA-256")
    ax.set_yscale("log")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(STEPS[::2])
    ax.set_xticklabels([f"{s//1000}k" for s in STEPS[::2]])
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_dir}/{fname}.png")

print("\nDone!")
