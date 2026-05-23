#!/usr/bin/env python3
"""
probe_gradient_geometry.py — Measure timestep-stratified gradient compatibility
between the denoising loss and the REPA alignment loss across training checkpoints.

For each (checkpoint, t value, microbatch) cell this records:
    cos(g_diff, g_repa)           — geometric alignment
    ||g_repa|| / ||g_diff||       — relative magnitude (the r_t ratio from main.pdf)
    ||g_diff||, ||g_repa||        — raw norms

Two scopes are reported:
    "full"          — every SiT parameter except the REPA projector head h_psi
    "blocks_0_3"    — only the parameters reached by ∇L_REPA: x/t/y embedders +
                      transformer blocks [0, encoder_depth). Use this for the
                      meaningful geometric comparison; "full" is HASTE's view.

The probe set is a deterministic slice of the CelebA training set; the same
(image, noise, t) tuple is used across every cell so that across-cell
differences are not data variance.

Usage:
    /home/seankopylov/.venv/bin/python scripts/probe_gradient_geometry.py \\
        --run-dir runs/celeba_sit_b2_repa_dinov2b_gpus4-7 \\
        --gpus 3,4,5,6,7

    # Re-run with EMA weights, reusing the cached probe data:
    /home/seankopylov/.venv/bin/python scripts/probe_gradient_geometry.py \\
        --run-dir runs/celeba_sit_b2_repa_dinov2b_gpus4-7 \\
        --gpus 3,4,5,6,7 --weights ema --skip-prepare
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/seankopylov/projects/repa_diffusion_celeba")
REPA_ROOT = PROJECT_ROOT / "REPA"
sys.path.insert(0, str(REPA_ROOT))

# These imports require REPA on sys.path.
from models.sit import SiT_models  # noqa: E402
from dataset import CustomDataset  # noqa: E402
from utils import load_encoders  # noqa: E402


# ── Probe-data caching ────────────────────────────────────────────────────────
LATENTS_SCALE = 0.18215  # SD-VAE
LATENTS_BIAS = 0.0


def sample_posterior(moments: torch.Tensor) -> torch.Tensor:
    """Mirror of REPA/train.py:sample_posterior, but already includes scale+bias."""
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * LATENTS_SCALE + LATENTS_BIAS


def _preprocess_for_dinov2(x_uint8: torch.Tensor) -> torch.Tensor:
    """Mirror of REPA/train.py:preprocess_raw_image for dinov2: x is uint8 (B,3,256,256)."""
    x = x_uint8.float() / 255.0
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = F.interpolate(x, size=224, mode="bicubic")
    return x


def prepare_probe_data(
    data_dir: Path,
    n_samples: int,
    n_microbatches: int,
    batch_size: int,
    seed: int,
    cache_path: Path,
    device: torch.device,
) -> None:
    """Build and cache the frozen probe set.

    Saves a .pt file with:
        latents      (N, 4, 32, 32)  fp32   — VAE-sampled, scaled, biased
        raw_images   (N, 3, 256, 256) uint8 — for any future use
        labels       (N,)             int64
        teacher_feats (N, 256, 768)   fp32   — DINOv2-B patch tokens
        eps          (n_microbatches, batch_size, 4, 32, 32) fp32  — frozen noise

    The sample indices are picked deterministically by seed from CustomDataset.
    """
    assert n_samples == n_microbatches * batch_size, (
        f"n_samples ({n_samples}) must equal n_microbatches*batch_size "
        f"({n_microbatches}*{batch_size}={n_microbatches * batch_size})"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Loading CustomDataset from {data_dir} …", flush=True)
    dataset = CustomDataset(str(data_dir))
    n_total = len(dataset)
    print(f"[prepare] Dataset size = {n_total:,}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)[:n_samples].tolist()

    raw_imgs = []
    moments = []
    labels = []
    for i, idx in enumerate(perm):
        img, feat, lbl = dataset[idx]
        raw_imgs.append(img)  # uint8 (3,256,256)
        moments.append(feat)  # fp32 (1,8,32,32) — moments
        labels.append(lbl)
    raw_imgs_t = torch.stack(raw_imgs).to(torch.uint8)
    moments_t = torch.stack(moments).squeeze(1).float()  # (N, 8, 32, 32)
    labels_t = torch.stack(labels).long()
    print(f"[prepare] Stacked: raw {tuple(raw_imgs_t.shape)} moments {tuple(moments_t.shape)} labels {tuple(labels_t.shape)}")

    # Deterministic VAE sample (one shot)
    g_vae = torch.Generator(device="cpu").manual_seed(seed + 1)
    eps_vae = torch.randn(moments_t.shape[0], 4, 32, 32, generator=g_vae)
    mean, std = torch.chunk(moments_t, 2, dim=1)
    latents = (mean + std * eps_vae) * LATENTS_SCALE + LATENTS_BIAS  # (N,4,32,32)
    print(f"[prepare] VAE-sampled latents: shape {tuple(latents.shape)}, "
          f"mean={latents.mean():.3f} std={latents.std():.3f}")

    # Teacher features (DINOv2-B), one frozen pass
    print(f"[prepare] Loading DINOv2-B and computing teacher features on {device} …")
    encoders, encoder_types, _ = load_encoders("dinov2-vit-b", device, 256)
    assert len(encoders) == 1
    encoder = encoders[0]
    encoder.eval()
    teacher_feats_chunks = []
    with torch.no_grad():
        for i in range(0, n_samples, 32):
            batch = raw_imgs_t[i:i + 32].to(device)
            batch_pp = _preprocess_for_dinov2(batch)
            out = encoder.forward_features(batch_pp)
            assert "x_norm_patchtokens" in out, f"Got DINOv2 output keys {list(out.keys())}"
            feats = out["x_norm_patchtokens"].float()  # (b, 256, 768)
            teacher_feats_chunks.append(feats.cpu())
    teacher_feats = torch.cat(teacher_feats_chunks, dim=0)
    print(f"[prepare] Teacher feats: shape {tuple(teacher_feats.shape)}, "
          f"mean={teacher_feats.mean():.3f} std={teacher_feats.std():.3f}")
    del encoder, encoders
    torch.cuda.empty_cache()

    # Frozen noise tensor for the probe — same eps per (mb, sample) used across
    # every (ckpt, t) cell so cross-cell differences are not noise variance.
    g_eps = torch.Generator(device="cpu").manual_seed(seed + 2)
    eps_probe = torch.randn(n_microbatches, batch_size, 4, 32, 32, generator=g_eps)

    torch.save({
        "latents": latents,
        "raw_images": raw_imgs_t,
        "labels": labels_t,
        "teacher_feats": teacher_feats,
        "eps": eps_probe,
        "indices": perm,
        "meta": {
            "seed": seed,
            "n_samples": n_samples,
            "n_microbatches": n_microbatches,
            "batch_size": batch_size,
            "latents_scale": LATENTS_SCALE,
            "latents_bias": LATENTS_BIAS,
        },
    }, str(cache_path))
    print(f"[prepare] Wrote probe cache: {cache_path}")


# ── Model loading ─────────────────────────────────────────────────────────────
def _strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def build_model(args_dict: dict, z_dims: list, device: torch.device) -> torch.nn.Module:
    """Build a SiT model with the same kwargs that train.py used."""
    latent_size = args_dict["resolution"] // 8
    block_kwargs = {
        "fused_attn": args_dict.get("fused_attn", True),
        "qk_norm": args_dict.get("qk_norm", False),
    }
    model = SiT_models[args_dict["model"]](
        input_size=latent_size,
        num_classes=args_dict["num_classes"],
        use_cfg=(args_dict.get("cfg_prob", 0.0) > 0),
        z_dims=z_dims,
        encoder_depth=args_dict["encoder_depth"],
        **block_kwargs,
    )
    return model.to(device)


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, weights: str) -> int:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if weights == "ema":
        sd = _strip_module_prefix(ckpt["ema"])
    elif weights == "raw":
        sd = _strip_module_prefix(ckpt["model"])
    else:
        raise ValueError(f"weights must be 'raw' or 'ema', got {weights!r}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load] missing keys (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"  [load] unexpected keys (first 5): {unexpected[:5]}", flush=True)
    return int(ckpt.get("steps", -1))


# ── Gradient collection ──────────────────────────────────────────────────────
def make_scope_masks(model: torch.nn.Module, encoder_depth: int) -> dict[str, list[str]]:
    """Pre-compute the parameter name lists for each scope."""
    pre_projector_blocks = {f"blocks.{i}." for i in range(encoder_depth)}
    full_names, narrow_names = [], []
    for name, _ in model.named_parameters():
        if name.startswith("projectors."):
            continue
        full_names.append(name)
        if (
            name.startswith("x_embedder.")
            or name.startswith("t_embedder.")
            or name.startswith("y_embedder.")
            or any(name.startswith(p) for p in pre_projector_blocks)
        ):
            narrow_names.append(name)
    return {"full": full_names, "blocks_0_3": narrow_names}


def collect_flat_grad(model: torch.nn.Module, names: list[str]) -> torch.Tensor:
    """Concatenate flattened gradients for the named parameters; treat None as zeros."""
    parts = []
    name_to_param = dict(model.named_parameters())
    for n in names:
        p = name_to_param[n]
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), device=p.device))
        else:
            parts.append(p.grad.detach().flatten())
    return torch.cat(parts)


# ── Per-cell measurement ─────────────────────────────────────────────────────
def repa_proj_loss(zs_tilde: list[torch.Tensor], teacher_feats: torch.Tensor) -> torch.Tensor:
    """Mirror of REPA/loss.py SILoss projection-loss block.

    zs_tilde is a list of tensors (one per teacher). Here we use exactly one teacher,
    so we expect len == 1 and shape (B, T, C).
    """
    assert len(zs_tilde) == 1, f"Expected 1 teacher, got {len(zs_tilde)}"
    z_tilde = F.normalize(zs_tilde[0], dim=-1)
    z = F.normalize(teacher_feats, dim=-1)
    # mean over patches, mean over batch — matches SILoss: mean_flat(-(z*ẑ).sum(-1)) / bsz
    per_sample = -(z * z_tilde).sum(dim=-1).mean(dim=-1)  # (B,)
    return per_sample.mean()


def measure_cell(
    model: torch.nn.Module,
    latents: torch.Tensor,        # (B, 4, 32, 32)
    eps: torch.Tensor,             # (B, 4, 32, 32)
    labels: torch.Tensor,          # (B,)
    teacher_feats: torch.Tensor,   # (B, 256, 768)
    t_value: float,
    scope_names: dict[str, list[str]],
    proj_coeff: float,
) -> dict:
    """One forward + two backwards; returns one row per scope."""
    B = latents.size(0)
    device = latents.device
    t = torch.full((B, 1, 1, 1), t_value, device=device, dtype=latents.dtype)
    alpha_t = 1.0 - t
    sigma_t = t
    d_alpha_t = -1.0
    d_sigma_t = 1.0
    model_input = alpha_t * latents + sigma_t * eps
    target = d_alpha_t * latents + d_sigma_t * eps

    model.zero_grad(set_to_none=True)
    out, zs_tilde = model(model_input, t.flatten(), y=labels)
    loss_diff = (out - target).pow(2).mean()
    loss_repa = repa_proj_loss(zs_tilde, teacher_feats) * proj_coeff

    # First backward: denoising. Keep graph for the second backward.
    loss_diff.backward(retain_graph=True)
    g_diff = {scope: collect_flat_grad(model, names) for scope, names in scope_names.items()}

    model.zero_grad(set_to_none=True)
    loss_repa.backward()
    g_repa = {scope: collect_flat_grad(model, names) for scope, names in scope_names.items()}

    rows = []
    for scope in scope_names:
        gd = g_diff[scope]
        gr = g_repa[scope]
        gd_norm = float(gd.norm())
        gr_norm = float(gr.norm())
        cos = float((gd * gr).sum() / (gd_norm * gr_norm + 1e-30))
        rows.append({
            "scope": scope,
            "cos_sim": cos,
            "g_diff_norm": gd_norm,
            "g_repa_norm": gr_norm,
            "ratio": gr_norm / (gd_norm + 1e-30),
            "loss_diff": float(loss_diff.detach()),
            "loss_repa_scaled": float(loss_repa.detach()),
        })
    return rows


# ── Worker process ───────────────────────────────────────────────────────────
def worker_main(
    rank: int,
    gpu_id: int,
    ckpt_paths: list[Path],
    args_dict: dict,
    z_dims: list,
    probe_cache: Path,
    t_values: list[float],
    weights: str,
    out_csv: Path,
    proj_coeff: float,
) -> None:
    """One worker per GPU. Processes its assigned subset of checkpoints."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    pid = os.getpid()
    print(f"[worker {rank}|gpu {gpu_id}|pid {pid}] start; ckpts={[p.stem for p in ckpt_paths]}", flush=True)

    cache = torch.load(str(probe_cache), map_location="cpu", weights_only=False)
    latents_cpu = cache["latents"]                 # (N, 4, 32, 32)
    labels_cpu = cache["labels"]                   # (N,)
    teacher_cpu = cache["teacher_feats"]           # (N, 256, 768)
    eps_cpu = cache["eps"]                          # (n_mb, B, 4, 32, 32)
    n_mb, B = eps_cpu.shape[:2]

    model = build_model(args_dict, z_dims, device)
    scope_names = make_scope_masks(model, args_dict["encoder_depth"])
    print(f"[worker {rank}] scope sizes: "
          f"full={sum(dict(model.named_parameters())[n].numel() for n in scope_names['full']):,} "
          f"blocks_0_3={sum(dict(model.named_parameters())[n].numel() for n in scope_names['blocks_0_3']):,}",
          flush=True)

    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "ckpt_step", "t", "microbatch", "weights", "scope",
            "cos_sim", "ratio", "g_diff_norm", "g_repa_norm",
            "loss_diff", "loss_repa_scaled",
        ])
        wr.writeheader()

        for ckpt_path in ckpt_paths:
            t0 = time.perf_counter()
            steps = load_checkpoint_into_model(model, ckpt_path, weights)
            model.eval()  # disable y_embedder dropout — deterministic gradients
            # Make sure all params receive gradients (eval doesn't change requires_grad).
            for p in model.parameters():
                p.requires_grad_(True)
            print(f"[worker {rank}] loaded {ckpt_path.name} ({weights}) steps={steps} "
                  f"in {time.perf_counter()-t0:.1f}s", flush=True)

            for t_val in t_values:
                for mb in range(n_mb):
                    idx_lo = mb * B
                    idx_hi = idx_lo + B
                    latents = latents_cpu[idx_lo:idx_hi].to(device)
                    labels = labels_cpu[idx_lo:idx_hi].to(device)
                    teacher = teacher_cpu[idx_lo:idx_hi].to(device)
                    eps = eps_cpu[mb].to(device)

                    rows = measure_cell(
                        model=model,
                        latents=latents,
                        eps=eps,
                        labels=labels,
                        teacher_feats=teacher,
                        t_value=t_val,
                        scope_names=scope_names,
                        proj_coeff=proj_coeff,
                    )
                    for r in rows:
                        r.update({
                            "ckpt_step": steps,
                            "t": t_val,
                            "microbatch": mb,
                            "weights": weights,
                        })
                        wr.writerow(r)
                    f.flush()
                print(f"[worker {rank}] ckpt={ckpt_path.stem} t={t_val:.2f} done", flush=True)

    print(f"[worker {rank}] finished; csv={out_csv}", flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_results(csv_path: Path, out_dir: Path, weights: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[df["weights"] == weights]
    if df.empty:
        print(f"[plot] no rows for weights={weights}; skipping plot")
        return

    for scope in sorted(df["scope"].unique()):
        sub = df[df["scope"] == scope]
        agg = sub.groupby(["ckpt_step", "t"]).agg(
            cos_mean=("cos_sim", "mean"),
            cos_std=("cos_sim", "std"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
        ).reset_index()

        ckpt_steps = sorted(agg["ckpt_step"].unique())
        cmap = plt.get_cmap("viridis")
        colors = {s: cmap(i / max(1, len(ckpt_steps) - 1)) for i, s in enumerate(ckpt_steps)}

        # Cosine vs t
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in ckpt_steps:
            row = agg[agg["ckpt_step"] == s].sort_values("t")
            ax.plot(row["t"], row["cos_mean"], marker="o", label=f"{s//1000}k", color=colors[s])
            ax.fill_between(row["t"],
                            row["cos_mean"] - row["cos_std"],
                            row["cos_mean"] + row["cos_std"],
                            alpha=0.15, color=colors[s])
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("Diffusion timestep t")
        ax.set_ylabel(r"cos($\nabla L_\mathrm{diff}$, $\nabla \lambda L_\mathrm{REPA}$)")
        ax.set_title(f"Gradient compatibility vs t  ({scope}, weights={weights})")
        ax.legend(title="ckpt step", loc="best", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()
        png = out_dir / f"cos_vs_t__{scope}__{weights}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"[plot] {png}")

        # Ratio vs t
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in ckpt_steps:
            row = agg[agg["ckpt_step"] == s].sort_values("t")
            ax.plot(row["t"], row["ratio_mean"], marker="o", label=f"{s//1000}k", color=colors[s])
            ax.fill_between(row["t"],
                            row["ratio_mean"] - row["ratio_std"],
                            row["ratio_mean"] + row["ratio_std"],
                            alpha=0.15, color=colors[s])
        ax.set_yscale("log")
        ax.set_xlabel("Diffusion timestep t")
        ax.set_ylabel(r"$\Vert\nabla \lambda L_\mathrm{REPA}\Vert / \Vert\nabla L_\mathrm{diff}\Vert$")
        ax.set_title(f"REPA/diff gradient norm ratio vs t  ({scope}, weights={weights})")
        ax.legend(title="ckpt step", loc="best", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4, which="both")
        fig.tight_layout()
        png = out_dir / f"ratio_vs_t__{scope}__{weights}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"[plot] {png}")


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to the training run directory (must contain args.json and checkpoints/).")
    p.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "celeba256"))
    p.add_argument("--ckpt-steps", type=str, default="10000,50000,100000,150000,200000")
    p.add_argument("--t-values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--n-microbatches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpus", type=str, default="0",
                   help="Comma-separated GPU ids, e.g. 3,4,5,6,7")
    p.add_argument("--weights", choices=["raw", "ema"], default="raw")
    p.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "reports" / "grad_geometry"))
    p.add_argument("--probe-cache", type=str, default=None,
                   help="Path to probe-data .pt cache. Default: <out-dir>/probe_data.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-prepare", action="store_true",
                   help="Reuse existing probe cache; skip the data-prep stage.")
    p.add_argument("--prepare-only", action="store_true",
                   help="Build the probe cache and exit.")
    p.add_argument("--prepare-gpu", type=int, default=None,
                   help="GPU id to use for the DINOv2 forward during prepare. "
                        "Defaults to the first id in --gpus.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_cache = Path(args.probe_cache).resolve() if args.probe_cache else out_dir / "probe_data.pt"

    # Read training args
    with open(run_dir / "args.json") as f:
        args_dict = json.load(f)
    print(f"[main] run_dir={run_dir}")
    print(f"[main] model={args_dict['model']} num_classes={args_dict['num_classes']} "
          f"encoder_depth={args_dict['encoder_depth']} enc_type={args_dict.get('enc_type')} "
          f"proj_coeff={args_dict.get('proj_coeff')}")
    if args_dict.get("enc_type") != "dinov2-vit-b":
        print(f"[main] WARNING: this probe assumes enc_type=dinov2-vit-b but got "
              f"{args_dict.get('enc_type')!r}. Teacher features in the cache will mismatch.")
    proj_coeff = float(args_dict.get("proj_coeff", 0.5))

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU id")
    ckpt_steps = [int(s) for s in args.ckpt_steps.split(",") if s.strip() != ""]
    t_values = [float(v) for v in args.t_values.split(",") if v.strip() != ""]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_paths = []
    for s in ckpt_steps:
        p = ckpt_dir / f"{s:07d}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        ckpt_paths.append(p)
    print(f"[main] ckpts={[p.name for p in ckpt_paths]}")
    print(f"[main] t_values={t_values}")
    print(f"[main] gpus={gpus} weights={args.weights}")

    n_samples = args.n_microbatches * args.batch_size

    # ── Stage 1: probe-data cache ─────────────────────────────────────────────
    if args.skip_prepare and probe_cache.exists():
        print(f"[main] skip-prepare: reusing {probe_cache}")
    else:
        prep_gpu = args.prepare_gpu if args.prepare_gpu is not None else gpus[0]
        prep_device = torch.device(f"cuda:{prep_gpu}")
        torch.cuda.set_device(prep_gpu)
        prepare_probe_data(
            data_dir=data_dir,
            n_samples=n_samples,
            n_microbatches=args.n_microbatches,
            batch_size=args.batch_size,
            seed=args.seed,
            cache_path=probe_cache,
            device=prep_device,
        )
        if args.prepare_only:
            print("[main] prepare-only: exiting")
            return

    # ── Stage 2: per-worker measurement ───────────────────────────────────────
    # z_dims must match teacher embed dim for SiT projector head. DINOv2-B = 768.
    z_dims = [768]

    # Distribute checkpoints round-robin across workers
    worker_ckpts: list[list[Path]] = [[] for _ in gpus]
    for i, p in enumerate(ckpt_paths):
        worker_ckpts[i % len(gpus)].append(p)
    worker_csvs = [out_dir / f"measurements__worker{r}_{args.weights}.csv" for r in range(len(gpus))]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (gpu_id, ckpts, csvp) in enumerate(zip(gpus, worker_ckpts, worker_csvs)):
        if not ckpts:
            continue
        p = ctx.Process(
            target=worker_main,
            args=(rank, gpu_id, ckpts, args_dict, z_dims, probe_cache,
                  t_values, args.weights, csvp, proj_coeff),
            daemon=False,
        )
        p.start()
        procs.append(p)

    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(p.exitcode)
    if failed:
        print(f"[main] FAIL — worker exit codes: {failed}")
        sys.exit(1)

    # ── Stage 3: merge + plot ─────────────────────────────────────────────────
    merged_csv = out_dir / f"measurements_{args.weights}.csv"
    with open(merged_csv, "w") as out:
        first = True
        for csvp in worker_csvs:
            if not csvp.exists():
                continue
            with open(csvp) as src:
                lines = src.readlines()
                if not lines:
                    continue
                if first:
                    out.writelines(lines)
                    first = False
                else:
                    out.writelines(lines[1:])
    print(f"[main] merged → {merged_csv}")

    plot_results(merged_csv, out_dir, args.weights)
    print(f"[main] done.")


if __name__ == "__main__":
    main()
