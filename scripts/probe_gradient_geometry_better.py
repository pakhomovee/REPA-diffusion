#!/usr/bin/env python3
"""
probe_gradient_geometry_better.py — same physical measurement as
probe_gradient_geometry.py, with substantially more samples for tight
confidence intervals and a richer per-cell record that lets us compute the
"aggregate-gradient" cosine (cos of E[g]) in addition to the per-microbatch
mean cosine (E[cos(g)]).

Defaults vs original:
    n_microbatches   8 →  64     8× more samples per cell
    batch_size      16 →  64     4× larger per-microbatch gradient (less noisy)
    t_values         5 →  11     0.05, 0.10, 0.20, …, 0.90, 0.95
                                  (original 5 values are a subset)
    out_dir          reports/better_grad_geometry
    gpus             "0,1,2,3,4,5,6,7"

Scopes kept identical to the original: "full" and "blocks_0_3".

Extra CSV columns (per microbatch):
    dot_product   ⟨g_diff, g_repa⟩
    g_diff_sq     ‖g_diff‖²
    g_repa_sq     ‖g_repa‖²
These let downstream analysis compute the aggregate-gradient cosine:
    cos_agg = Σ⟨g₁,g₂⟩ / √(Σ‖g₁‖² · Σ‖g₂‖²)
which estimates cos(E[g_diff], E[g_repa]) instead of E[cos(g_diff, g_repa)].

This script does NOT produce plots — that is handled by
analyze_grad_geometry_better.py, which adds 95 % bootstrap CIs.

Usage:
    /home/seankopylov/.venv/bin/python scripts/probe_gradient_geometry_better.py \\
        --run-dir runs/celeba_sit_b2_repa_dinov2b_gpus4-7 \\
        --gpus 0,1,2,3,4,5,6,7 --weights raw

    # Reuse cache for the EMA run:
    /home/seankopylov/.venv/bin/python scripts/probe_gradient_geometry_better.py \\
        --run-dir runs/celeba_sit_b2_repa_dinov2b_gpus4-7 \\
        --gpus 0,1,2,3,4,5,6,7 --weights ema --skip-prepare
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

from models.sit import SiT_models  # noqa: E402
from dataset import CustomDataset  # noqa: E402
from utils import load_encoders  # noqa: E402


# ── Probe-data caching ────────────────────────────────────────────────────────
LATENTS_SCALE = 0.18215
LATENTS_BIAS = 0.0


def _preprocess_for_dinov2(x_uint8: torch.Tensor) -> torch.Tensor:
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
    """Build and cache the frozen probe set (same structure as the original)."""
    assert n_samples == n_microbatches * batch_size, (
        f"n_samples ({n_samples}) must equal n_microbatches*batch_size "
        f"({n_microbatches}*{batch_size}={n_microbatches * batch_size})"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Loading CustomDataset from {data_dir} …", flush=True)
    dataset = CustomDataset(str(data_dir))
    n_total = len(dataset)
    print(f"[prepare] Dataset size = {n_total:,}", flush=True)
    if n_samples > n_total:
        raise ValueError(f"Requested n_samples={n_samples} > dataset size {n_total}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)[:n_samples].tolist()

    raw_imgs, moments, labels = [], [], []
    for idx in perm:
        img, feat, lbl = dataset[idx]
        raw_imgs.append(img)
        moments.append(feat)
        labels.append(lbl)
    raw_imgs_t = torch.stack(raw_imgs).to(torch.uint8)
    moments_t = torch.stack(moments).squeeze(1).float()  # (N, 8, 32, 32)
    labels_t = torch.stack(labels).long()
    print(f"[prepare] Stacked: raw {tuple(raw_imgs_t.shape)} moments {tuple(moments_t.shape)} "
          f"labels {tuple(labels_t.shape)}", flush=True)

    g_vae = torch.Generator(device="cpu").manual_seed(seed + 1)
    eps_vae = torch.randn(moments_t.shape[0], 4, 32, 32, generator=g_vae)
    mean, std = torch.chunk(moments_t, 2, dim=1)
    latents = (mean + std * eps_vae) * LATENTS_SCALE + LATENTS_BIAS
    print(f"[prepare] VAE-sampled latents: shape {tuple(latents.shape)} "
          f"mean={latents.mean():.3f} std={latents.std():.3f}", flush=True)

    print(f"[prepare] Loading DINOv2-B and computing teacher features on {device} …", flush=True)
    encoders, _, _ = load_encoders("dinov2-vit-b", device, 256)
    assert len(encoders) == 1
    encoder = encoders[0]
    encoder.eval()
    teacher_feats_chunks = []
    with torch.no_grad():
        for i in range(0, n_samples, 32):
            batch = raw_imgs_t[i:i + 32].to(device)
            batch_pp = _preprocess_for_dinov2(batch)
            out = encoder.forward_features(batch_pp)
            assert "x_norm_patchtokens" in out, f"DINOv2 output keys {list(out.keys())}"
            feats = out["x_norm_patchtokens"].float()
            teacher_feats_chunks.append(feats.cpu())
    teacher_feats = torch.cat(teacher_feats_chunks, dim=0)
    print(f"[prepare] Teacher feats: shape {tuple(teacher_feats.shape)} "
          f"mean={teacher_feats.mean():.3f} std={teacher_feats.std():.3f}", flush=True)
    del encoder, encoders
    torch.cuda.empty_cache()

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
    print(f"[prepare] Wrote probe cache: {cache_path}", flush=True)


# ── Model loading ─────────────────────────────────────────────────────────────
def _strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def build_model(args_dict: dict, z_dims: list, device: torch.device) -> torch.nn.Module:
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
    assert len(zs_tilde) == 1, f"Expected 1 teacher, got {len(zs_tilde)}"
    z_tilde = F.normalize(zs_tilde[0], dim=-1)
    z = F.normalize(teacher_feats, dim=-1)
    per_sample = -(z * z_tilde).sum(dim=-1).mean(dim=-1)
    return per_sample.mean()


def measure_cell(
    model: torch.nn.Module,
    latents: torch.Tensor,
    eps: torch.Tensor,
    labels: torch.Tensor,
    teacher_feats: torch.Tensor,
    t_value: float,
    scope_names: dict[str, list[str]],
    proj_coeff: float,
) -> list[dict]:
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

    loss_diff.backward(retain_graph=True)
    g_diff = {scope: collect_flat_grad(model, names) for scope, names in scope_names.items()}

    model.zero_grad(set_to_none=True)
    loss_repa.backward()
    g_repa = {scope: collect_flat_grad(model, names) for scope, names in scope_names.items()}

    rows = []
    for scope in scope_names:
        gd = g_diff[scope]
        gr = g_repa[scope]
        dot = float((gd * gr).sum())
        gd_sq = float((gd * gd).sum())
        gr_sq = float((gr * gr).sum())
        gd_norm = gd_sq ** 0.5
        gr_norm = gr_sq ** 0.5
        denom = gd_norm * gr_norm
        cos = dot / denom if denom > 0 else 0.0
        ratio = gr_norm / gd_norm if gd_norm > 0 else 0.0
        rows.append({
            "scope": scope,
            "cos_sim": cos,
            "ratio": ratio,
            "g_diff_norm": gd_norm,
            "g_repa_norm": gr_norm,
            "dot_product": dot,
            "g_diff_sq": gd_sq,
            "g_repa_sq": gr_sq,
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
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    pid = os.getpid()
    print(f"[worker {rank}|gpu {gpu_id}|pid {pid}] start; ckpts={[p.stem for p in ckpt_paths]}",
          flush=True)

    cache = torch.load(str(probe_cache), map_location="cpu", weights_only=False)
    latents_cpu = cache["latents"]
    labels_cpu = cache["labels"]
    teacher_cpu = cache["teacher_feats"]
    eps_cpu = cache["eps"]
    n_mb, B = eps_cpu.shape[:2]
    print(f"[worker {rank}] cache loaded; n_mb={n_mb} B={B}", flush=True)

    model = build_model(args_dict, z_dims, device)
    scope_names = make_scope_masks(model, args_dict["encoder_depth"])
    n_full = sum(dict(model.named_parameters())[n].numel() for n in scope_names["full"])
    n_b03 = sum(dict(model.named_parameters())[n].numel() for n in scope_names["blocks_0_3"])
    print(f"[worker {rank}] scope sizes: full={n_full:,} blocks_0_3={n_b03:,}", flush=True)

    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "ckpt_step", "t", "microbatch", "weights", "scope",
            "cos_sim", "ratio", "g_diff_norm", "g_repa_norm",
            "dot_product", "g_diff_sq", "g_repa_sq",
            "loss_diff", "loss_repa_scaled",
        ])
        wr.writeheader()

        for ckpt_path in ckpt_paths:
            t0 = time.perf_counter()
            steps = load_checkpoint_into_model(model, ckpt_path, weights)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(True)
            print(f"[worker {rank}] loaded {ckpt_path.name} ({weights}) steps={steps} "
                  f"in {time.perf_counter() - t0:.1f}s", flush=True)

            for t_val in t_values:
                t_start = time.perf_counter()
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
                dt = time.perf_counter() - t_start
                print(f"[worker {rank}] ckpt={ckpt_path.stem} t={t_val:.2f} "
                      f"done in {dt:.1f}s ({dt / n_mb * 1000:.0f} ms/mb)", flush=True)

    print(f"[worker {rank}] finished; csv={out_csv}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "celeba256"))
    p.add_argument("--ckpt-steps", type=str, default="10000,50000,100000,150000,200000")
    p.add_argument("--t-values", type=str,
                   default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    p.add_argument("--n-microbatches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--weights", choices=["raw", "ema"], default="raw")
    p.add_argument("--out-dir", type=str,
                   default=str(PROJECT_ROOT / "reports" / "better_grad_geometry"))
    p.add_argument("--probe-cache", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--prepare-gpu", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_cache = Path(args.probe_cache).resolve() if args.probe_cache else out_dir / "probe_data.pt"

    with open(run_dir / "args.json") as f:
        args_dict = json.load(f)
    print(f"[main] run_dir={run_dir}", flush=True)
    print(f"[main] model={args_dict['model']} num_classes={args_dict['num_classes']} "
          f"encoder_depth={args_dict['encoder_depth']} enc_type={args_dict.get('enc_type')} "
          f"proj_coeff={args_dict.get('proj_coeff')}", flush=True)
    if args_dict.get("enc_type") != "dinov2-vit-b":
        print(f"[main] WARNING: enc_type={args_dict.get('enc_type')!r}, expected dinov2-vit-b. "
              f"Teacher feats in cache may mismatch.", flush=True)
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
    print(f"[main] ckpts={[p.name for p in ckpt_paths]}", flush=True)
    print(f"[main] t_values={t_values} ({len(t_values)} values)", flush=True)
    print(f"[main] gpus={gpus} weights={args.weights}", flush=True)
    print(f"[main] n_microbatches={args.n_microbatches} batch_size={args.batch_size} "
          f"→ {args.n_microbatches * args.batch_size} total images", flush=True)

    n_samples = args.n_microbatches * args.batch_size

    # Stage 1 — probe cache
    if args.skip_prepare and probe_cache.exists():
        print(f"[main] skip-prepare: reusing {probe_cache}", flush=True)
    else:
        prep_gpu = args.prepare_gpu if args.prepare_gpu is not None else gpus[0]
        torch.cuda.set_device(prep_gpu)
        prepare_probe_data(
            data_dir=data_dir,
            n_samples=n_samples,
            n_microbatches=args.n_microbatches,
            batch_size=args.batch_size,
            seed=args.seed,
            cache_path=probe_cache,
            device=torch.device(f"cuda:{prep_gpu}"),
        )
        if args.prepare_only:
            print("[main] prepare-only: exiting", flush=True)
            return

    # Stage 2 — measurement
    z_dims = [768]
    worker_ckpts: list[list[Path]] = [[] for _ in gpus]
    for i, p in enumerate(ckpt_paths):
        worker_ckpts[i % len(gpus)].append(p)
    worker_csvs = [out_dir / f"measurements__worker{r}_{args.weights}.csv"
                   for r in range(len(gpus))]

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
        print(f"[main] FAIL — worker exit codes: {failed}", flush=True)
        sys.exit(1)

    # Stage 3 — merge
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
    print(f"[main] merged → {merged_csv}", flush=True)
    print("[main] done.", flush=True)


if __name__ == "__main__":
    main()
