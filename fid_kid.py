#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from cleanfid import fid as cleanfid
from pathlib import Path
import lpips
from torchvision import transforms

# ── 1. GENERATION ─────────────────────────────────────────────────────────────
def generate_samples(
    ckpt_path: str,
    output_dir: str,
    model: str = "SiT-B/2",
    num_classes: int = 16,
    num_fid_samples: int = 1000,
    per_proc_batch_size: int = 64,
    encoder_depth: int = 4,
    projector_embed_dims: int = 768,
    path_type: str = "linear",
    mode: str = "sde",
    num_steps: int = 250,
    cfg_scale: float = 1.8,
    guidance_high: float = 0.7,
    repa_root: str = "/content/REPA-diffusion/REPA",
    device_count: int = 1,
) -> str:
    """
    Run generate.py for a single checkpoint and return the path to the
    produced .npz file.

    generate.py always writes to {repa_root}/samples/<name>.npz.
    We clean that folder before each run to avoid picking up a stale
    file from a previous run, then move the result to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = Path(repa_root) / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Remove any leftover .npz so we can't accidentally grab a stale one
    for old in samples_dir.glob("*.npz"):
        old.unlink()
        print(f"Removed stale: {old.name}")

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={device_count}",
        f"{repa_root}/generate.py",
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
    ]

    print(f"\n{'='*60}")
    print(f"Generating: {Path(ckpt_path).name}")
    print(f"{'='*60}")
    print("CMD:", " ".join(cmd))

    subprocess.run(cmd, cwd=repa_root, check=True)

    npz_files = sorted(samples_dir.glob("*.npz"), key=os.path.getmtime, reverse=True)
    assert npz_files, f"No .npz found in {samples_dir} after generation."

    npz_src = npz_files[0]
    npz_dst = Path(output_dir) / npz_src.name
    npz_src.rename(npz_dst)
    print(f"Saved: {npz_dst}")
    return str(npz_dst)

def generate_all_checkpoints(
    checkpoints: dict[str, list[str]],
    samples_root: str = "/content/REPA-diffusion/samples",
    **generate_kwargs,
) -> dict[str, list[str]]:
    npz_paths: dict[str, list[str]] = {}

    for run_name, ckpt_list in checkpoints.items():
        npz_paths[run_name] = []

        for ckpt_path in ckpt_list:
            step = Path(ckpt_path).stem
            out_dir = os.path.join(samples_root, run_name, f"step_{step}")

            # Check if an .npz already exists in this output dir
            existing = list(Path(out_dir).glob("*.npz")) if Path(out_dir).exists() else []
            if existing:
                print(f"Skipping {run_name} @ {step} — found existing: {existing[0].name}")
                npz_paths[run_name].append(str(existing[0]))
                continue

            npz = generate_samples(ckpt_path, out_dir, **generate_kwargs)
            npz_paths[run_name].append(npz)

    return npz_paths

def compute_fid_kid(
    npz_paths: dict[str, list[str]],
    real_data_dir: str,
    checkpoint_steps: list[int],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:

    def npz_to_png_dir(npz_path: str) -> str:
        """Extract .npz images to a sibling /images/ folder, skip if already done."""
        img_dir = str(Path(npz_path).parent / "images")
        if Path(img_dir).exists() and any(Path(img_dir).glob("*.png")):
            print(f"  Reusing existing images: {img_dir}")
            return img_dir
        os.makedirs(img_dir, exist_ok=True)
        samples = np.load(npz_path)["arr_0"]  # (N, H, W, C) uint8
        print(f"  Extracting {len(samples)} images to {img_dir} ...")
        for i, img in enumerate(samples):
            Image.fromarray(img).save(os.path.join(img_dir, f"{i:06d}.png"))
        return img_dir

    fid_scores: dict[str, list[float]] = {}
    kid_scores: dict[str, list[float]] = {}

    for run_name, paths in npz_paths.items():
        fid_scores[run_name] = []
        kid_scores[run_name] = []

        for step, npz_path in zip(checkpoint_steps, paths):
            print(f"\nScoring {run_name} @ step {step} ...")
            img_dir = npz_to_png_dir(npz_path)

            fid_val = cleanfid.compute_fid(
                fdir1=img_dir,
                fdir2=real_data_dir,
                mode="clean",
            )
            kid_val = cleanfid.compute_kid(
                fdir1=img_dir,
                fdir2=real_data_dir,
                mode="clean",
            )

            fid_scores[run_name].append(fid_val)
            kid_scores[run_name].append(kid_val)
            print(f"  FID: {fid_val:.2f}  KID: {kid_val:.5f}")

    return fid_scores, kid_scores

def plot_fid(
    checkpoint_steps: list[int],
    fid_scores: dict[str, list[float]],
    output_path: str = "fid_plot.png",
    title: str = "FID vs Training Steps",
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, scores in fid_scores.items():
        ax.plot(checkpoint_steps, scores, marker="o", label=run_name)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("FID ↓")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"FID plot saved: {output_path}")


def plot_kid(
    checkpoint_steps: list[int],
    kid_scores: dict[str, list[float]],
    output_path: str = "kid_plot.png",
    title: str = "KID vs Training Steps",
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, scores in kid_scores.items():
        scaled = [s * 1e3 for s in scores]
        ax.plot(checkpoint_steps, scaled, marker="o", label=run_name)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("KID × 10³ ↓")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"KID plot saved: {output_path}")

def compute_lpips_from_npz(
    npz_a: str,
    npz_b: str,
    label_a: str = "baseline",
    label_b: str = "repa",
    batch_size: int = 32,
    image_size: int = 256,
    net: str = "alex",
    device: str | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    """
    Compute LPIPS between two sets of generated images stored as .npz files.
    Since you have no real images, this compares baseline-generated vs
    REPA-generated to quantify perceptual diversity / difference between
    the two models.

    Both .npz files must contain an 'arr_0' key with shape (N, H, W, C),
    uint8, values in [0, 255] — the format produce.py writes.

    Args:
        npz_a:       Path to first .npz  (e.g. baseline final checkpoint).
        npz_b:       Path to second .npz (e.g. REPA final checkpoint).
        label_a/b:   Names used in the printed summary.
        batch_size:  Pairs per forward pass (reduce if OOM).
        image_size:  Resize to this before scoring.
        net:         LPIPS backbone: "alex" | "vgg" | "squeeze".
        device:      "cuda" / "cpu" / None (auto).
        max_samples: Cap number of pairs (both npz are truncated to this).

    Returns:
        {"mean": float, "std": float, "n": int}

    Example:
        result = compute_lpips_from_npz(
            npz_a="samples/baseline/step_0040000/samples.npz",
            npz_b="samples/repa/step_0040000/samples.npz",
        )
        print(f"LPIPS: {result['mean']:.4f} ± {result['std']:.4f}")
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    images_a = np.load(npz_a)["arr_0"]   # (N, H, W, C) uint8
    images_b = np.load(npz_b)["arr_0"]

    n = min(len(images_a), len(images_b))
    if max_samples is not None:
        n = min(n, max_samples)
    images_a = images_a[:n]
    images_b = images_b[:n]
    print(f"Scoring LPIPS: {label_a} vs {label_b}  ({n} pairs)")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    scores = []
    for i in range(0, n, batch_size):
        batch_a = torch.stack([preprocess(img) for img in images_a[i:i+batch_size]]).to(device)
        batch_b = torch.stack([preprocess(img) for img in images_b[i:i+batch_size]]).to(device)

        with torch.no_grad():
            batch_scores = loss_fn(batch_a, batch_b).squeeze().cpu()
        scores.extend(batch_scores.tolist() if batch_scores.dim() > 0 else [batch_scores.item()])
        print(f"  {i + len(batch_a)}/{n} pairs scored...")

    scores = np.array(scores)
    result = {"mean": float(scores.mean()), "std": float(scores.std()), "n": int(n)}
    print(f"\nLPIPS ({net})  {label_a} vs {label_b}: "
          f"{result['mean']:.4f} ± {result['std']:.4f}  (n={result['n']})")
    return result

if __name__ == "__main__":
    ROOT       = Path("/content/REPA-diffusion")
    STEPS      = [10000, 20000, 30000, 40000]
    STEP_STRS  = [f"{s:07d}" for s in STEPS]

    # ── point at your checkpoint dirs ────────────────────────────────────────
    checkpoints = {
        "Baseline": [
            ROOT / "baseline/celeba_sit_b2_baseline_gpus2-7_40k/checkpoints" / f"{s}.pt"
            for s in STEP_STRS
        ],
        "REPA (DINOv2-B)": [
            ROOT / "repa/celeba_sit_b2_repa_dinov2b_gpus4-7/checkpoints" / f"{s}.pt"
            for s in STEP_STRS
        ],
    }
    checkpoints = {k: [str(p) for p in v] for k, v in checkpoints.items()}

    npz_paths = generate_all_checkpoints(
        checkpoints=checkpoints,
        samples_root=str(ROOT / "samples"),
        num_fid_samples=100,
        per_proc_batch_size=64,
    )

    fid_scores, kid_scores = compute_fid_kid(
        npz_paths=npz_paths,
        real_data_dir=str(ROOT / "data/celeba256/images"),
        checkpoint_steps=STEPS,
    )
    plot_fid(STEPS, fid_scores, output_path=str(ROOT / "fid_plot.png"))
    plot_kid(STEPS, kid_scores, output_path=str(ROOT / "kid_plot.png"))


    compute_lpips_from_npz(
        npz_a=npz_paths["Baseline"][-1],
        npz_b=npz_paths["REPA (DINOv2-B)"][-1],
        label_a="Baseline",
        label_b="REPA (DINOv2-B)",
    )
