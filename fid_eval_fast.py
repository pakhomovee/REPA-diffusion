#!/usr/bin/env python3
"""
fid_eval_fast.py — One-command parallel FID/KID/LPIPS scorer.

Design:
  • N worker processes, one per GPU, each owns its InceptionV3 instance.
  • Checkpoints distributed round-robin across workers → all GPUs run in parallel.
  • _load_npz_fast: C-level read at full disk/cache speed (no Python zipfile overhead).
  • Real stats loaded once per worker from mmap'd files.
  • Results written to CSV progressively as each checkpoint finishes.
  • Optional LPIPS diversity computed at the end (also fast-loaded).

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python fid_eval_fast.py \\
      --runs "Baseline:fid_samples/Baseline" "REPA (DINOv2-B):fid_samples/REPA (DINOv2-B)" \\
      --steps 10000:200000:10000 \\
      --n-gpus 4 \\
      --batch-size 512 \\
      --num-workers 8 \\
      --diversity \\
      --out scores.csv \\
      --plot-dir fid_results
"""
import os, sys, re, csv, ast, time, argparse, zipfile, struct as _struct
from pathlib import Path
from multiprocessing import Pool

import numpy as np

# ── Fast NPZ utilities ────────────────────────────────────────────────────────
# Defined at module level so workers can import them.

def _npz_sample_count(path):
    """Read shape from numpy header only — no array data loaded."""
    with zipfile.ZipFile(path) as zf:
        with zf.open("arr_0.npy") as f:
            f.read(6)
            major = _struct.unpack("BB", f.read(2))[0]
            hlen = (_struct.unpack("<H", f.read(2))[0] if major == 1
                    else _struct.unpack("<I", f.read(4))[0])
            header = f.read(hlen).decode("latin1")
    m = re.search(r"shape.*?\((\d+)", header)
    return int(m.group(1)) if m else 0


def _load_npz_fast(path, key="arr_0"):
    """C-level read that bypasses Python zipfile chunking — full disk/cache speed."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        hdr = f.read(30)
        fname_len, extra_len = _struct.unpack_from("<HH", hdr, 26)
        f.seek(info.header_offset + 30 + fname_len + extra_len)
        f.read(6)  # numpy magic
        major = f.read(1)[0]; f.read(1)
        hlen = (_struct.unpack("<H", f.read(2))[0] if major == 1
                else _struct.unpack("<I", f.read(4))[0])
        d = ast.literal_eval(f.read(hlen).decode("latin1").strip())
        shape, dtype = tuple(d["shape"]), np.dtype(d["descr"])
        raw = f.read(int(np.prod(shape)) * dtype.itemsize)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ── Worker state (per-process globals) ────────────────────────────────────────
_state: dict = {}

def _worker_init(local_gpu_idx: int, stats_dir: str, batch_size: int, num_dl_workers: int):
    """Called once when each Pool process starts. Sets up GPU + models + stats."""
    # Restrict this process to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_idx)

    import torch
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import frechet_distance, kernel_distance
    from cleanfid.resize import build_resizer
    from torchvision import transforms

    device = torch.device("cuda:0")
    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)

    sd = Path(stats_dir)
    fid_stats = np.load(sd / "celeba256_clean_custom_na.npz")
    kid_ref   = np.load(sd / "celeba256_clean_custom_na_kid.npz", mmap_mode="r")["feats"]

    fn_resize  = build_resizer("clean")
    to_tensor  = transforms.ToTensor()

    _state.update(
        device=device,
        feat_model=feat_model,
        mu_real=fid_stats["mu"],
        sigma_real=fid_stats["sigma"],
        kid_ref=kid_ref,
        batch_size=batch_size,
        num_dl_workers=num_dl_workers,
        fn_resize=fn_resize,
        to_tensor=to_tensor,
        gpu_idx=local_gpu_idx,
        frechet_distance=frechet_distance,
        kernel_distance=kernel_distance,
    )
    print(f"  [GPU {local_gpu_idx}] worker ready", flush=True)


class _NpDataset:
    """Wraps a (N,H,W,C) uint8 array for DataLoader with cleanfid preprocessing."""
    def __init__(self, arr, fn_resize, to_tensor):
        self.arr = arr
        self.fn_resize = fn_resize
        self.to_tensor = to_tensor

    def __len__(self): return len(self.arr)

    def __getitem__(self, i):
        img = self.fn_resize(np.array(self.arr[i]))  # → (299,299,3) float32 [0,255]
        return self.to_tensor(img)                    # → (3,299,299) float32 [0,255]


def _score_checkpoint(task: tuple) -> tuple:
    """
    Worker function: score one checkpoint.
    task = (run_name, step, npz_path)
    Returns (run_name, step, fid, kid_e3, elapsed_s)
    """
    import torch
    run_name, step, npz_path = task
    s = _state
    t0 = time.perf_counter()

    # Load samples
    imgs = _load_npz_fast(npz_path)
    t_load = time.perf_counter() - t0

    # Extract InceptionV3 features
    dataset = _NpDataset(imgs, s["fn_resize"], s["to_tensor"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=s["batch_size"],
        num_workers=s["num_dl_workers"],
        shuffle=False, drop_last=False, pin_memory=True,
    )
    feats = []
    s["feat_model"].eval()
    with torch.no_grad():
        for batch in loader:
            feats.append(s["feat_model"](batch.to(s["device"])).cpu().numpy())
    gen_feats = np.concatenate(feats)
    t_feat = time.perf_counter() - t0 - t_load

    # FID + KID
    mu_gen    = np.mean(gen_feats, 0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    fid = float(s["frechet_distance"](s["mu_real"], s["sigma_real"], mu_gen, sigma_gen))
    kid = float(s["kernel_distance"](s["kid_ref"], gen_feats)) * 1e3
    elapsed = time.perf_counter() - t0

    print(
        f"  [GPU {s['gpu_idx']}] {run_name}@{step//1000}k "
        f"FID={fid:.2f} KID×10³={kid:.3f}  "
        f"(load={t_load:.1f}s feat={t_feat:.1f}s total={elapsed:.1f}s)",
        flush=True,
    )
    return run_name, step, fid, kid, elapsed


def _compute_diversity_fast(npz_path: str, label: str, n_pairs: int = 2000,
                            batch_size: int = 64) -> float:
    """LPIPS pairwise diversity, standalone (run in main process after Pool is closed)."""
    import torch, lpips as lpips_lib
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    imgs = _load_npz_fast(npz_path)
    N = len(imgs)
    rng = np.random.default_rng(42)
    ia = rng.integers(0, N, n_pairs); ib = rng.integers(0, N, n_pairs)
    same = ia == ib; ib[same] = (ib[same] + 1) % N

    scores = []
    for i in range(0, n_pairs, batch_size):
        ba = torch.stack([prep(imgs[j]) for j in ia[i:i+batch_size]]).to(device)
        bb = torch.stack([prep(imgs[j]) for j in ib[i:i+batch_size]]).to(device)
        with torch.no_grad():
            out = loss_fn(ba, bb).squeeze().cpu()
        scores.extend(out.tolist() if out.dim() > 0 else [out.item()])
    div = float(np.mean(scores))
    print(f"Diversity LPIPS ({label}): {div:.4f}  (n_pairs={n_pairs})", flush=True)
    return div


# ── Plotting ──────────────────────────────────────────────────────────────────
def _make_plots(results: dict, steps: list, out_dir: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    out_dir.mkdir(exist_ok=True)
    BLUE = "#2196F3"; ORANGE = "#FF5722"
    COLORS  = {"Baseline": BLUE,  "REPA (DINOv2-B)": ORANGE}
    MARKERS = {"Baseline": "o",   "REPA (DINOv2-B)": "s"}

    run_names = list(results.keys())
    for metric_i, (ylabel, fname) in enumerate([("FID ↓", "fid_vs_steps"),
                                                 ("KID×10³ ↓", "kid_vs_steps")]):
        for log_scale in (False, True):
            fig, ax = plt.subplots(figsize=(8, 5))
            for rn in run_names:
                xs = [s for s in steps if s in results[rn]]
                ys = [results[rn][s][metric_i] for s in xs]
                ax.plot(xs, ys, color=COLORS.get(rn, "gray"),
                        marker=MARKERS.get(rn, "o"), ms=5, lw=2, label=rn)
            ax.set_xlabel("Training Steps"); ax.set_ylabel(ylabel)
            scale_tag = " (log)" if log_scale else ""
            ax.set_title(f"{ylabel.split()[0]} vs Steps{scale_tag} — SiT-B/2 CelebA-256")
            if log_scale:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
                ax.grid(True, which="both", alpha=0.3)
            else:
                ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(steps[::2])
            ax.set_xticklabels([f"{s//1000}k" for s in steps[::2]])
            plt.tight_layout()
            suffix = "_log" if log_scale else ""
            plt.savefig(out_dir / f"{fname}{suffix}.png", dpi=150, bbox_inches="tight")
            plt.close()
    print(f"Plots saved → {out_dir}/", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Parallel FID/KID scorer")
    ap.add_argument("--runs", nargs="+", required=True,
                    help='"name:path" pairs, e.g. "Baseline:fid_samples/Baseline"')
    ap.add_argument("--steps", default="10000:200000:10000",
                    help="start:stop:step or comma list (stop is inclusive)")
    ap.add_argument("--n-gpus",      type=int, default=4)
    ap.add_argument("--batch-size",  type=int, default=512,
                    help="InceptionV3 batch size per GPU")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers per GPU process")
    ap.add_argument("--diversity",   action="store_true",
                    help="Compute LPIPS diversity for final checkpoint")
    ap.add_argument("--n-pairs",     type=int, default=2000)
    ap.add_argument("--out",         default="scores.csv")
    ap.add_argument("--plot-dir",    default="fid_results")
    ap.add_argument("--stats-dir",   default=None,
                    help="Override cleanfid stats dir (auto-detected if omitted)")
    args = ap.parse_args()

    # Parse steps
    if ":" in args.steps:
        s, e, d = [int(x) for x in args.steps.split(":")]
        steps = list(range(s, e + 1, d))
    else:
        steps = [int(x) for x in args.steps.split(",")]

    # Parse runs
    runs = {}
    for item in args.runs:
        name, path = item.split(":", 1)
        runs[name] = Path(path)

    # Discover NPZ files
    tasks = []  # (run_name, step, npz_path)
    for run_name, run_dir in runs.items():
        for step in steps:
            step_dir = run_dir / f"step_{step:07d}"
            npzs = list(step_dir.glob("*.npz")) if step_dir.exists() else []
            if not npzs:
                print(f"WARN: no NPZ for {run_name} @ step {step:,}", flush=True)
                continue
            n = _npz_sample_count(str(npzs[0]))
            print(f"  [found] {run_name}@{step//1000}k → {npzs[0].name} ({n:,} samples)",
                  flush=True)
            tasks.append((run_name, step, str(npzs[0])))

    if not tasks:
        sys.exit("No NPZ files found — check --runs paths.")

    # Locate stats dir
    if args.stats_dir:
        stats_dir = args.stats_dir
    else:
        import cleanfid as _cfd
        stats_dir = str(Path(_cfd.__file__).parent / "stats")
    print(f"\nStats dir: {stats_dir}", flush=True)
    print(f"Scoring {len(tasks)} checkpoints across {args.n_gpus} GPUs…\n", flush=True)

    # One Pool per GPU (1 worker each), tasks dispatched round-robin
    pools = []
    for i in range(args.n_gpus):
        p = Pool(
            processes=1,
            initializer=_worker_init,
            initargs=(i, stats_dir, args.batch_size, args.num_workers),
        )
        pools.append(p)

    futures = []
    for i, task in enumerate(tasks):
        pool = pools[i % args.n_gpus]
        futures.append((task[0], task[1], pool.apply_async(_score_checkpoint, (task,))))

    # Collect results and write CSV
    results: dict[str, dict[int, tuple]] = {}  # run_name → {step: (fid, kid)}
    out_path = Path(args.out)
    with open(out_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["run", "step", "fid", "kid_e3", "elapsed_s"])
        for run_name, step, fut in futures:
            rn, s, fid, kid, elapsed = fut.get()
            writer.writerow([rn, s, f"{fid:.4f}", f"{kid:.6f}", f"{elapsed:.1f}"])
            csvf.flush()
            results.setdefault(rn, {})[s] = (fid, kid)

    for p in pools:
        p.close(); p.join()

    print(f"\n✓ All checkpoints scored. CSV → {out_path}", flush=True)

    # Summary table
    for rn in results:
        best_s = min(results[rn], key=lambda s: results[rn][s][0])
        best_fid, best_kid = results[rn][best_s]
        print(f"  {rn}: best FID={best_fid:.2f} @ {best_s//1000}k  "
              f"KID×10³={best_kid:.3f}", flush=True)

    # Diversity
    if args.diversity:
        print("\n── LPIPS Diversity ──────────────────────────────────────────", flush=True)
        for run_name, run_dir in runs.items():
            last_step = max(steps)
            npzs = list((run_dir / f"step_{last_step:07d}").glob("*.npz"))
            if not npzs:
                print(f"  WARN: no NPZ for {run_name}@{last_step//1000}k, skipping")
                continue
            _compute_diversity_fast(str(npzs[0]), label=run_name, n_pairs=args.n_pairs)

    # Plots
    print("\n── Plots ────────────────────────────────────────────────────────", flush=True)
    _make_plots(results, steps, Path(args.plot_dir))

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)
    main()
