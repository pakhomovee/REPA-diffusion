"""
data/latent_cache.py

Two responsibilities:
  1. PRE-ENCODING SCRIPT — encodes an entire dataset through a frozen VAE and
     saves each sample's latent + class label to disk as individual .pt files.
     Run once before training; never repeated unless the dataset or VAE changes.

  2. CACHE LOADER — a torch.utils.data.Dataset that reads the pre-cached .pt
     files at training time, bypassing the VAE entirely. This is ~10x faster
     than encoding on-the-fly every epoch.

CLI usage (pre-encoding)::

    python data/latent_cache.py \\
        --dataset    imagenet \\
        --root       /data/imagenet \\
        --fraction   0.05 \\
        --resolution 256 \\
        --out_dir    /cache/imagenet_f05_r256 \\
        --batch_size 64 \\
        --num_workers 8 \\
        --seed       42

    python data/latent_cache.py \\
        --dataset    stanford_cars \\
        --root       /data \\
        --resolution 256 \\
        --out_dir    /cache/stanford_cars_r256 \\
        --batch_size 32 \\
        --num_workers 4

Cache layout on disk::

    out_dir/
      train/
        000000000.pt   # {"latent": Tensor[C, H_lat, W_lat], "label": int}
        000000001.pt
        ...
      val/             # (ImageNet only — no subsampling applied)
        000000000.pt
        ...

Each .pt file is a dict: {"latent": torch.FloatTensor[C, H//8, W//8], "label": int}
The SD-VAE downsamples by 8x, so a 256x256 image -> 4x32x32 latent.
"""

import argparse
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ------------------------------------------------------------------------------
# Cache writer  (pre-encoding script logic)
# ------------------------------------------------------------------------------

def encode_and_cache(
    dataset_name: Literal["imagenet", "stanford_cars"],
    root: str,
    out_dir: str,
    split: str = "train",
    fraction: float = 1.0,
    resolution: int = 256,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
    device: Optional[str] = None,
    resume: bool = True,
) -> None:
    """
    Encode every image in the dataset through a frozen VAE and write the
    resulting latents to disk as individual .pt files.

    Safe to interrupt and re-run: when ``resume=True`` (default), any .pt file
    that already exists is skipped, so only the remaining samples are encoded.

    Args:
        dataset_name: ``"imagenet"`` or ``"stanford_cars"``.
        root:         Path to the raw image dataset root.
        out_dir:      Directory under which a ``<split>/`` sub-folder is created
                      to hold the .pt cache files.
        split:        ``"train"``, ``"val"``, or ``"test"``.
        fraction:     Sub-sampling fraction applied to the ImageNet train split
                      (e.g. 0.05 keeps 5%). Ignored for Stanford Cars and for
                      the val/test splits.
        resolution:   Pixel resolution fed to the VAE encoder.
        batch_size:   Number of images encoded per forward pass — tune to fill
                      available GPU VRAM.
        num_workers:  DataLoader worker processes for image loading.
        seed:         Fixed random seed used by ``deterministic_subset`` in
                      ``dataset.py`` — must match the seed used at training time
                      so the subset is identical.
        device:       Torch device string (e.g. ``"cuda"``, ``"cuda:1"``,
                      ``"cpu"``). ``None`` -> auto-detect.
        resume:       When True, skip files that already exist on disk.
                      Set False to force a full re-encode.
    """
    # Lazy imports — keeps the module importable without heavy deps at import time
    from data.dataset import get_dataset
    from models.autoencoder import get_vae

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ------------------------------------------------------------------
    # Dataset & loader
    # shuffle=False is critical: we assign file indices sequentially,
    # so the ordering must be stable across runs for resume to work.
    # ------------------------------------------------------------------
    dataset = get_dataset(
        dataset_name=dataset_name,
        root=root,
        split=split,
        fraction=fraction,
        resolution=resolution,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Frozen VAE
    # ------------------------------------------------------------------
    vae = get_vae(device=device)
    vae.eval()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    cache_dir = Path(out_dir) / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    print(f"[latent_cache] Encoding {total} images -> {cache_dir}")
    print(
        f"  dataset={dataset_name}  split={split}  fraction={fraction}  "
        f"resolution={resolution}  device={device}"
    )

    global_idx = 0
    skipped = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Encoding [{split}]", unit="batch"):
            images = images.to(device, non_blocking=True)   # (B, 3, H, W)

            # VAE encode: (B, 3, H, W) -> (B, C_lat, H//8, W//8)
            latents = vae.encode(images).cpu()

            for i in range(latents.shape[0]):
                out_path = cache_dir / f"{global_idx:09d}.pt"

                if resume and out_path.exists():
                    global_idx += 1
                    skipped += 1
                    continue

                label = (
                    labels[i].item()
                    if isinstance(labels[i], torch.Tensor)
                    else int(labels[i])
                )
                torch.save({"latent": latents[i], "label": label}, out_path)
                global_idx += 1

    written = global_idx - skipped
    print(
        f"[latent_cache] Done.  "
        f"Written: {written}  Skipped (already existed): {skipped}  "
        f"Total: {global_idx}"
    )


# ------------------------------------------------------------------------------
# LatentCacheDataset  (fast training-time loader)
# ------------------------------------------------------------------------------

class LatentCacheDataset(Dataset):
    """
    Reads pre-cached latent tensors written by :func:`encode_and_cache`.

    Each ``.pt`` file on disk contains::

        {"latent": torch.FloatTensor[C, H_lat, W_lat], "label": int}

    This dataset returns ``(latent, label)`` tuples directly, with zero VAE
    overhead — making it the correct data source for the training loop in
    ``train.py`` (step 1: "Sample batch z, label from latent cache").

    Args:
        cache_dir: Path to the *split-level* cache directory, i.e. the folder
                   that contains the numbered ``.pt`` files directly
                   (e.g. ``/cache/imagenet_f05_r256/train``).

    Raises:
        FileNotFoundError: If ``cache_dir`` does not exist.
        RuntimeError:      If no ``.pt`` files are found inside ``cache_dir``.

    Example::

        dataset = LatentCacheDataset("/cache/imagenet_f05_r256/train")
        latent, label = dataset[0]
        # latent: FloatTensor[4, 32, 32]  (SD-VAE output for 256x256 input)
        # label:  int in [0, 999]
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}\n"
                "Run `python data/latent_cache.py --dataset ... --out_dir ...` "
                "to generate the cache first."
            )

        # Sort for deterministic, reproducible ordering across runs
        self.files = sorted(self.cache_dir.glob("*.pt"))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .pt files found in {self.cache_dir}. "
                "The cache directory exists but appears to be empty. "
                "Check that encoding completed successfully."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            latent: ``torch.FloatTensor`` of shape ``[C, H_lat, W_lat]``
            label:  class index as Python ``int``
        """
        sample = torch.load(self.files[idx], weights_only=True)
        latent: torch.Tensor = sample["latent"]   # FloatTensor [C, H_lat, W_lat]
        label: int = int(sample["label"])
        return latent, label

    def latent_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of a single latent tensor ``(C, H_lat, W_lat)``
        by peeking at the first file. Useful for model initialisation.
        """
        sample = torch.load(self.files[0], weights_only=True)
        return tuple(sample["latent"].shape)

    def __repr__(self) -> str:
        return (
            f"LatentCacheDataset("
            f"cache_dir={self.cache_dir}, "
            f"n_samples={len(self.files)})"
        )


def get_latent_cache_dataloader(
    cache_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Return a :class:`torch.utils.data.DataLoader` backed by
    :class:`LatentCacheDataset`.

    This is the primary entry point consumed by ``train.py``::

        loader = get_latent_cache_dataloader(cfg.cache_dir, cfg.batch_size)
        for latents, labels in loader:
            ...   # latents: (B, C, H_lat, W_lat), labels: (B,)

    Args:
        cache_dir:   Path to the split-level cache directory (contains .pt files).
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        pin_memory:  Pin tensors in page-locked memory for faster GPU transfer.
        shuffle:     Shuffle each epoch (True for training, False for evaluation).

    Returns:
        A configured ``DataLoader`` yielding
        ``(latent_batch [B, C, H_lat, W_lat], label_batch [B])`` pairs.
    """
    dataset = LatentCacheDataset(cache_dir=cache_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # drop last incomplete batch for stable training steps
    )


# ------------------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-encode a dataset through a frozen SD-VAE and save latents to "
            "disk as .pt files. Run once before training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imagenet", "stanford_cars"],
        help="Dataset to encode.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the raw image dataset root directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output root directory. A <split>/ sub-folder is created inside.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        metavar="SPLIT",
        help=(
            "Which splits to encode. Pass multiple values to encode several "
            "splits in one call. Example: --splits train val"
        ),
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help=(
            "Sub-sampling fraction for the ImageNet train split "
            "(e.g. 0.05 = 5%%). Must match the fraction used at training time."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Pixel resolution (H = W) fed into the VAE encoder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size — tune to available GPU VRAM.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes for image loading.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for deterministic sub-sampling. "
            "Must match the seed used at training time."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Torch device override (e.g. 'cuda', 'cuda:1', 'cpu'). "
            "Defaults to 'cuda' if available, else 'cpu'."
        ),
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help=(
            "Disable resume mode: re-encode all samples even if .pt files "
            "already exist. By default existing files are skipped."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    for split in args.splits:
        encode_and_cache(
            dataset_name=args.dataset,
            root=args.root,
            out_dir=args.out_dir,
            split=split,
            fraction=args.fraction,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            resume=not args.no_resume,
        )