#!/usr/bin/env python3
"""Export LSUN Church images from Kaggle .npy to REPA-compatible image folders.

Kaggle dataset: https://www.kaggle.com/datasets/ajaykgp12/lsunchurch

The Kaggle archive ships as a single NumPy file:
    church_outdoor_train_lmdb_color_64.npy  —  shape (126227, 64, 64, 3), uint8

The output structure expected by REPA preprocessing (dataset_tools.py) is:
    <output_dir>/
        images/
            000000.jpg
            000001.jpg
            ...
        dataset.json   <- {"labels": [["000000.jpg", 0], ...]}

Images are upsampled from 64×64 to --resolution (default 256) with Lanczos
filtering.  Because LSUN Church has no semantic classes the class id is
always 0 — pass --num-classes=1 to the training scripts.

After running this script, encode VAE latents with:
    cd REPA/preprocessing
    python dataset_tools.py encode \\
        --source <output_dir> \\
        --dest <output_dir>/vae-sd \\
        --model-url stabilityai/sd-vae-ft-mse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from torchvision import transforms
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lsunchurch import LSUNChurchDataset  # noqa: E402


def export_lsunchurch(
    root_dir: Path,
    output_dir: Path,
    resolution: int,
    max_images: int | None,
) -> None:
    # Upsample 64×64 → resolution×resolution with Lanczos
    transform = transforms.Resize(
        (resolution, resolution),
        interpolation=transforms.InterpolationMode.LANCZOS,
    )

    dataset = LSUNChurchDataset(root_dir=str(root_dir), transform=transform)
    total = len(dataset) if max_images is None else min(len(dataset), max_images)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels: list[list] = []
    for idx in tqdm(range(total), desc="Exporting LSUN Church"):
        img, info_dict = dataset[idx]
        rel_path = f"{idx:06d}.jpg"
        out_path = images_dir / rel_path
        if not out_path.exists():
            img.save(out_path, quality=95)
        labels.append([rel_path, info_dict["class_id"]])

    dataset_json_path = output_dir / "dataset.json"
    with dataset_json_path.open("w") as f:
        json.dump({"labels": labels}, f)

    print(f"\nExported {len(labels)} images ({resolution}×{resolution}) to {output_dir}")
    print("Number of classes: 1  (unconditional — LSUN Church has no labels)")
    print(f"dataset.json written to {dataset_json_path}")
    print()
    print("Next step – encode VAE latents:")
    print(
        f"  cd REPA/preprocessing\n"
        f"  python dataset_tools.py encode \\\n"
        f"      --source {output_dir} \\\n"
        f"      --dest {output_dir}/vae-sd \\\n"
        f"      --model-url stabilityai/sd-vae-ft-mse"
    )
    print()
    print(f"Then train with  --data-dir={output_dir}  --num-classes=1")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export LSUN Church (.npy) to REPA image-folder format."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "lsun_church",
        help=(
            "Directory containing church_outdoor_train_lmdb_color_64.npy "
            "(will attempt auto-download via kagglehub if absent)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "lsun_church256",
        help="Destination directory for the exported REPA-compatible dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "Output image resolution (default: 256). "
            "Source images are 64×64 and will be upsampled with Lanczos."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Cap export to the first N images (useful for quick smoke-tests).",
    )
    args = parser.parse_args()
    export_lsunchurch(args.root_dir, args.output_dir, args.resolution, args.max_images)


if __name__ == "__main__":
    main()
