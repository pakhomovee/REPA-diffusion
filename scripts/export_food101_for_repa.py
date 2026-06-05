#!/usr/bin/env python3
"""Export Food-101 images to REPA-compatible image-folder format.

Food-101 has 101 food categories:
    - 75,750 training images  (750 per class)
    - 25,250 test images      (250 per class)
Images were rescaled to a max side of 512 px by the dataset authors.

This script downloads Food-101 automatically via torchvision (no Kaggle
account needed) and exports centre-cropped, resized JPEGs into a folder
structure REPA can read directly.

Output structure:
    <output_dir>/
        000-apple_pie/
            000000.jpg ...
        001-baby_back_ribs/
            ...
        dataset.json      <- {"labels": [["000-apple_pie/000000.jpg", 0], ...]}
        classes.json      <- {"0": "apple_pie", "1": "baby_back_ribs", ...}

Usage:
    # Train split (default)
    python scripts/export_food101_for_repa.py

    # Test split
    python scripts/export_food101_for_repa.py --split test --output-dir data/food101_256_test

    # Both splits together (for unconditional / larger training pool)
    python scripts/export_food101_for_repa.py --split both
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from food101 import Food101Dataset  # noqa: E402


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "class"


def export_split(
    root_dir: Path,
    output_dir: Path,
    split: str,
    resolution: int,
    label_offset: int = 0,
    existing_labels: list | None = None,
    existing_class_counters: dict | None = None,
    max_images_per_class: int | None = None,
) -> tuple[list, dict]:
    """Export one split; returns (labels_list, class_counters)."""

    resize = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])

    dataset = Food101Dataset(root_dir=str(root_dir), split=split, download=True)
    print(f"\n[{split}] {len(dataset)} images, {dataset.num_classes} classes")

    # Pre-select indices to export (cap per class before iterating)
    if max_images_per_class is not None:
        class_counts: dict[int, int] = dict(existing_class_counters) if existing_class_counters else {}
        selected_indices = []
        for idx in range(len(dataset)):
            _, info = dataset[idx]
            cid = info["class_id"]
            if class_counts.get(cid, 0) < max_images_per_class:
                selected_indices.append(idx)
                class_counts[cid] = class_counts.get(cid, 0) + 1
        print(f"  → capped to {len(selected_indices)} images "
              f"({max_images_per_class} per class)")
    else:
        selected_indices = list(range(len(dataset)))

    # REPA expects data_dir/images/ and data_dir/vae-sd/
    images_dir = output_dir / "images"
    class_dirs: dict[int, Path] = {}
    for class_id, class_name in enumerate(dataset.classes):
        cdir = images_dir / f"{class_id:03d}-{slugify(class_name)}"
        cdir.mkdir(parents=True, exist_ok=True)
        class_dirs[class_id] = cdir

    labels = list(existing_labels) if existing_labels else []
    class_counters = dict(existing_class_counters) if existing_class_counters else {}

    for idx in tqdm(selected_indices, desc=f"Exporting {split}"):
        img_pil, info = dataset[idx]
        class_id = info["class_id"]
        img_idx  = class_counters.get(class_id, 0)
        class_counters[class_id] = img_idx + 1

        rel_path = f"{class_dirs[class_id].name}/{img_idx:06d}.jpg"
        out_path = images_dir / rel_path

        if not out_path.exists():
            img_resized = resize(img_pil)
            img_resized.save(out_path, quality=95)

        labels.append([rel_path, class_id])

    return labels, class_counters


def export(
    root_dir: Path,
    output_dir: Path,
    split: str,
    resolution: int,
    max_images_per_class: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed the dataset once to get class list
    seed_ds = Food101Dataset(root_dir=str(root_dir), split="train", download=True)
    class_names = seed_ds.classes
    num_classes  = seed_ds.num_classes

    labels: list = []
    counters: dict = {}

    splits_to_run = ["train", "test"] if split == "both" else [split]
    for sp in splits_to_run:
        labels, counters = export_split(
            root_dir, output_dir, sp, resolution,
            existing_labels=labels,
            existing_class_counters=counters,
            max_images_per_class=max_images_per_class,
        )

    # Write dataset.json into images/ so dataset_tools.py encode can read labels
    images_dir = output_dir / "images"
    dataset_json = images_dir / "dataset.json"
    with dataset_json.open("w") as f:
        json.dump({"labels": labels}, f)

    # Write classes.json into output_dir root for reference
    classes_json = output_dir / "classes.json"
    with classes_json.open("w") as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2)

    print(f"\nExported {len(labels)} images → {images_dir}")
    print(f"Number of classes : {num_classes}")
    print(f"dataset.json      → {dataset_json}")
    print(f"classes.json      → {classes_json}")
    print()
    print("Next step – encode VAE latents:")
    print(
        f"  cd REPA/preprocessing\n"
        f"  python dataset_tools.py encode \\\n"
        f"      --source {images_dir} \\\n"
        f"      --dest {output_dir}/vae-sd \\\n"
        f"      --model-url stabilityai/sd-vae-ft-mse"
    )
    print()
    print(f"Then train with  --data-dir={output_dir}  --num-classes={num_classes}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export Food-101 to REPA image-folder format."
    )
    parser.add_argument(
        "--root-dir", type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "food101",
        help="Directory where Food-101 is stored or will be downloaded.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "food101_256",
        help="Destination directory for the exported REPA-compatible dataset.",
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Output image resolution (default: 256).",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="train",
        help=(
            "Which split to export: train (75,750), test (25,250), "
            "or both (101,000). Default: train."
        ),
    )
    parser.add_argument(
        "--max-images-per-class", type=int, default=None, metavar="N",
        help=(
            "Cap the number of images exported per class. "
            "Useful for smoke tests (e.g. --max-images-per-class 5). "
            "Default: no limit."
        ),
    )
    args = parser.parse_args()
    export(args.root_dir, args.output_dir, args.split, args.resolution,
           max_images_per_class=args.max_images_per_class)


if __name__ == "__main__":
    main()
