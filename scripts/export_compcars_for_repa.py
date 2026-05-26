#!/usr/bin/env python3
"""Export CompCars images from Kaggle to REPA-compatible image folders.

Kaggle dataset: https://www.kaggle.com/datasets/renancostaalencar/compcars

The output structure expected by REPA preprocessing (dataset_tools.py) is:
    <output_dir>/
        <class_dir>/          e.g. 000-honda_civic/
            000000.jpg
            000001.jpg
            ...
        dataset.json          <- {"labels": [["<class_dir>/000000.jpg", <class_id>], ...]}

Classes are defined at the make × model level (same granularity as the
Stanford Cars split in this repo).  Pass the printed --num-classes value
to the training scripts.

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
import re
import sys
from pathlib import Path

from torchvision import transforms
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from compcars import CompCarsDataset  # noqa: E402


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "class"


def export_compcars(
    root_dir: Path,
    output_dir: Path,
    resolution: int,
    min_class_size: int,
) -> None:
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])

    dataset = CompCarsDataset(root_dir=str(root_dir), transform=transform)

    # Optional: drop tiny classes (reduces noise from rare models)
    if min_class_size > 1:
        from collections import Counter
        counts = Counter(class_id for _, class_id in dataset.samples)
        kept_ids = {cid for cid, cnt in counts.items() if cnt >= min_class_size}
        # Remap to contiguous ids
        old_to_new = {old: new for new, old in enumerate(sorted(kept_ids))}
        dataset.samples = [
            (p, old_to_new[cid]) for p, cid in dataset.samples if cid in kept_ids
        ]
        dataset.classes = [dataset.classes[old] for old in sorted(kept_ids)]
        print(
            f"Kept {len(dataset.classes)} classes with >= {min_class_size} images "
            f"({len(dataset)} samples total)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create class directories
    for class_id, class_name in enumerate(dataset.classes):
        (output_dir / f"{class_id:03d}-{slugify(class_name)}").mkdir(
            parents=True, exist_ok=True
        )

    labels: list[list] = []
    class_counters: dict[int, int] = {}

    for img_path, class_id in tqdm(dataset.samples, desc="Exporting CompCars"):
        class_name = dataset.classes[class_id]
        class_dir = f"{class_id:03d}-{slugify(class_name)}"
        img_idx = class_counters.get(class_id, 0)
        class_counters[class_id] = img_idx + 1
        rel_path = f"{class_dir}/{img_idx:06d}.jpg"
        out_path = output_dir / rel_path
        if not out_path.exists():
            img = dataset.transform(
                __import__("PIL").Image.open(img_path).convert("RGB")
            ) if dataset.transform else __import__("PIL").Image.open(img_path).convert("RGB")
            img.save(out_path, quality=95)
        labels.append([rel_path, class_id])

    dataset_json_path = output_dir / "dataset.json"
    with dataset_json_path.open("w") as f:
        json.dump({"labels": labels}, f)

    classes_json_path = output_dir / "classes.json"
    with classes_json_path.open("w") as f:
        json.dump(
            {str(i): name for i, name in enumerate(dataset.classes)}, f, indent=2
        )

    num_classes = len(dataset.classes)
    print(f"Exported {len(labels)} images to {output_dir}")
    print(f"Number of classes: {num_classes}  (make × model)")
    print(f"dataset.json   -> {dataset_json_path}")
    print(f"classes.json   -> {classes_json_path}")
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
    print(f"Then train with  --data-dir={output_dir}  --num-classes={num_classes}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export CompCars to REPA image-folder format."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "compcars",
        help="Root directory containing (or receiving) the Kaggle dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "compcars256",
        help="Destination directory for the exported REPA-compatible dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Output image resolution (default: 256).",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Drop classes with fewer than N images (default: 10). "
            "Set to 1 to keep all classes."
        ),
    )
    args = parser.parse_args()
    export_compcars(args.root_dir, args.output_dir, args.resolution, args.min_class_size)


if __name__ == "__main__":
    main()
