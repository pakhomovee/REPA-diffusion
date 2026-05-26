#!/usr/bin/env python3
"""Export CompCars images from Kaggle to REPA-compatible image folders.

Kaggle dataset: https://www.kaggle.com/datasets/renancostaalencar/compcars

Expected on-disk layout of the downloaded archive:
    <root_dir>/
        image/
            <make_id>/
                <model_id>/
                    <year>/
                        *.jpg
        label/
        misc/
        train_test_split/

The script locates the 'image/' directory automatically regardless of where
exactly the archive was extracted.

Output structure (REPA image-folder format):
    <output_dir>/
        000-make1_model10/
            000000.jpg
            ...
        001-make1_model11/
            ...
        dataset.json     <- {"labels": [["000-.../000000.jpg", class_id], ...]}
        classes.json     <- {class_id: "make{X}_model{Y}", ...}

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
from collections import Counter
from pathlib import Path

from PIL import Image
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
    resize = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])

    # Load dataset WITHOUT transform so we can filter first, then resize on export
    dataset = CompCarsDataset(root_dir=str(root_dir), transform=None)
    print(f"Found {len(dataset)} images across {dataset.num_classes} classes before filtering.")

    # Filter classes below min_class_size
    if min_class_size > 1:
        counts = Counter(class_id for _, class_id in dataset.samples)
        kept_ids = sorted(cid for cid, cnt in counts.items() if cnt >= min_class_size)
        old_to_new = {old: new for new, old in enumerate(kept_ids)}
        dataset.samples = [
            (p, old_to_new[cid]) for p, cid in dataset.samples if cid in old_to_new
        ]
        dataset.classes = [dataset.classes[old] for old in kept_ids]
        print(
            f"After --min-class-size={min_class_size}: "
            f"{len(dataset.classes)} classes, {len(dataset.samples)} images."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create class directories
    class_dirs = {}
    for class_id, class_name in enumerate(dataset.classes):
        cdir = output_dir / f"{class_id:03d}-{slugify(class_name)}"
        cdir.mkdir(parents=True, exist_ok=True)
        class_dirs[class_id] = cdir

    labels: list[list] = []
    class_counters: dict[int, int] = {}

    for img_path, class_id in tqdm(dataset.samples, desc="Exporting CompCars"):
        img_idx = class_counters.get(class_id, 0)
        class_counters[class_id] = img_idx + 1

        class_dir = class_dirs[class_id]
        rel_path = f"{class_dir.name}/{img_idx:06d}.jpg"
        out_path = output_dir / rel_path

        if not out_path.exists():
            img = Image.open(img_path).convert("RGB")
            img = resize(img)
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
    print(f"\nExported {len(labels)} images → {output_dir}")
    print(f"Number of classes: {num_classes}  (make × model)")
    print(f"dataset.json  → {dataset_json_path}")
    print(f"classes.json  → {classes_json_path}")
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
        help=(
            "Root directory of the CompCars archive "
            "(parent of the 'image/' folder). "
            "Will attempt kagglehub auto-download if absent."
        ),
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
