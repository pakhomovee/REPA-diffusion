#!/usr/bin/env python3
"""Export Stanford Cars from Hugging Face datasets cache to REPA image folders."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "class"


def export_split(dataset_name: str, split: str, cache_dir: Path, output_dir: Path) -> list[str]:
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))
    class_names = list(ds.features["label"].names)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in enumerate(class_names):
        (split_dir / f"{class_id:03d}-{slugify(class_name)}").mkdir(parents=True, exist_ok=True)

    labels = []
    for idx, sample in enumerate(tqdm(ds, desc=f"export {split}")):
        label = int(sample["label"])
        class_dir = f"{label:03d}-{slugify(class_names[label])}"
        rel_path = f"{class_dir}/{idx:06d}.jpg"
        out_path = split_dir / rel_path
        if not out_path.exists():
            sample["image"].convert("RGB").save(out_path, quality=95)
        labels.append([rel_path, label])

    with (split_dir / "dataset.json").open("w") as f:
        json.dump({"labels": labels}, f)
    return class_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="tanganke/stanford_cars")
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "hf_datasets_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "stanford_cars_imagefolder",
    )
    args = parser.parse_args()

    classes = None
    for split in ("train", "test"):
        split_classes = export_split(args.dataset_name, split, args.cache_dir, args.output_dir)
        if classes is None:
            classes = split_classes
        elif classes != split_classes:
            raise RuntimeError("Train/test class names differ")

    with (args.output_dir / "classes.json").open("w") as f:
        json.dump({str(i): name for i, name in enumerate(classes or [])}, f, indent=2)
    print(f"Exported Stanford Cars to {args.output_dir}")


if __name__ == "__main__":
    main()
