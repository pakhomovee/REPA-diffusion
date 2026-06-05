#!/usr/bin/env python3
"""Export CompCars images with classes assigned at the MAKE (manufacturer) level.

Unlike export_compcars_for_repa.py which uses make×model as the class,
this script assigns one class per car manufacturer (e.g. Toyota, BMW, Audi).
This gives 163 classes at most, similar in granularity to Stanford Cars (196).

Kaggle dataset: https://www.kaggle.com/datasets/renancostaalencar/compcars

Output structure (REPA image-folder format):
    <output_dir>/
        000-toyota/
            000000.jpg
            ...
        001-bmw/
            ...
        dataset.json     <- {"labels": [["000-toyota/000000.jpg", 0], ...]}
        classes.json     <- {"0": "Toyota", "1": "BMW", ...}

After running this script, encode VAE latents with:
    cd REPA/preprocessing
    python dataset_tools.py encode \\
        --source <output_dir> \\
        --dest <output_dir>/vae-sd \\
        --model-url stabilityai/sd-vae-ft-mse

Usage:
    # Export all makes
    python scripts/export_compcars_by_make.py

    # Keep only specific makes
    python scripts/export_compcars_by_make.py --makes Toyota Honda BMW Audi

    # Drop makes with fewer than N images
    python scripts/export_compcars_by_make.py --min-class-size 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── Helpers ────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "unknown"


def is_make_root(p: Path) -> bool:
    """True if ≥80% of direct children are numeric directories (make_id level)."""
    try:
        children = [c for c in p.iterdir() if c.is_dir()]
        if not children:
            return False
        numeric = sum(1 for c in children if c.name.isdigit())
        return numeric / len(children) >= 0.8
    except PermissionError:
        return False


def locate_image_root(root: Path) -> Path:
    """Find the directory whose children are numeric make_id folders."""
    for candidate in [root / "image", root / "data" / "image", root]:
        if candidate.is_dir() and is_make_root(candidate):
            return candidate

    for dirpath_str, dirnames, _ in os.walk(str(root)):
        dirpath = Path(dirpath_str)
        if any(part in {"label", "annotation", "misc", "train_test_split"}
               for part in dirpath.parts):
            dirnames[:] = []
            continue
        if is_make_root(dirpath):
            return dirpath

    zips = list(root.glob("*.zip"))
    if zips:
        print(f"Extracting {zips[0]} ...")
        with zipfile.ZipFile(zips[0], "r") as z:
            z.extractall(root)
        return locate_image_root(root)

    raise RuntimeError(
        f"Could not locate CompCars image/ directory in {root}.\n"
        "Download from https://www.kaggle.com/datasets/renancostaalencar/compcars "
        f"and extract into {root}."
    )


def load_make_names(misc_dir: Path) -> dict[int, str]:
    """
    Load make_id -> make name from misc/make_model_name.mat.
    Returns {make_id (1-indexed): "Toyota", ...}
    Falls back to numeric strings if scipy is unavailable or file is missing.
    """
    mat_path = misc_dir / "make_model_name.mat"
    if not mat_path.exists():
        print(f"Warning: {mat_path} not found — using numeric make IDs as names.")
        return {}

    try:
        import scipy.io as sio  # type: ignore
    except ImportError:
        print("Warning: scipy not installed — using numeric make IDs as names.")
        print("         Run:  pip install scipy")
        return {}

    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    make_name_list = mat.get("make_names") or mat.get("MN")
    if make_name_list is None:
        print("Warning: make_names not found in .mat file — using numeric IDs.")
        return {}

    return {i + 1: str(name).strip() for i, name in enumerate(make_name_list)}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_samples(image_root: Path) -> list[tuple[Path, int]]:
    """Walk image_root and return (image_path, make_id) pairs."""
    samples = []
    for img_path in sorted(image_root.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = img_path.relative_to(image_root).parts
        # parts = (make_id, model_id, year, filename)
        try:
            make_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        samples.append((img_path, make_id))
    return samples


# ── Main export ────────────────────────────────────────────────────────────

def export(
    root_dir: Path,
    misc_dir: Path,
    output_dir: Path,
    resolution: int,
    min_class_size: int,
    filter_makes: list[str] | None,
    max_images_per_class: int | None = None,
) -> None:

    image_root = locate_image_root(root_dir)
    print(f"Image root: {image_root}")

    make_names = load_make_names(misc_dir)

    print("Scanning images ...")
    samples = collect_samples(image_root)
    print(f"Found {len(samples)} images across {len(set(m for _, m in samples))} makes.")

    # Count images per make
    counts = Counter(make_id for _, make_id in samples)

    # Resolve display name for each make_id
    def make_display(make_id: int) -> str:
        return make_names.get(make_id, f"make{make_id}")

    # Apply --makes filter (substring match on name, case-insensitive)
    if filter_makes:
        terms = [t.lower() for t in filter_makes]
        def name_matches(make_id: int) -> bool:
            name = make_display(make_id).lower()
            return any(t in name for t in terms)
    else:
        name_matches = lambda make_id: True  # keep all

    # Apply --min-class-size and name filter; build final class list
    kept_makes = sorted(
        make_id for make_id in counts
        if counts[make_id] >= min_class_size and name_matches(make_id)
    )

    if not kept_makes:
        raise RuntimeError(
            "No makes survived filtering. "
            "Try lowering --min-class-size or adjusting --makes."
        )

    make_to_class = {make_id: class_id for class_id, make_id in enumerate(kept_makes)}
    class_names   = {class_id: make_display(make_id)
                     for make_id, class_id in make_to_class.items()}

    print(f"\nClasses after filtering: {len(kept_makes)}")
    for class_id, make_id in enumerate(kept_makes):
        print(f"  {class_id:>4}  {make_display(make_id):<30}  ({counts[make_id]} images)")

    # Filter samples
    kept_samples = [
        (path, make_to_class[make_id])
        for path, make_id in samples
        if make_id in make_to_class
    ]
    print(f"\nTotal images to export: {len(kept_samples)}")

    # Create output class directories
    output_dir.mkdir(parents=True, exist_ok=True)
    class_dirs: dict[int, Path] = {}
    for class_id, make_id in enumerate(kept_makes):
        cdir = output_dir / f"{class_id:03d}-{slugify(make_display(make_id))}"
        cdir.mkdir(parents=True, exist_ok=True)
        class_dirs[class_id] = cdir

    resize = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resolution),
    ])

    labels: list[list] = []
    class_counters: dict[int, int] = {}

    for img_path, class_id in tqdm(kept_samples, desc="Exporting"):
        img_idx = class_counters.get(class_id, 0)
        if max_images_per_class is not None and img_idx >= max_images_per_class:
            continue
        class_counters[class_id] = img_idx + 1

        rel_path = f"{class_dirs[class_id].name}/{img_idx:06d}.jpg"
        out_path = output_dir / rel_path

        if not out_path.exists():
            img = Image.open(img_path).convert("RGB")
            img = resize(img)
            img.save(out_path, quality=95)

        labels.append([rel_path, class_id])

    # Write dataset.json
    dataset_json = output_dir / "dataset.json"
    with dataset_json.open("w") as f:
        json.dump({"labels": labels}, f)

    # Write classes.json  {str(class_id): "make_name"}
    classes_json = output_dir / "classes.json"
    with classes_json.open("w") as f:
        json.dump({str(k): v for k, v in class_names.items()}, f, indent=2)

    num_classes = len(kept_makes)
    print(f"\nExported {len(labels)} images → {output_dir}")
    print(f"Number of classes: {num_classes}  (one per manufacturer)")
    print(f"dataset.json  → {dataset_json}")
    print(f"classes.json  → {classes_json}")
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


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export CompCars with one class per car manufacturer."
    )
    parser.add_argument(
        "--root-dir", type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "compcars",
        help="Root directory of the CompCars archive (parent of image/).",
    )
    parser.add_argument(
        "--misc-dir", type=Path, default=None,
        help=(
            "Path to the misc/ folder (contains make_model_name.mat). "
            "Defaults to <root-dir>/misc."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "compcars256_by_make",
        help="Destination directory for the exported dataset.",
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Output image resolution (default: 256).",
    )
    parser.add_argument(
        "--min-class-size", type=int, default=100, metavar="N",
        help=(
            "Drop makes with fewer than N images (default: 100). "
            "Helps avoid tiny classes from rare manufacturers."
        ),
    )
    parser.add_argument(
        "--makes", nargs="*", metavar="NAME", default=None,
        help=(
            "Optional whitelist of make names to include "
            "(case-insensitive substring match). "
            "Example: --makes Toyota Honda BMW Audi. "
            "If omitted, all makes passing --min-class-size are kept."
        ),
    )
    parser.add_argument(
        "--max-images-per-class", type=int, default=None, metavar="N",
        help=(
            "Cap the number of images exported per make class. "
            "Useful for smoke tests (e.g. --max-images-per-class 5). "
            "Default: no limit."
        ),
    )
    args = parser.parse_args()

    misc_dir = args.misc_dir or (args.root_dir / "misc")

    export(
        root_dir=args.root_dir,
        misc_dir=misc_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        min_class_size=args.min_class_size,
        filter_makes=args.makes,
        max_images_per_class=args.max_images_per_class,
    )


if __name__ == "__main__":
    main()
