#!/usr/bin/env python3
"""Diagnose a REPA data directory before training.

Usage:
    python scripts/check_dataset_dir.py --data-dir data/compcars256_by_make_test
"""
import argparse, json, os
from pathlib import Path

def check(data_dir: Path):
    images_dir   = data_dir / "images"
    vae_dir      = data_dir / "vae-sd"
    img_json     = images_dir / "dataset.json"
    vae_json     = vae_dir    / "dataset.json"

    print(f"\n{'='*60}")
    print(f"Data dir : {data_dir}  (exists={data_dir.exists()})")
    print(f"{'='*60}")

    # images/
    print(f"\nimages/  (exists={images_dir.exists()})")
    if images_dir.exists():
        imgs = sorted(images_dir.rglob("*.jpg")) + sorted(images_dir.rglob("*.png"))
        print(f"  .jpg/.png files : {len(imgs)}")
        if imgs:
            print(f"  first           : {imgs[0].relative_to(images_dir)}")
            print(f"  last            : {imgs[-1].relative_to(images_dir)}")
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
        print(f"  subdirectories  : {len(subdirs)}")
        for d in subdirs[:5]:
            n = len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            print(f"    {d.name}/  ({n} images)")
        if len(subdirs) > 5:
            print(f"    ... and {len(subdirs)-5} more")

    # images/dataset.json
    print(f"\nimages/dataset.json  (exists={img_json.exists()})")
    if img_json.exists():
        with img_json.open() as f:
            d = json.load(f)
        labels = d.get("labels")
        if labels is None:
            print("  labels = null  ← PROBLEM: no labels stored")
        else:
            print(f"  entries         : {len(labels)}")
            unique_cls = len(set(x[1] for x in labels))
            print(f"  unique classes  : {unique_cls}")
            if labels:
                print(f"  first entry     : {labels[0]}")
                print(f"  last  entry     : {labels[-1]}")

    # vae-sd/
    print(f"\nvae-sd/  (exists={vae_dir.exists()})")
    if vae_dir.exists():
        npys = sorted(vae_dir.rglob("*.npy"))
        print(f"  .npy files      : {len(npys)}")
        if npys:
            print(f"  first           : {npys[0].relative_to(vae_dir)}")
            print(f"  last            : {npys[-1].relative_to(vae_dir)}")

    # vae-sd/dataset.json
    print(f"\nvae-sd/dataset.json  (exists={vae_json.exists()})")
    if vae_json.exists():
        with vae_json.open() as f:
            d = json.load(f)
        labels = d.get("labels")
        if labels is None:
            print("  labels = null  ← PROBLEM: encode found no labels in images/dataset.json")
            print("                   Re-export images (they may be in the wrong directory)")
        else:
            print(f"  entries         : {len(labels)}")
            unique_cls = len(set(x[1] for x in labels if x is not None))
            print(f"  unique classes  : {unique_cls}")
            if labels:
                print(f"  first entry     : {labels[0]}")

    # Cross-check counts
    if vae_dir.exists() and images_dir.exists():
        n_imgs = len(sorted(images_dir.rglob("*.jpg")) + sorted(images_dir.rglob("*.png")))
        n_npys = len(sorted(vae_dir.rglob("*.npy")))
        print(f"\nCross-check:")
        print(f"  images : {n_imgs}")
        print(f"  latents: {n_npys}")
        if n_imgs == 0:
            print("  ✗ No images found — export step wrote files to wrong location")
        elif n_npys == 0:
            print("  ✗ No latents found — encode step did not run or failed")
        elif n_imgs != n_npys:
            print(f"  ✗ Count mismatch — dataset.py will raise AssertionError")
        else:
            print(f"  ✓ Counts match")

    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    check(args.data_dir)
