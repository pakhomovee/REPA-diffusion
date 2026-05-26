#!/usr/bin/env python3
"""Merge per-GPU VAE encoding shards into a single vae-sd directory.

After running dataset_tools.py encode with --rank / --world-size, you
get one sub-directory per GPU:
    <dest>/shard_0000_of_0004/
    <dest>/shard_0001_of_0004/
    ...

This script merges all shards into <dest>/ directly, combines the
per-shard dataset.json files into one, and optionally removes the shard
sub-directories.

Usage:
    python merge_vae_shards.py --dest /data/vae-sd --world-size 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def merge(dest: Path, world_size: int, remove_shards: bool) -> None:
    shard_dirs = [dest / f"shard_{r:04d}_of_{world_size:04d}" for r in range(world_size)]

    # Validate all shards are present
    missing = [str(d) for d in shard_dirs if not d.is_dir()]
    if missing:
        raise RuntimeError(
            f"Missing shard directories:\n" + "\n".join(missing)
        )

    all_labels: list = []
    total_files = 0

    for shard_dir in shard_dirs:
        # Load this shard's dataset.json
        json_path = shard_dir / "dataset.json"
        with json_path.open() as f:
            meta = json.load(f)
        shard_labels = meta.get("labels") or []

        # Move every .npy file into dest, preserving sub-folder structure
        for npy_path in sorted(shard_dir.rglob("*.npy")):
            rel = npy_path.relative_to(shard_dir)
            out_path = dest / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(npy_path), str(out_path))
            total_files += 1

        all_labels.extend(shard_labels)
        print(f"  merged {shard_dir.name}  ({len(shard_labels)} entries)")

    # Sort by filename so the combined dataset.json is in index order
    all_labels.sort(key=lambda x: x[0] if x is not None else "")

    labels_valid = all(x is not None for x in all_labels)
    metadata = {"labels": all_labels if labels_valid else None}
    out_json = dest / "dataset.json"
    with out_json.open("w") as f:
        json.dump(metadata, f)
    print(f"\nWrote {out_json}  ({len(all_labels)} entries)")
    print(f"Total .npy files moved: {total_files}")

    if remove_shards:
        for shard_dir in shard_dirs:
            shutil.rmtree(shard_dir)
        print("Removed shard directories.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-GPU VAE shard directories into one."
    )
    parser.add_argument("--dest",        required=True, type=Path,
                        help="Top-level VAE output directory (parent of shard_* dirs).")
    parser.add_argument("--world-size",  required=True, type=int,
                        help="Number of shards (= number of GPUs used during encoding).")
    parser.add_argument("--keep-shards", action="store_true",
                        help="Do not delete shard sub-directories after merging.")
    args = parser.parse_args()
    merge(args.dest, args.world_size, remove_shards=not args.keep_shards)


if __name__ == "__main__":
    main()
