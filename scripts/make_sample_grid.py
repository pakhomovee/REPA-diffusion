#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    paths = sorted(args.sample_dir.glob("*.png"))[: args.limit]
    if not paths:
        raise SystemExit(f"No PNG files found in {args.sample_dir}")

    images = [Image.open(path).convert("RGB") for path in paths]
    tile_w, tile_h = images[0].size
    rows = math.ceil(len(images) / args.cols)
    title_h = 34 if args.title else 0
    grid = Image.new("RGB", (args.cols * tile_w, rows * tile_h + title_h), "white")

    if args.title:
        draw = ImageDraw.Draw(grid)
        draw.text((10, 10), args.title, fill=(0, 0, 0))

    for idx, image in enumerate(images):
        x = (idx % args.cols) * tile_w
        y = title_h + (idx // args.cols) * tile_h
        grid.paste(image, (x, y))

    out = args.out or args.sample_dir / "grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(out)


if __name__ == "__main__":
    main()
