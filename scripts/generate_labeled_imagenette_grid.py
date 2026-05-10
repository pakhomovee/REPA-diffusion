#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont


IMAGENETTE_CLASSES = [
    ("n01440764", "tench"),
    ("n02102040", "English springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chain saw"),
    ("n03028079", "church"),
    ("n03394916", "French horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
]


def parse_projector_embed_dims(value, state_dict):
    value = (value or "auto").strip().lower()
    if value in {"", "none", "null", "no"}:
        return []
    if value != "auto":
        return [int(part.strip()) for part in value.split(",") if part.strip()]

    dims = []
    prefix = "projectors."
    for key, tensor in state_dict.items():
        if key.startswith(prefix) and key.endswith(".4.weight"):
            dims.append((int(key.split(".")[1]), tensor.shape[0]))
    return [dim for _, dim in sorted(dims)]


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_grid(images, labels, footer, out_path, cols=4):
    tile_w, tile_h = images[0].size
    label_h = 34
    footer_h = 74
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h) + footer_h), "white")
    draw = ImageDraw.Draw(grid)
    label_font = load_font(18)
    footer_font = load_font(16)

    for idx, (image, label) in enumerate(zip(images, labels)):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        draw.rectangle([x, y, x + tile_w, y + label_h], fill=(245, 245, 245))
        draw.text((x + 8, y + 7), label, fill=(0, 0, 0), font=label_font)
        grid.paste(image, (x, y + label_h))

    footer_y = rows * (tile_h + label_h)
    draw.rectangle([0, footer_y, grid.width, grid.height], fill=(235, 235, 235))
    for line_idx, line in enumerate(footer):
        draw.text((10, footer_y + 10 + line_idx * 24), line, fill=(0, 0, 0), font=footer_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to train.py checkpoint .pt")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model", default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--projector-embed-dims", default="auto")
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--class-ids",
        default=None,
        help="Optional comma-separated class ids. If fewer than num-images are given, they are repeated.",
    )
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--mode", choices=["sde", "ode"], default="sde")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = Path(os.environ.get("REPA_ROOT", Path(__file__).resolve().parent.parent))
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    repa_dir = project_root / "REPA"
    sys.path.insert(0, str(repa_dir))
    os.chdir(repa_dir)

    from models.sit import SiT_models
    from samplers import euler_maruyama_sampler, euler_sampler

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt[args.weights]
    projector_dims = parse_projector_embed_dims(args.projector_embed_dims, state_dict)

    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=projector_dims,
        encoder_depth=args.encoder_depth,
        fused_attn=False,
        qk_norm=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    z = torch.randn(
        args.num_images, model.in_channels, latent_size, latent_size,
        device=device, generator=generator,
    )
    if args.class_ids:
        class_ids = [int(part.strip()) for part in args.class_ids.split(",") if part.strip()]
        if not class_ids:
            raise ValueError("--class-ids was provided but no class ids were parsed")
        bad_ids = [class_id for class_id in class_ids if class_id < 0 or class_id >= args.num_classes]
        if bad_ids:
            raise ValueError(f"class ids out of range for num_classes={args.num_classes}: {bad_ids}")
        repeated = (class_ids * math.ceil(args.num_images / len(class_ids)))[:args.num_images]
        y = torch.tensor(repeated, device=device, dtype=torch.long)
    else:
        y = torch.randint(0, args.num_classes, (args.num_images,), device=device, generator=generator)

    sampling_kwargs = dict(
        model=model,
        latents=z,
        y=y,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
        path_type=args.path_type,
    )
    if args.mode == "sde":
        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
    else:
        samples = euler_sampler(**sampling_kwargs).to(torch.float32)

    latents_scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    latents_bias = torch.tensor([0.0] * 4, device=device).view(1, 4, 1, 1)
    samples = vae.decode((samples - latents_bias) / latents_scale).sample
    samples = torch.clamp((samples + 1) / 2, 0, 1)
    samples = (255 * samples).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = ckpt_path.stem
        out_dir = project_root / "samples" / f"labeled_{ckpt_name}_{args.num_images}img_{args.num_steps}steps_{stamp}"
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pil_images = []
    labels = []
    metadata = []
    for idx, (array, class_id_tensor) in enumerate(zip(samples, y.cpu())):
        class_id = int(class_id_tensor)
        wnid, name = IMAGENETTE_CLASSES[class_id] if class_id < len(IMAGENETTE_CLASSES) else ("?", f"class {class_id}")
        image = Image.fromarray(array)
        image_path = out_dir / f"{idx:03d}_class-{class_id:02d}_{wnid}.png"
        image.save(image_path)
        pil_images.append(image)
        labels.append(f"{class_id}: {name}")
        metadata.append({"index": idx, "class_id": class_id, "wnid": wnid, "name": name, "file": image_path.name})

    footer = [
        f"ckpt={ckpt_path}",
        f"model={args.model}, weights={args.weights}, mode={args.mode}, steps={args.num_steps}, seed={args.seed}, cfg={args.cfg_scale}, projector_dims={projector_dims or 'none'}",
    ]
    grid_path = out_dir / "grid.png"
    draw_grid(pil_images, labels, footer, grid_path, cols=args.cols)

    info = {
        "checkpoint": str(ckpt_path),
        "model": args.model,
        "weights": args.weights,
        "num_steps": args.num_steps,
        "mode": args.mode,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "class_ids": args.class_ids,
        "projector_embed_dims": projector_dims,
        "classes": metadata,
        "grid": str(grid_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(info, indent=2))
    print(grid_path)


if __name__ == "__main__":
    main()
