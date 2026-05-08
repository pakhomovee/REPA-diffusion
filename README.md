# REPA Diffusion Experiments

A configurable codebase for training and evaluating latent diffusion models (SiT) with and without **REPA** (REPresentation Alignment) auxiliary loss. Supports ablation experiments across data scales, teacher encoder types, and image domains — all reproducible via YAML config files with no code changes between runs.

> Based on: *"Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think"* (Yu et al., ICLR 2025 Oral)

---

## Overview

REPA accelerates diffusion transformer training by aligning intermediate hidden states of the denoising model with patch-level features from a frozen pretrained visual encoder (teacher). The total training objective is:

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda \cdot \mathcal{L}_{\text{REPA}}$$

where $\mathcal{L}_{\text{diff}}$ is the MSE velocity-prediction loss and $\mathcal{L}_{\text{REPA}}$ is the negative mean patch-wise cosine similarity between projected student hidden states and teacher features.

---

## Repository Structure

```
.
├── data/
│   ├── dataset.py          # ImageNet + Stanford Cars loaders with deterministic subsampling
│   └── latent_cache.py     # Pre-encode dataset through VAE; serve cached latents at train time
│
├── models/
│   ├── autoencoder.py      # Frozen SD-VAE wrapper (encode / decode)
│   ├── sit.py              # SiT-S/B/L backbone — returns (v_pred, hidden_states)
│   ├── teachers.py         # Frozen teacher encoders: DINOv2 / CLIP / ResNet-50 / none
│   └── projection_head.py  # MLP: Linear → LayerNorm → GELU → Linear (student→teacher dim)
│
├── losses/
│   ├── diffusion_loss.py   # MSE(v_pred, eps − z)
│   ├── repa_loss.py        # −mean_cosine_sim(project(H_ℓ), teacher_features)
│   └── combined_loss.py    # L_diff + λ·L_REPA; returns loss + logging dict
│
├── train.py                # Single-entry training script (config-driven)
├── evaluate.py             # Linear probe (top-1/top-5) and FID evaluation
│
└── analysis/
    ├── plot_convergence.py     # FID vs. steps / wall-clock time curves
    ├── plot_ablations.py       # Bar charts and heatmaps across teachers / fractions
    └── plot_representations.py # Cosine similarity + linear probe accuracy over training
```

---

## Installation

```bash
git clone <repo>
cd <repo>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate
pip install torch-fidelity lpips timm
```

> **Note:** DINOv2 requires inputs with resolution divisible by 14. The teacher encoder automatically resizes pixel images to 224×224 internally before feature extraction.

---

## Quick Start

### 1. Build the latent cache

Pre-encode images through the frozen VAE once. This is required before training — encoding on-the-fly is ~10× slower.

```bash
# ImageNet — 10% subset at 256×256
python data/latent_cache.py \
  --dataset imagenet \
  --root /data/imagenet \
  --out_dir /cache/imagenet_f010_r256/train \
  --splits train \
  --fraction 0.10 \
  --resolution 256

# Stanford Cars — full dataset
python data/latent_cache.py \
  --dataset stanford_cars \
  --root /data/stanford_cars \
  --out_dir /cache/stanford_cars_r256/train \
  --splits train \
  --resolution 256
```

### 2. Smoke test (10 steps)

```bash
python train.py \
  --config configs/baseline_imagenet_10pct.yaml \
  --override total_steps=10 save_every=10 log_every=1
```

### 3. Launch training

```bash
# Baseline — pure diffusion, no REPA
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/baseline_imagenet_10pct.yaml &

# REPA + DINOv2
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/repa_dinov2_imagenet_10pct.yaml &
```

Multi-GPU (4 GPUs):
```bash
torchrun --nproc_per_node=4 train.py --config configs/repa_dinov2_imagenet_10pct.yaml
```

---

## Training

`train.py` is the single entry point for all experiments. It is fully controlled by a YAML config file — no code changes are needed between runs.

### Config override from CLI

Any config key can be overridden at launch without editing the file:

```bash
python train.py --config configs/baseline_imagenet_10pct.yaml \
  --override total_steps=200000 lambda=0.5 lr=2e-4
```

### Training loop (per step)

1. Sample batch `(z, label)` from latent cache
2. Sample `eps ~ N(0, I)`, `t ~ Uniform(0, 1)`
3. Compute `z_t = (1−t)·z + t·eps`
4. Forward pass through SiT → `v_pred`, `hidden_states`
5. If `teacher != "none"`: extract teacher features from pixel images
6. Compute combined loss: `L = L_diff + λ·L_REPA`
7. Backpropagate, optimizer step, EMA update
8. Log losses to `.jsonl` every `log_every` steps

### Training log format

Each line in `<checkpoint_dir>/train.jsonl`:
```json
{"step": 1000, "time_elapsed": 42.1, "loss_diff": 1.23, "loss_repa": -0.31, "cos_sim": 0.31, "loss_total": 0.98, "lambda": 0.5}
```

### Resuming

Training resumes automatically from `<checkpoint_dir>/latest.pt` if it exists.

---

## Evaluation

```bash
# Linear probe — top-1 / top-5 accuracy on frozen hidden states
python evaluate.py \
  --config configs/repa_dinov2_imagenet_10pct.yaml \
  --checkpoint checkpoints/repa_dinov2_imagenet_10pct/latest.pt \
  --eval_mode linear_probe \
  --probe_layer 6

# FID — generate 10k images and compare to real val set
python evaluate.py \
  --config configs/repa_dinov2_imagenet_10pct.yaml \
  --checkpoint checkpoints/repa_dinov2_imagenet_10pct/latest.pt \
  --eval_mode fid \
  --fid_num_samples 10000
```

Results are appended to `<checkpoint_dir>/eval.jsonl`.

---

## Analysis

```bash
# Convergence curves — FID vs. steps and wall-clock time
python analysis/plot_convergence.py --logdir checkpoints/

# Ablation bar charts and heatmaps
python analysis/plot_ablations.py --results_dir checkpoints/

# Representation diagnostics — cosine similarity + linear probe over training
python analysis/plot_representations.py --logdir checkpoints/
```

---

## Key Design Decisions

**No color jitter in augmentation.** Color jitter corrupts the spatial structure of features that the teacher relies on for alignment. Only random horizontal flip and center crop are applied.

**Latent cache is mandatory.** Encoding images through the VAE on every training step is ~10× slower than serving pre-cached latents. Always run `latent_cache.py` before training.

**Teacher inputs are always resized to 224×224.** DINOv2 uses 14×14 patches and requires resolution divisible by 14. Pixel images are resized internally inside `models/teachers.py` — the dataloader resolution and the teacher input resolution are decoupled.

**EMA weights are used for evaluation.** All `evaluate.py` runs load the EMA model by default (`use_ema=True`). EMA weights consistently outperform raw weights at evaluation time.

**`align_layer` is 0-indexed.** Layer 6 in a 12-block SiT-S corresponds to the middle block, which the REPA paper identifies as optimal for alignment.

---

## Module Reference

### `data/dataset.py`

| Function | Description |
|---|---|
| `get_dataset(dataset_name, root, split, fraction, resolution)` | Main entry point — returns a `torch.utils.data.Dataset` |
| `get_dataloader(...)` | Wraps `get_dataset` in a `DataLoader` |
| `deterministic_subset(dataset, fraction, seed)` | Stratified subsampling with fixed seed |

Supported `dataset_name` values: `"imagenet"`, `"stanford_cars"`.
Supported `fraction` values: `0.01`, `0.05`, `0.10`, `0.20`, `1.0`.

### `data/latent_cache.py`

| Class / Function | Description |
|---|---|
| `LatentCacheDataset(cache_dir)` | Reads pre-encoded `.pt` files; returns `(latent, label)` |
| `encode_and_cache(...)` | Encodes a dataset split through the VAE and saves to disk |

CLI arguments: `--dataset`, `--root`, `--out_dir`, `--splits`, `--fraction`, `--resolution`, `--batch_size`, `--device`.

### `models/sit.py`

| Class | Description |
|---|---|
| `SiT` | Diffusion transformer. `forward(z_t, t, labels)` returns `(v_pred, hidden_states)` where `hidden_states` is a `dict[int, Tensor]` keyed by block index |
| `get_sit_model(size, num_classes, input_size)` | Factory. `size` ∈ `{"S", "B", "L"}` |

### `models/teachers.py`

| Class | Description |
|---|---|
| `TeacherEncoder` | Unified frozen teacher. `forward(pixel_images)` returns `(B, N, C)` patch features |
| `get_teacher(name, device)` | Factory. `name` ∈ `{"dinov2", "clip", "resnet50", "none"}` |

All teachers run under `torch.no_grad()`. Inputs are resized to 224×224 and normalized with ImageNet stats internally.

### `models/projection_head.py`

Two-layer MLP: `Linear(student_dim) → LayerNorm → GELU → Linear(teacher_dim)`.

| Function | Description |
|---|---|
| `get_projection_head(student_dim, teacher_dim)` | Factory — instantiates head from dims |

### `losses/`

| Module | Returns |
|---|---|
| `DiffusionLoss` | Scalar MSE: `‖v_pred − (eps − z)‖²` |
| `REPALoss` | `(loss, cos_sim_mean)` — loss is `−mean_cosine_sim` |
| `CombinedLoss` | `(total_loss, log_dict)` — `log_dict` has keys `loss_diff`, `loss_repa`, `cos_sim`, `loss_total`, `lambda` |
| `get_combined_loss(repa_head, lam)` | Factory used by `train.py` |

---

## Environment

Tested with:
- Python 3.12
- PyTorch 2.x + CUDA 12.4+
- `diffusers >= 0.27`, `transformers >= 4.40`, `accelerate >= 0.29`
- `torch-fidelity`, `lpips`, `timm`
