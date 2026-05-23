# REPA-Diffusion

Research project investigating **REPresentation Alignment (REPA)** as a training objective for diffusion transformers under constrained data and compute.
The companion report is [`main.pdf`](main.pdf).

The REPA implementation is vendored as a git submodule under [`REPA/`](REPA/), a fork of [sihyun-yu/REPA](https://github.com/sihyun-yu/REPA) with the project-specific patches listed below.

---

## Status

| Dataset | Configuration | Steps | Status |
|---|---|---|---|
| CelebA-256 (16 classes) | SiT-B/2 baseline (no REPA) | 200 000 | Done |
| CelebA-256 (16 classes) | SiT-B/2 + REPA (DINOv2-ViT-B) | 200 000 | Done |
| Imagenette-256 (10 classes) | SiT-B/2 baseline (no REPA) | 30 000 | Done |
| Imagenette-256 (10 classes) | SiT-B/2 + REPA (DINOv2-ViT-B) | 100 000 | Done |
| Imagenette-256 (10 classes) | SiT-B/2 baseline (no REPA) | 100 000 | Planned |
| Stanford Cars / LSUN Church | SiT-B/2 ± REPA | — | Planned (export + launcher scripts only) |

Headline CelebA result, measured on 10 000 generated samples per checkpoint with `clean-fid`:

| Metric @ 200k steps | Baseline | REPA (DINOv2-B) |
|---|---|---|
| FID  | 6.56 | **6.06** |
| KID × 10³ | 4.49 | **4.37** |

REPA reaches the baseline's 200 k-step FID at roughly 130 k–140 k steps (≈ 1.5× speedup along the step axis); the gap stops widening at later checkpoints.

The full per-checkpoint trajectory is in [`scores.csv`](scores.csv) and [`fid_results/`](fid_results/).

---

## Repository layout

```
repa_diffusion_celeba/
├── README.md
├── main.pdf                              ← project report
├── scores.csv                            ← FID / KID per checkpoint (Baseline + REPA, CelebA)
│
├── REPA/                                 ← git submodule, branch=main on sekopylov/REPA
│   ├── train.py                          ← training entry-point (accelerate launch …)
│   ├── generate.py                       ← class-conditional ODE/SDE sampler
│   ├── loss.py                           ← SILoss: denoising + REPA projection loss
│   ├── samplers.py                       ← Euler ODE/SDE samplers, CFG
│   ├── models/sit.py                     ← SiT model definitions
│   ├── dataset.py                        ← CustomDataset (raw images + VAE latents + labels)
│   ├── preprocessing/                    ← VAE encoder utilities
│   └── utils.py                          ← teacher-encoder loading (DINOv2, CLIP, MAE, …)
│
├── scripts/
│   ├── export_celeba_for_repa.py         ← CelebA → REPA image folder + dataset.json
│   ├── export_stanford_cars_for_repa.py  ← Stanford Cars export (planned experiment)
│   ├── celeba.py                         ← CelebADataset helper (gdown loader)
│   │
│   ├── run_celeba_baseline.sh            ← CelebA, no REPA
│   ├── run_celeba_repa.sh                ← CelebA, REPA + DINOv2-B
│   ├── run_celeba_encode_and_train_4gpu.sh  ← end-to-end: parallel VAE encode then train
│   ├── resume_celeba_baseline_40k_to_200k_gpus2-4.sh   ← actual launcher for the 200k baseline
│   ├── resume_celeba_repa_80k_to_200k_gpus1_5_6_7.sh   ← actual launcher for the 200k REPA run
│   │
│   ├── run_imagenette_baseline.sh        ← multi-GPU Imagenette baseline
│   ├── run_imagenette_repa.sh            ← multi-GPU Imagenette REPA
│   ├── run_imagenette_baseline_1gpu.sh   ← single-GPU variant (gradient accumulation)
│   ├── run_imagenette_repa_1gpu.sh       ← single-GPU variant (gradient accumulation)
│   ├── launch_imagenette_baseline_30k_gpus0-3.sh        ← nohup launcher (used for the 30k run)
│   ├── launch_imagenette_repa_30k_gpus4-7.sh            ← nohup launcher (used for the 100k REPA run)
│   ├── launch_imagenette_repa_50k_to_100k_gpus0-5.sh    ← continuation launcher
│   │
│   ├── run_stanford_cars_baseline.sh     ← Stanford Cars (planned)
│   ├── run_stanford_cars_repa.sh         ← Stanford Cars (planned)
│   │
│   ├── generate_imagenette_samples.sh    ← sampling wrapper for Imagenette checkpoints
│   ├── generate_labeled_imagenette_grid.py  ← labelled class-grid generator
│   ├── generate_labeled_celeba_grid.py      ← attribute-bitmask grid (16 CelebA classes)
│   ├── make_sample_grid.py               ← contact-sheet builder
│   │
│   ├── probe_gradient_geometry.py        ← per-(checkpoint, t) gradient-geometry probe
│   │
│   ├── show_imagenette_30k_progress.sh   ← tail TensorBoard events
│   ├── start_tensorboard.sh              ← launch TensorBoard
│   └── utils/                            ← shared helpers
│
├── fid_kid.py                            ← FID/KID/LPIPS evaluator (single-process, multi-GPU samplers)
├── fid_eval_fast.py                      ← multi-GPU parallel scorer (one worker per GPU)
├── finish_scoring.py                     ← one-shot driver that produced scores.csv
├── fid_results/                          ← committed FID/KID plots
│
├── nearest_train_image_audit.ipynb       ← pixel-MSE nearest-train-image audit
├── fid_kid_lpips.ipynb                   ← exploratory Colab-style notebook (superseded by fid_kid.py)
│
├── docs/assets/                          ← figures referenced from this README and main.pdf
├── reports/                              ← analysis artifacts (gitignored except small figures)
│   ├── grad_geometry/                    ← cos(grad L_diff, grad L_REPA) plots
│   └── nearest_train_pixel_audit/        ← audit outputs
│
├── runs/                                 ← experiment outputs (gitignored)
│   ├── celeba_sit_b2_baseline_gpus2-7_40k/
│   └── celeba_sit_b2_repa_dinov2b_gpus4-7/
├── logs/                                 ← stdout logs (gitignored)
├── samples/                              ← generated PNGs (gitignored)
└── data/                                 ← raw + preprocessed datasets (gitignored)
```

---

## Setup

```bash
git clone --recurse-submodules https://github.com/pakhomovee/REPA-diffusion.git repa_diffusion_celeba
cd repa_diffusion_celeba
python -m venv .venv && source .venv/bin/activate
pip install -r REPA/requirements.txt
pip install "setuptools<81" --force-reinstall    # TensorBoard / setuptools compatibility shim
pip install gdown natsort                        # required by scripts/celeba.py for CelebA download
```

A working installation requires CUDA and `accelerate` (already in `REPA/requirements.txt`).
The CelebA pipeline below additionally requires `clean-fid` and `lpips` for the FID/KID/LPIPS stage:

```bash
pip install clean-fid lpips
```

---

## CelebA pipeline

CelebA-256 is the project's primary single-domain experiment (202 599 face images, 40 binary attributes). The class label is a bitmask over a chosen subset of attributes; with the default four (`Male`, `Smiling`, `Young`, `Attractive`) the class count is `2⁴ = 16`.

### 1. Export images

Downloads the dataset from Google Drive on first use and writes a REPA-compatible image folder:

```bash
python scripts/export_celeba_for_repa.py
```

Output:
```
data/celeba256/
├── images/         ← 202 599 centre-cropped 256×256 JPEGs (000000.jpg … 202598.jpg)
└── dataset.json    ← [["000000.jpg", <class_id>], …]
```

Overrides:
```bash
python scripts/export_celeba_for_repa.py \
    --root-dir <path> --output-dir <path> \
    --selected-attrs Male Smiling Young \
    --resolution 256
```
The class count must equal `2 ^ |selected-attrs|`.

### 2. Encode VAE latents

REPA trains on SD-VAE latents rather than raw pixels. The upstream tool encodes serially; for the actual runs we used a parallel multi-GPU encoder embedded in `run_celeba_encode_and_train_4gpu.sh`. Either path is fine.

Single-GPU (upstream):
```bash
cd REPA/preprocessing
python dataset_tools.py encode \
    --source ../../data/celeba256 \
    --dest   ../../data/celeba256/vae-sd \
    --model-url stabilityai/sd-vae-ft-mse
```

Parallel (one worker per GPU, batched, atomic writes):
```bash
ENC_GPUS=4,5,6,7 bash scripts/run_celeba_encode_and_train_4gpu.sh
# Detects whether vae-sd/ is already complete and skips re-encoding.
```

Output layout:
```
data/celeba256/
├── images/
├── vae-sd/
│   ├── dataset.json
│   └── 00000/img-mean-std-00000000.npy …
```

### 3. Training

Both training scripts wrap `accelerate launch REPA/train.py` with the project's hyperparameters (see the *Configuration* table below) and accept the same environment-variable overrides.

Baseline (no REPA, `--enc-type none`, `--proj-coeff 0.0`):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_PROCESSES=6 BATCH_SIZE=256 \
MAX_TRAIN_STEPS=200000 CHECKPOINTING_STEPS=10000 \
EXP_NAME=celeba_sit_b2_baseline \
bash scripts/run_celeba_baseline.sh
```

REPA with DINOv2-ViT-B teacher (`--enc-type dinov2-vit-b`, `--proj-coeff 0.5`, `--encoder-depth 4`):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_PROCESSES=6 BATCH_SIZE=256 \
MAX_TRAIN_STEPS=200000 CHECKPOINTING_STEPS=10000 \
EXP_NAME=celeba_sit_b2_repa_dinov2b \
bash scripts/run_celeba_repa.sh
```

The two production 200 k-step runs reported in [`scores.csv`](scores.csv) were resumed from intermediate checkpoints:

```bash
bash scripts/resume_celeba_baseline_40k_to_200k_gpus2-4.sh        # 40k → 200k on 3 GPUs, BATCH_SIZE=126
bash scripts/resume_celeba_repa_80k_to_200k_gpus1_5_6_7.sh        # 80k → 200k on 4 GPUs, BATCH_SIZE=128
```

If a different `--selected-attrs` was used at export, pass the matching class count via `NUM_CLASSES`.

### 4. Sampling

`scripts/generate_labeled_celeba_grid.py` is a thin wrapper around the Imagenette grid generator that overrides the per-class label set with the 16 CelebA attribute bitmasks. It calls `REPA/generate.py` under the hood.

Example (REPA model, ODE sampler, CFG = 4):
```bash
CUDA_VISIBLE_DEVICES=0 /home/seankopylov/.venv/bin/python \
    scripts/generate_labeled_celeba_grid.py \
    --ckpt runs/celeba_sit_b2_repa_dinov2b_gpus4-7/checkpoints/0200000.pt \
    --out-dir samples/celeba_repa_200k_cfg4 \
    --class-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --num-images 10 \
    --num-steps 250 \
    --mode ode \
    --cfg-scale 4.0 \
    --weights model
```

Key flags:
- `--weights model | ema` — raw optimiser weights vs EMA weights. EMA is closer to the REPA paper's sampling setup; `model` is informative at early checkpoints.
- `--mode ode | sde` — deterministic ODE Euler vs stochastic SDE.
- `--cfg-scale` — classifier-free guidance scale (1 = no guidance).
- `--class-ids` — class id list; deterministic per `--seed`.

### 5. FID / KID / LPIPS evaluation

[`fid_kid.py`](fid_kid.py) sweeps a list of checkpoints, samples a configurable number of images per checkpoint via `REPA/generate.py`, and computes FID, KID, and intra-model LPIPS diversity. Implementation notes:

- Real-image InceptionV3 statistics are precomputed once (cached under `clean-fid`'s stats dir) so subsequent scorings reuse them.
- `.npz` sample files are memory-mapped and streamed directly through InceptionV3, avoiding the round-trip through individual PNG files.
- A single InceptionV3 instance is reused across all scorings.
- LPIPS diversity is mean pairwise LPIPS *within* a single model's generated samples (the REPA-paper metric).

Quick smoke test (3 checkpoints × 2 000 samples each):
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
    /home/seankopylov/.venv/bin/python fid_kid.py --quick
```

Full sweep (the run that produced [`scores.csv`](scores.csv)):
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
    nohup /home/seankopylov/.venv/bin/python fid_kid.py > logs/fid_kid.log 2>&1 &
```

[`fid_eval_fast.py`](fid_eval_fast.py) is an alternative one-worker-per-GPU evaluator that distributes checkpoints round-robin and loads each `.npz` via a direct file read (bypassing the Python `zipfile` chunker); it shares the same fast-NPZ utilities and stats-caching logic.

### 6. Gradient-geometry probe

[`scripts/probe_gradient_geometry.py`](scripts/probe_gradient_geometry.py) measures the cosine between the denoising-loss gradient and the REPA-loss gradient, and the corresponding gradient-norm ratio, for each (checkpoint, diffusion-timestep) cell on a REPA-trained model. Both quantities are evaluated in two scopes:

- `blocks_0_3` — embeddings plus the transformer blocks before the REPA projector tap, i.e. the parameters on which `∇L_REPA` is non-zero. This is the geometrically meaningful subspace.
- `full` — all SiT parameters except the projector head.

A frozen 128-image probe set (CelebA latents, raw images for the DINOv2 teacher, labels, and per-microbatch noise tensors) is cached to disk so the same `(z, ε, t)` tuple is used in every cell. Each cell does one forward pass and two backward passes (`retain_graph=True`) using the same `SILoss` formula as training. Workers are distributed one-per-GPU.

Run on five checkpoints × five `t` values × eight microbatches (5×5×8 = 200 cells):
```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    /home/seankopylov/.venv/bin/python scripts/probe_gradient_geometry.py \
    --run-dir runs/celeba_sit_b2_repa_dinov2b_gpus4-7 \
    --ckpt-steps 10000,50000,100000,150000,200000 \
    --t-values  0.1,0.3,0.5,0.7,0.9 \
    --n-microbatches 8 --batch-size 16 \
    --gpus 3,4,5,6,7 --weights raw
```

Re-run with EMA weights, reusing the same probe cache:
```bash
… --weights ema --skip-prepare
```

Output:
- `reports/grad_geometry/measurements_{raw,ema}.csv` — raw per-cell measurements (gitignored).
- `reports/grad_geometry/cos_vs_t__{full,blocks_0_3}__{raw,ema}.png`
- `reports/grad_geometry/ratio_vs_t__{full,blocks_0_3}__{raw,ema}.png`

On 5 × A100 the full grid finishes in roughly two minutes (probe-data preparation included).

### 7. Memorization audit

[`nearest_train_image_audit.ipynb`](nearest_train_image_audit.ipynb) computes pixel-MSE nearest neighbours between generated images and training images, then renders a grid of the closest matches. The notebook is parameterised at the top by:

```python
GENERATED_DIR = PROJECT_ROOT / "samples" / "celeba_repa_40k_model_cfg4_ode25_classes0-15_seed163"
TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "celeba256" / "images"
REPORT_DIR = PROJECT_ROOT / "reports" / "nearest_train_pixel_audit" / "<name>"
```

The same notebook applies to Imagenette by substituting the corresponding directories.

---

## Imagenette pipeline

Imagenette-256 (10 classes, ~9 469 training images) is used as a small-data stress test of REPA. Results are reported in [`main.pdf`](main.pdf) §7.1. Memorization becomes a dominant effect at 100 000 steps on this scale, motivating the CelebA experiments above.

### Dataset preparation

```bash
mkdir -p data/downloads data/raw
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -P data/downloads/
tar -xzf data/downloads/imagenette2-320.tgz -C data/raw/

# Preprocess to 256×256 + SD-VAE latents
python REPA/preprocessing/encoders.py \
    --data_path data/raw/imagenette2-320/train \
    --output_path data/imagenette256-train \
    --resolution 256
python REPA/preprocessing/encoders.py \
    --data_path data/raw/imagenette2-320/val \
    --output_path data/imagenette256-val \
    --resolution 256
```

Expected layout:
```
data/imagenette256-train/
├── images/{dataset.json, 00000/img00000000.png …}
└── vae-sd/{dataset.json, 00000/img-mean-std-00000000.npy …}
```

### Training

Multi-GPU baseline (30 k steps, used to produce the comparison in main.pdf §7.1):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=256 \
MAX_TRAIN_STEPS=30000 CHECKPOINTING_STEPS=5000 \
EXP_NAME=imagenette_sit_b2_baseline_30k \
bash scripts/run_imagenette_baseline.sh
```

Multi-GPU REPA (100 k steps):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=256 \
MAX_TRAIN_STEPS=100000 CHECKPOINTING_STEPS=10000 \
EXP_NAME=imagenette_sit_b2_repa_100k \
bash scripts/run_imagenette_repa.sh
```

`nohup`-style launchers used for the two actual runs are in `scripts/launch_imagenette_*.sh`. A continuation launcher (`launch_imagenette_repa_50k_to_100k_gpus0-5.sh`) demonstrates the resume-from-checkpoint path.

Single-GPU variants (`run_imagenette_*_1gpu.sh`) use gradient accumulation to maintain an effective batch size of 256; `LOCAL_BATCH` and `ACCUM_STEPS` are overridable for low-VRAM machines.

### Sampling

```bash
CUDA_VISIBLE_DEVICES=0 /home/seankopylov/.venv/bin/python \
    scripts/generate_labeled_imagenette_grid.py \
    --ckpt runs/<exp>/checkpoints/0100000.pt \
    --out-dir samples/<name> \
    --class-ids 0,1,2,3,4,5,6,7,8,9 \
    --num-images 10 --num-steps 250 --mode ode --cfg-scale 4.0 \
    --weights model
```

### Monitoring

```bash
watch -n 30 ./scripts/show_imagenette_30k_progress.sh

BASELINE_EXP=<baseline_exp> REPA_EXP=<repa_exp> \
    ./scripts/start_tensorboard.sh
ssh -L 6006:127.0.0.1:6006 <server>
# then open http://localhost:6006
```

---

## Stanford Cars (planned)

Dataset export and training launchers are present but no training has been run.

```bash
python scripts/export_stanford_cars_for_repa.py    # downloads from Hugging Face (the Stanford URLs are dead)
# Then preprocess with REPA/preprocessing/encoders.py as in the Imagenette pipeline.
bash scripts/run_stanford_cars_baseline.sh
bash scripts/run_stanford_cars_repa.sh
```

`--num-classes=196` with the standard SiT-B/2 + DINOv2-B configuration is the intended setting.

---

## Configuration

All training runs share the SiT-B/2 hyperparameters from the REPA paper:

| Parameter | Value |
|---|---|
| Model | SiT-B/2 |
| Resolution | 256 × 256 (latent 32 × 32 × 4) |
| Path type | linear (`z_t = (1 − t) z + t ε`) |
| Prediction target | v (velocity) |
| Weighting | uniform |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Betas | (0.9, 0.999) |
| Weight decay | 0 |
| Mixed precision | fp16 (training), fp32 (probe) |
| TF32 matmul | enabled |
| Global batch | 256 (varies for resumed runs: 126 / 128) |
| CFG dropout | 0.1 |
| REPA teacher | DINOv2-ViT-B |
| REPA projection coefficient `λ` | 0.5 |
| REPA encoder depth (projector tap) | 4 |

CelebA: 16 classes; Imagenette: 10 classes; Stanford Cars: 196 classes.

---

## REPA submodule patches

The vendored REPA fork ([`sekopylov/REPA`](https://github.com/sekopylov/REPA), branch `main`) carries the following project-specific changes:

| Change | File(s) |
|---|---|
| True baseline mode: `--enc-type none` skips the teacher entirely | `train.py`, `loss.py` |
| Non-ImageNet class counts via `--num-classes` | `train.py` |
| CFG null-class fix for small class counts in samplers | `samplers.py` |
| Per-step timing logs (`step_time`, `teacher_time`) | `train.py` |
| TensorBoard tracker support (`--report-to tensorboard`) | `train.py` |
| `--projector-embed-dims none` for baseline checkpoint generation | `generate.py` |
| `weights_only=False` checkpoint loading for PyTorch 2.6 | `train.py`, `generate.py` |
| Safer checkpoint save via `accelerator.unwrap_model()` | `train.py` |
| `--no-sample-at-step-one` flag | `train.py` |
| `--vae-chunk-size` for OOM-safe VAE decode at large per-process batch sizes | `generate.py` |
| Parallel npz builder in `create_npz_from_sample_folder` | `generate.py` |

`.gitmodules` pins `branch = main`, so `git submodule update --remote REPA` keeps the working tree current.

---

## Links

- Main project repository: [`pakhomovee/REPA-diffusion`](https://github.com/pakhomovee/REPA-diffusion)
- REPA submodule (fork): [`sekopylov/REPA`](https://github.com/sekopylov/REPA)
- Original REPA paper: Sihyun Yu et al., *Representation Alignment for Generation*, ICLR 2025 ([arXiv:2410.06940](https://arxiv.org/abs/2410.06940))
- Upstream REPA code: [`sihyun-yu/REPA`](https://github.com/sihyun-yu/REPA)
