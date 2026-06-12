# REPA-Σ: gradient surgery for representation-alignment diffusion training

This directory documents the REPA-Σ experiments on CelebA-256 with SiT-B/2.

## Method

Standard REPA optimizes a scalarized objective `L = L_diff + λ·L_repa`, where
`L_repa` aligns intermediate SiT features with a frozen DINOv2-B teacher. When
the auxiliary gradient `g_repa` is anti-aligned with the diffusion gradient
`g_diff`, the alignment term actively fights the denoising objective.

**REPA-Σ** removes only the conflicting component of `g_repa` (PCGrad-style
projection) before adding it to `g_diff`, so the alignment signal can never
decrease the diffusion descent. Three variants are studied:

| Variant | Reference direction for the projection |
|---|---|
| `hard`  | the current per-batch `g_diff` |
| `bloop` | an **EMA** of `g_diff` (β = 0.99) — lower-variance, theoretically diffusion-consistent |
| `+ λ-anneal` | either of the above with `λ` linearly decayed to 0 |

The implementation lives in `REPA/train_sigma.py` (`--sigma-mode {off,hard,bloop}`,
`--sigma-bloop-beta`, `--lambda-anneal`). The `bloop` path keeps its EMA on-GPU
in fp32, skips updates on non-finite (fp16-overflow) gradients, and clips the
projection coefficient — without these guards a single `inf` poisons the EMA and
the run diverges to NaN.

## Results — FID (CelebA-256, 50k samples, clean-fid)

All runs share architecture, optimizer, data, batch size (128) and seed.

| Method | FID @ 200k | FID @ 400k |
|---|---|---|
| Baseline (no REPA)        | 6.560 | 5.869 |
| REPA (const-λ)            | 6.065 | 5.614 |
| REPA-Σ hard (const-λ)     | 6.031 | — |
| REPA + λ-anneal           | 6.114 | — |
| REPA-Σ + λ-anneal         | 6.015 | — |
| **REPA-Σ bloop (EMA)**    | **5.889** | **5.411** |

Raw per-checkpoint numbers are in the top-level `scores*.csv`; plots in
`fid_results/`.

## Long-horizon finding

Extending the baseline, REPA, and REPA-Σ-bloop runs to 400k tests the bias
prediction (standard REPA converges to a biased optimum; the EMA surgery does
not):

- The baseline's gap to REPA narrows over 200k→400k (6.56→5.87 vs 6.07→5.61) —
  consistent with REPA being biased.
- **REPA-Σ bloop is lower than both at all nine matched checkpoints in the
  220k–400k range** (sign test p = 0.002; p ≈ 0.03 after correcting for
  checkpoint autocorrelation). Mean paired gap: 0.22 FID vs REPA, 0.63 vs
  baseline.

Caveat: single training seed per method; sampling-seed variance only. Training-
seed robustness is left to future work.

## Reproduction

```bash
# Train (3× A100-40GB). Paths default to the repo root; override REPO_ROOT if needed.
bash scripts/train_repa_sigma_bloop_gpus5_7.sh            # 0 → 200k, EMA surgery
bash scripts/resume_celeba_repa_sigma_bloop_200k_to_400k_gpus5_7.sh   # 200k → 400k

# Evaluate FID/KID at every checkpoint (6-GPU sampling + clean-fid scoring).
# Generates 50k samples per checkpoint, scores, appends to CSV, frees the npz.
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 NUM_GPUS=6 python score_400k_and_bloop.py
```

`fid_kid.py` is the reusable FID/KID pipeline; the `score_*.py` drivers wrap it
with checkpoint discovery, robust DDP-sampling recovery, and CSV-based resume.
