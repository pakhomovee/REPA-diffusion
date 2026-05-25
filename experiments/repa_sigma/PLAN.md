# REPA-Σ — Per-batch t-conditional gradient surgery for REPA

**Status**: implementation phase
**Author**: Sean Kopylov + Claude (REPA-Σ design discussion, 2026-05-25)
**Branch**: `experiment/repa-sigma` in parent repo and REPA submodule
**Tracker**: `experiments/repa_sigma/LOG.md`

---

## 1. The hypothesis

REPA's auxiliary alignment loss and the diffusion denoising loss have **statistically significant gradient conflict** at low-to-mid diffusion timesteps (t ∈ [0.1, 0.3]), measured directly in `reports/better_grad_geometry/`. The conflict is largest in early training but **persists at 150k** (cos = −0.083 with 95% CI [−0.122, −0.043] for blocks_0_3 at t=0.2). Where REPA fights denoising, every update partially undoes part of the other. HASTE (May 2025) handles this by *terminating* REPA after some step τ; we propose to handle it by *surgery* — removing only the locally conflicting component of the REPA gradient, batch by batch, preserving everywhere else REPA cooperates.

**Pitch line**: HASTE turns REPA off. REPA-Σ turns REPA off only where the gradients say it should be off.

## 2. Empirical anchors from existing data

From `reports/better_grad_geometry/measurements_raw.csv` (4096 images per cell, 11 t values, 5 checkpoints, n=64 microbatches, bootstrap 95% CIs):

| ckpt | scope | t   | cos (raw) | 95% CI | significant conflict? |
|------|-------|-----|-----------|--------|----------------------|
| 10k  | b0_3  | 0.10 | −0.148 | [−0.170, −0.127] | yes |
| 10k  | b0_3  | 0.20 | **−0.244** | [−0.272, −0.217] | yes |
| 10k  | b0_3  | 0.30 | −0.125 | [−0.168, −0.081] | yes |
| 150k | b0_3  | 0.20 | **−0.083** | [−0.122, −0.043] | yes (late-training persistence) |
| 200k | b0_3  | 0.20 | +0.090 | [+0.058, +0.119] | no (recovered) |

Useful work `cos × ratio` peaks at t≈0.7 (~0.32 magnitude at 200k) and is near-zero at low t. At t=0.05 the REPA gradient still has 39% the norm of the diffusion gradient with cos ≈ 0.06 — substantial gradient energy spent in a near-random direction. Per-batch cos ≈ aggregate cos within ~5pp → the conflict measurement is signal, not noise.

## 3. Method

### 3.1 The surgery (PCGrad, restricted to REPA's support)

REPA's auxiliary loss only reaches the projector head, embedders (x_embedder, t_embedder, y_embedder), and transformer blocks `[0, encoder_depth)`. With `encoder_depth=4` this is blocks 0–3. We refer to this set as **REPA's support S**.

Algorithm per minibatch:

```
Forward(model, x_t, t) → (model_output, zs_tilde)
L_diff = mean((model_output − v_target)²)           # denoising
L_repa = mean(-cos_sim(proj(zs_tilde), zs_teacher)) # alignment

# Two backward passes
g_d ← grad(L_diff, params)                          # full set
g_r ← grad(λ · L_repa, params)                      # nonzero only on S

# Restrict inner products to S
dot     = Σ_{p ∈ S} ⟨g_d[p], g_r[p]⟩
norm_sq = Σ_{p ∈ S} ‖g_d[p]‖²

# Sign of dot governs surgery
if dot < threshold:                                  # conflict regime
    α = dot / (norm_sq + ε)                          # negative scalar
    for p in S:
        g_r[p] ← g_r[p] − α · g_d[p]                 # removes anti-aligned component

# Combine
final_grad[p] ← g_d[p] + g_r[p]                      # for p ∈ S, just g_d outside S
```

Note: `λ` is already absorbed into `g_r` because we backward `λ · L_repa`, so the surgery acts on the scaled REPA gradient. The check `dot < 0` is sign-invariant under λ > 0 scaling, so the trigger semantics are preserved.

### 3.2 Modes (selected via `--sigma-mode`)

- **`off`**: vanilla REPA (single backward of `L_diff + λ L_repa`). Used as a self-consistency check that the new script reproduces the existing run.
- **`hard`**: project when `dot < 0` (the canonical PCGrad). **This is the primary experiment.**
- **`threshold`**: project when `cos < τ` for a configurable τ (e.g., τ = −0.05). Avoids triggering on noise.
- **`bloop`**: Bloop-style — keep an EMA of `g_d` and use it as the projection direction. More stable but adds state. *(Implemented but not the primary experiment.)*

### 3.3 Why two backward passes (not torch.autograd.grad)

Accelerate + DDP + GradScaler is the path of least magic. Doing `accelerator.backward(L_diff, retain_graph=True)` then `accelerator.backward(λ · L_repa)` gives us both DDP all-reduce and fp16 scaling for free. The two gradient sets land in `.grad` between the two backwards, so we clone after each. Both are scaled by the same GradScaler factor → cos, dot, α, and the projected combined gradient are all scale-invariant. Setting `p.grad ← combined` before `optimizer.step()` lets GradScaler's `unscale_` and `step` work normally.

### 3.4 Surgery logging

Every step, log to `runs/<exp_name>/surgery_stats.csv`:
`global_step, t_mean, t_min, t_max, dot, norm_sq_d, norm_sq_r, cos, alpha, projected (bool), loss_diff, loss_repa`

This gives us a **per-step record** of when surgery fired, by how much, and how the cos surface changes during training. The single most useful piece of evidence for the analysis.

## 4. Baseline and comparison protocol

**Baseline** = the existing run at `runs/celeba_sit_b2_repa_dinov2b_gpus4-7` (constant-λ REPA, FID@200k = 6.06, scores.csv complete).

**Treatment** = REPA-Σ-hard, 0→200k from scratch, identical hyperparameters and seed.

**Decisions**:
- **From scratch, not resume from 80k**. Reason: cleanest comparison. Mixing surgery into an already-trained model risks mid-training instability or unclear attribution. The 2-day extra wall-time is worth it.
- **Same `seed=0`, same data shuffling**. Direct paired comparison.
- **Output dir**: `runs/celeba_sit_b2_repa_sigma_hard_gpus4-7`.

## 5. Evaluation

1. **FID/KID curves**: `fid_eval_fast.py` on every 10k checkpoint of REPA-Σ. Overlay with baseline scores.csv.
2. **Gradient geometry probe**: re-run `probe_gradient_geometry_better.py` on REPA-Σ checkpoints at 10k, 50k, 100k, 150k, 200k. Compare cos surface to baseline's.
3. **Surgery telemetry analysis**: from `surgery_stats.csv`, compute (a) how often surgery fires (b) the magnitude of `α` (c) the t-distribution of fires (d) evolution over training.

## 6. Success criteria

| Outcome | Interpretation |
|---|---|
| REPA-Σ FID@200k < 6.06 (baseline) | Surgery improves over constant-λ REPA — main win |
| REPA-Σ FID curve dominates baseline at every checkpoint | Strong win, surgery accelerates convergence |
| FID matches baseline but cos surface is uniformly non-negative | Surgery worked geometrically; FID isn't sensitive enough to detect on CelebA at this scale |
| FID matches baseline and cos surface unchanged | Surgery rarely fires; conflict zone doesn't matter in practice |
| FID worse | Surgery destabilizes training; reduces effective REPA signal |

A null result is still publishable as a measurement contribution paired with the existing gradient-geometry paper.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| GradScaler interaction with two backwards | Test in 200-step dry run; check loss_diff and loss_repa match baseline at step 1 |
| DDP all-reduce semantics with retain_graph | The cleanest path is to call `accelerator.backward` twice with retain_graph=True on the first. DDP correctly all-reduces each backward independently. Test in dry run. |
| Surgery fires too often (noise) | Use `threshold` mode with τ ≈ −0.05 as fallback |
| Compute: 200k steps × 2 backwards ≈ 1.5x baseline cost | Acceptable; ~2 days on 4 GPUs |
| Bug in scope mask (operating on wrong params) | Build mask via the same logic as `probe_gradient_geometry.py` (which we know works) |
| Forgetting to `.detach().clone()` between backwards | Explicit; test in dry run |
| `optimizer.zero_grad()` between backwards but the first backward's grads being lost | The protocol is: clone, then zero, then second backward, then clone, then surgery, then set. Followed strictly. |

## 8. Falsification

Pre-commit to: if REPA-Σ FID@200k is more than 0.5 above baseline (i.e., > 6.56), the experiment has clearly failed (worse than baseline-without-REPA). At that point, document the failure and switch to Idea 2 (REPA-T) instead of continuing to debug.

## 9. Compute and timeline

- **Implementation + dry run**: 1 session (now)
- **Training**: ~36–48 hours on 4× A100 (estimated 0.75 s/step × 200k = 41.7 h plus overhead)
- **FID evaluation**: ~1 hour total (uses fid_eval_fast.py, 4 GPUs parallel)
- **Gradient geometry re-probe**: ~30 min
- **Analysis + writeup**: 1 session

Total: ~3 days end-to-end.

## 10. Deliverables (PR contents)

- `REPA/train_sigma.py` — the modified training script
- `scripts/train_repa_sigma_gpus4_7.sh` — launch script
- `experiments/repa_sigma/PLAN.md` — this file
- `experiments/repa_sigma/LOG.md` — running journal
- `experiments/repa_sigma/RESULTS.md` — final report
- `runs/celeba_sit_b2_repa_sigma_hard_gpus4-7/scores.csv` — FID/KID curve
- `runs/celeba_sit_b2_repa_sigma_hard_gpus4-7/surgery_stats.csv` — surgery telemetry
- `reports/repa_sigma_grad_geometry/` — re-probed gradient geometry
- comparison plots (FID curves, cos surface deltas)

User opens PR after reviewing.
