# λ-annealing for REPA — analysis & proposal

**Status**: proposal (not yet implemented)
**Context**: follow-up to REPA-Σ (per-batch gradient surgery). Triggered by
Evgenii Pakhomov's correction in chat + a toy simulation he ran.
**Date**: 2026-05-27

---

## 1. What is proposed (and where the REPA-Σ argument falls short)

The REPA-Σ argument was: *"surgery removes the harmful part because
cos(g_diff, g_repa) < 0"*. Evgenii's correction:

> "Zeroing the antiparallel part is OK, **but the orthogonal part can also
> interfere with learning**, so we want REPA turned off at the end."

This is correct and important. Decompose the REPA gradient relative to the
denoising gradient:

```
g_repa = g_repa^∥   +   g_repa^⊥
         (along g_diff)   (orthogonal to g_diff)
```

- **REPA-Σ** removes only the **antiparallel** component `g_repa^∥` when it
  points against `g_diff` (cos < 0).
- The **orthogonal** component `g_repa^⊥` is *always* present and continuously
  pulls the model's representation toward the teacher's (DINOv2) manifold.
- DINOv2 is invariance-trained → it discards fine detail. So `g_repa^⊥` biases
  the final solution toward "semantically correct but detail-poor" — a
  **systematic bias** that surgery does not touch.

The only way to remove this bias asymptotically is to **anneal λ → 0 toward the
end of training**.

## 2. What the toy simulation shows (right panel is the key one)

The toy model: there is a "true detail-gain" b = 0.69; REPA biases it toward
0.42 (the teacher bias).

| Curve | Final detail-gain | Conclusion |
|---|---|---|
| Diffusion only (black) | **0.69** ✓ | recovers detail correctly but slowly |
| REPA const-λ (red) | **0.42** ✗ | stuck at teacher bias — never recovers detail |
| REPA-Σ (teal) | **~0.46** ✗ | surgery slightly better, but **still strongly biased** |
| REPA λ-anneal (gray) | **0.69** ✓ | annealing λ removes the bias, reaches truth |
| REPA-Σ + anneal (blue) | **0.69** ✓ | reaches truth **faster** than gray — best of both |

**Left panel (semantic MSE)** shows the flip side: on semantics ALL REPA
variants are good and nearly indistinguishable. So **the damage is invisible in
the semantic metric — it hides in the detail-gain**. This explains why in FID we
see that REPA helps but possibly leaves potential on the table for fine detail.

Key takeaway of the simulation: **REPA-Σ removes the directional conflict
(antiparallel), but does NOT remove the bias (orthogonal). Only λ→0 anneal
removes the bias.**

## 3. Consistency with what we already know

This is not an isolated idea — it converges from three directions:

1. **HASTE (NeurIPS'25)** — "REPA Works Until It Doesn't" — does exactly this,
   but with a **hard cutoff** (λ=const until step τ, then λ=0). A linear anneal
   is the smooth version of the same thing.
2. **Our gradient-geometry data** — cos(g_diff, g_repa) at high t drops from
   0.67 (10k) to 0.47 (200k). REPA becomes less cooperative over time —
   consistent with "bias accumulates".
3. **The invariance-precision hypothesis** (earlier discussion) — DINOv2 is
   invariant to detail, which is exactly the orthogonal bias pulling toward a
   detail-poor solution.

Everything points the same direction.

## 4. Does it make sense for OUR training (200k CelebA)?

**Yes, but with an honest caveat about the magnitude of the effect.**

Arguments FOR:
- Conceptually correct, confirmed by the simulation + HASTE.
- Cheap: only a λ-schedule change, reuses the REPA-Σ infrastructure.
- Completes the ablation — a full 5-method table like the simulation
  ("for uniformity", as Evgenii requests).
- Even if FID is neutral, a gradient-geometry re-probe will show whether the
  bias was removed.

Caveat (worth understanding up front):
- Our FID curves at 200k CelebA **do not show REPA degradation at the end**
  (6.06 vs 6.56 baseline — REPA still wins at 200k). We have **not entered** the
  "REPA hurts" regime that HASTE observes at ~4M iterations on ImageNet.
- Evgenii himself notes: **"this only gives a mega boost at early stages"**. By
  our final step (200k), the gain is probably **modest** (my prediction:
  0.1-0.3 FID), not dramatic.
- But the direction is right, and the **REPA-Σ + anneal** combination should be
  the best point.

## 5. How exactly to implement

Two new runs are needed (both from scratch, 200k, seed=0, all other
hyperparameters as ours):

| Run | sigma-mode | λ schedule | Simulation curve |
|---|---|---|---|
| **REPA + anneal** | `off` | 0.5 → 0 linear | gray "+REPA lambda-anneal" |
| **REPA-Σ + anneal** | `hard` | 0.5 → 0 linear | blue "+REPA-Σ + anneal" (beats all) |

This yields the full 5-method table (with our already-finished Baseline,
REPA-const, REPA-Σ).

**Change in `train_sigma.py`** — add a λ(s) schedule:

```python
# New arguments
parser.add_argument("--lambda-anneal", action="store_true",
                    help="Linearly anneal proj_coeff -> lambda_final over training")
parser.add_argument("--lambda-anneal-start", type=int, default=0,
                    help="Step at which annealing begins (constant proj_coeff before)")
parser.add_argument("--lambda-anneal-end", type=int, default=None,
                    help="Step at which lambda reaches lambda_final (default: max_train_steps)")
parser.add_argument("--lambda-final", type=float, default=0.0)

# In the training loop
def effective_lambda(step):
    if not args.lambda_anneal:
        return args.proj_coeff
    end = args.lambda_anneal_end or args.max_train_steps
    if step <= args.lambda_anneal_start:
        return args.proj_coeff
    if step >= end:
        return args.lambda_final
    frac = (step - args.lambda_anneal_start) / (end - args.lambda_anneal_start)
    return args.proj_coeff + frac * (args.lambda_final - args.proj_coeff)

lam = effective_lambda(global_step)
```

- In the **`off`** path (REPA + anneal): `loss = loss_mean + proj_loss_mean * lam`
- In the **surgery** path (REPA-Σ + anneal): backward `proj_loss_mean * lam`
  instead of `proj_loss_mean * proj_coeff`, then surgery and assembly as usual.
  I.e. the anneal is applied to the **already-cleaned** g_repa.

**Schedule**: linear 0.5 → 0 over all 200k (as Evgenii requested — "linearly").
I would make `lambda_anneal_start` configurable in case we want "hold λ until
100k, then anneal", but by default linear from the start, like in the simulation.

**Design nuance to discuss**: anneal to **exactly 0 at 200k**, or to 0 earlier
(e.g. by 160k) with pure denoising in the tail? In the toy simulation λ reaches
0 at the end. I would replicate that (linear to 0 at 200k) for direct
correspondence.

## 6. Compute

Two runs of ~9 hours each. Can run in parallel: one on GPUs 0-3, the other on
4-7 (if willing to occupy all 8). Or sequentially on 4-7 (~18 hours), leaving
0-3 free.

---

## Recommendation

Yes, worth doing — it completes the ablation, tests a theoretically-grounded
mechanism, and REPA-Σ + anneal is the best point in the simulation. But keep FID
expectations on our 200k CelebA modest (the main boost is early, and we already
win at the final step).

Options:
- **(a)** implement `--lambda-anneal` in `train_sigma.py` and launch **both**
  runs (REPA+anneal and REPA-Σ+anneal),
- **(b)** implement first + show the diff, launch on command,
- **(c)** launch only REPA-Σ+anneal (most promising per the simulation), without
  a separate REPA+anneal run.
