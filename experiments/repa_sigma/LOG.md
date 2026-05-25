# REPA-Σ — Running log

Append-only journal. Newest entries at the bottom.

---

## 2026-05-25 — Setup phase

- Created branches `experiment/repa-sigma` in both parent repo and REPA submodule.
- Wrote `experiments/repa_sigma/PLAN.md` capturing the full design, empirical anchors, success criteria, and risk register.
- Verified existing baseline: `runs/celeba_sit_b2_repa_dinov2b_gpus4-7` with FID@200k = 6.06, all 20 checkpoints (10k → 200k) present, scores.csv complete.
- Confirmed GPUs 0–7 are all idle, all A100-40GB.
- Will train on GPUs 4–7 as instructed; remaining GPUs free for parallel eval/probe.

## Next entry slot: implementation start

## 2026-05-25 — Implementation phase

- Wrote `REPA/train_sigma.py` (~800 lines). Surgery branch uses two
  `torch.autograd.grad` calls + manual all-reduce + manual GradScaler ops to
  sidestep DDP's auto-reducer; the reducer errors on undefined gradients for
  params outside REPA's support (blocks ≥ encoder_depth, final_layer).
- Wrote `scripts/train_repa_sigma_gpus4_7.sh`. GPUs 4-7, hyperparameters
  identical to the existing constant-λ baseline (proj_coeff=0.5, encoder_depth=4,
  seed=0, batch 128, fp16, 200k steps).
- First DDP-based dry run with `accelerator.backward` × 2 failed at step 2
  with 'Encountered gradient which is undefined, but still allreduced by DDP
  reducer.' Refactored surgery branch to use `torch.autograd.grad` + manual
  all-reduce.
- Second dry run failed: 'One of the differentiated Tensors does not require
  grad' — fixed by filtering `named_parameters()` to `p.requires_grad` only.
- Third dry run (30 steps, sigma_mode=hard): SUCCESS. Surgery fires when cos<0
  (e.g., step 2: cos=-0.034, alpha=-0.189; step 14: cos=-0.491, alpha=-0.148).
  No crashes. Loss decreased 1.97 → 1.54.
- Fourth dry run (10 steps, sigma_mode=off): SUCCESS. Loss values byte-identical
  to the sigma_mode=hard run for the first 10 steps — both are inside the
  GradScaler warmup phase where steps are skipped due to fp16 overflow, so
  surgery has no effect on the model state yet. After warmup, the two modes
  would diverge.
- Steady-state step time: ~0.13-0.21 s (hard mode) on 4× A100. Projected
  200k-step training time: ~10 hours.
- Surgery telemetry CSV (`runs/<exp>/surgery_stats.csv`) confirmed working,
  one row per step with dot, norm_sq_d, norm_sq_r, cos, alpha, projected.

## Next entry slot: launching full training
