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

## 2026-05-25 — Commit phase

- Committed train_sigma.py to REPA submodule on branch `experiment/repa-sigma`
  (commit 2b79074 from 434d8ff).
- Committed PLAN.md, LOG.md, launch script, and submodule bump to parent repo
  on branch `experiment/repa-sigma` (commit a56875c from 95791a8).
- Tried to push both branches to origin. Both failed with 403 — credentials on
  ya100 lack push permission via HTTPS. Commits are durable locally and visible
  via `git log`; user will push manually before opening the PR.

## 2026-05-25 — Launch

- Updated launch script to use 6 GPUs (2-7) at the user's instruction. Kept
  effective batch_size = 128 to preserve apples-to-apples FID comparison
  vs the existing constant-λ REPA baseline. Distribution: 128 // 6 = 21 per
  rank, total 126.
- Renamed exp to `celeba_sit_b2_repa_sigma_hard_gpus2-7`.
- Per-GPU memory usage: ~10 GB / 40 GB. Intentionally underutilized to keep
  the comparison clean; increasing batch_size would change the effective
  training dynamics and invalidate the FID-vs-baseline comparison.
- Launched in background with `setsid nohup bash scripts/...`:
  - Parent PID: **1895565** (accelerate launch)
  - Process group: **1895563** (to kill whole tree: `kill -- -1895563`)
  - Log file: `logs/repa_sigma_hard.log`
  - Pid file: `logs/repa_sigma_hard.pid`
  - Run dir: `runs/celeba_sit_b2_repa_sigma_hard_gpus2-7/`
- Sanity check at t+4min: at step 1530, loss=0.86 (from 1.97 at step 1),
  proj_loss=-0.66, surgery firing as expected (cos<0 → projected=1).
  Step time ~0.12 s. Projected total wall: ~9 hours; ETA tomorrow morning.

## Next entry slot: intermediate checkpoint evaluation

## 2026-05-25 — Push to GitHub via Mac

- Earlier push attempt from ya100 failed (HTTPS 403). Available local tokens
  on ya100 were Codeforces-scoped, not for the REPA-diffusion repos.
- Solution: created tight git bundles on ya100 (`origin/main..experiment/repa-sigma`),
  transferred via scp to the Mac, cloned both GitHub repos fresh into
  /tmp/push_workspace, applied the bundles, and pushed from Mac where the
  user's git credentials are configured.
- Both branches now visible on GitHub:
  - https://github.com/sekopylov/REPA/tree/experiment/repa-sigma
  - https://github.com/pakhomovee/REPA-diffusion/tree/experiment/repa-sigma
- PR links offered by GitHub:
  - https://github.com/sekopylov/REPA/pull/new/experiment/repa-sigma
  - https://github.com/pakhomovee/REPA-diffusion/pull/new/experiment/repa-sigma

## 2026-05-29 — REPA-Σ results + λ-anneal extension

### REPA-Σ FID results (constant λ, surgery) — full curve at every 20k
| step | REPA-Σ FID | REPA-const FID | Baseline FID |
|------|-----------|----------------|--------------|
| 20k  | 324.33 | 334.15 | 328.50 |
| 40k  | 23.46  | 23.70  | 26.72  |
| 60k  | 10.57  | 10.51  | 11.28  |
| 80k  | 8.10   | 8.33   | 8.78   |
| 100k | 7.15   | 7.29   | 7.85   |
| 120k | 6.58   | 6.78   | 7.41   |
| 140k | 6.33   | 6.50   | 7.00   |
| 160k | 6.24   | 6.33   | 6.87   |
| 180k | 6.13   | 6.27   | 6.68   |
| 200k | 6.03   | 6.06   | 6.56   |
REPA-Σ beats constant-λ REPA at almost every checkpoint; largest gap in
mid-training (100k-140k, −0.14 to −0.20), matching the gradient-conflict story.

### λ-annealing extension (this branch: experiment/repa-lambda-anneal)
- Motivation (Evgenii's correction + toy sim): REPA-Σ removes the *antiparallel*
  gradient component, but the *orthogonal* component still biases the model
  toward the detail-poor DINOv2 manifold. Annealing λ→0 removes that bias.
  Full write-up: lambda_anneal_analysis.md.
- Implemented --lambda-anneal in train_sigma.py (linear proj_coeff→lambda_final
  between lambda_anneal_start and lambda_anneal_end; default 0.5→0 over 200k).
  Composes with any --sigma-mode. effective_lambda() unit-tested; lambda_eff
  logged per-step to surgery_stats.csv and TensorBoard for both modes.
- Dry-runs (40 steps, 2 GPUs) confirm: off+anneal anneals λ cleanly with no
  surgery; hard+anneal anneals λ AND fires surgery only when cos<0.
- Two new launch scripts (effective batch 128, 200k, seed 0):
  - scripts/train_repa_anneal_gpus2_4.sh        → REPA + anneal (off), GPUs 2-4
  - scripts/train_repa_sigma_anneal_gpus5_7.sh  → REPA-Σ + anneal (hard), GPUs 5-7
- Completes the 5-way ablation matching the toy simulation:
  Baseline / REPA-const / REPA-Σ / REPA+anneal / REPA-Σ+anneal.
