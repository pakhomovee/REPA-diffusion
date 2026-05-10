#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

${PYTHON:-python} - <<'PY'
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:
    raise SystemExit(f"Could not import TensorBoard EventAccumulator: {exc}")

train_images = 9469
effective_batch = 256
target_steps = 30000

runs = [
    ("baseline", "imagenette_sit_b2_baseline_30k_gpus0-3", "logs/baseline_30k.pid"),
    ("repa", "imagenette_sit_b2_repa_dinov2b_30k_gpus4-7", "logs/repa_30k.pid"),
]

for label, exp_name, pid_path in runs:
    event_dir = Path("runs") / exp_name / "logs" / "REPA"
    pid = Path(pid_path).read_text().strip() if Path(pid_path).exists() else "?"

    step = 0
    loss = None
    proj = None
    if event_dir.exists():
        try:
            ea = EventAccumulator(str(event_dir))
            ea.Reload()
            losses = ea.Scalars("loss") if "loss" in ea.Tags().get("scalars", []) else []
            projs = ea.Scalars("proj_loss") if "proj_loss" in ea.Tags().get("scalars", []) else []
            if losses:
                step = losses[-1].step
                loss = losses[-1].value
            if projs:
                proj = projs[-1].value
        except Exception:
            pass

    epoch = step * effective_batch / train_images
    pct = 100 * step / target_steps
    loss_s = "n/a" if loss is None else f"{loss:.4f}"
    proj_s = "n/a" if proj is None else f"{proj:.4f}"
    print(
        f"{label:8s} pid={pid:>8s} step={step:5d}/{target_steps} "
        f"({pct:5.1f}%) epoch~{epoch:7.1f} loss={loss_s} proj={proj_s}"
    )
PY
