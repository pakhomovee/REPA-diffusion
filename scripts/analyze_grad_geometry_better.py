#!/usr/bin/env python3
"""
analyze_grad_geometry_better.py — statistically rigorous plots from the
better-grad-geometry CSVs.

Inputs:
    --csv-dir   directory containing measurements_raw.csv and measurements_ema.csv

For each (scope, weights) cell we report:
    * per-microbatch mean cosine, with 95 % bootstrap CI
    * "aggregate gradient" cosine — cos of summed gradients across microbatches,
      with paired bootstrap CI:
          cos_agg = Σ⟨g1,g2⟩ / √( Σ‖g1‖² · Σ‖g2‖² )
    * per-microbatch mean ratio, with 95 % bootstrap CI (log-transformed)
    * "useful work" = mean(cos × ratio) with 95 % bootstrap CI

Outputs (PNG, dpi=150):
    cos_vs_t__{scope}__{weights}.png         per-batch cos with 95 % CI bands
    cos_agg_vs_t__{scope}__{weights}.png     aggregate-grad cos with 95 % CI
    ratio_vs_t__{scope}__{weights}.png       ratio with 95 % CI (log y)
    useful_work_vs_t__{scope}__{weights}.png cos × ratio with 95 % CI
    cos_overlay__{scope}__{weights}.png      both perspectives overlaid (Jensen gap)
    summary_table_{weights}.csv              numeric summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_BOOT = 10_000
CI_LO, CI_HI = 2.5, 97.5  # 95 % percentile CI


# ── Bootstrap utilities ──────────────────────────────────────────────────────
def bootstrap_mean_ci(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of x."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, CI_LO)), float(np.percentile(means, CI_HI))


def bootstrap_log_mean_ci(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """CI on the geometric mean (log-bootstrap → exp). For strictly positive x."""
    x = np.asarray(x, dtype=float)
    if (x <= 0).any():
        # Fall back to plain mean if any non-positive
        return bootstrap_mean_ci(x, n_boot)
    lx = np.log(x)
    n = len(lx)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = lx[idx].mean(axis=1)
    return (
        float(np.exp(lx.mean())),
        float(np.exp(np.percentile(means, CI_LO))),
        float(np.exp(np.percentile(means, CI_HI))),
    )


def bootstrap_aggregate_cos_ci(
    dot: np.ndarray, gd_sq: np.ndarray, gr_sq: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    """Paired bootstrap CI for cos_agg = sum(dot) / sqrt(sum(gd_sq) * sum(gr_sq))."""
    dot = np.asarray(dot, dtype=float)
    gd_sq = np.asarray(gd_sq, dtype=float)
    gr_sq = np.asarray(gr_sq, dtype=float)
    n = len(dot)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    def cos_from_sums(d, a, b):
        denom = np.sqrt(a * b)
        return d / denom if denom > 0 else 0.0

    point = cos_from_sums(dot.sum(), gd_sq.sum(), gr_sq.sum())
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_cos = np.array([
        cos_from_sums(dot[i].sum(), gd_sq[i].sum(), gr_sq[i].sum()) for i in idx
    ])
    return float(point), float(np.percentile(boot_cos, CI_LO)), float(np.percentile(boot_cos, CI_HI))


# ── Plotting helpers ─────────────────────────────────────────────────────────
def _ckpt_colors(steps):
    cmap = plt.get_cmap("viridis")
    if len(steps) == 1:
        return {steps[0]: cmap(0.5)}
    return {s: cmap(i / (len(steps) - 1)) for i, s in enumerate(sorted(steps))}


def _label_for_step(step: int) -> str:
    return f"{step // 1000}k"


def _style_ax(ax, title, xlabel, ylabel, show_zero=False):
    if show_zero:
        ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(title="ckpt step", loc="best", fontsize=9, framealpha=0.85)


def plot_cos(df_sub, out_path, weights, scope, kind):
    """kind ∈ {'per_batch', 'aggregate'} — chooses which summary to plot."""
    steps = sorted(df_sub["ckpt_step"].unique())
    colors = _ckpt_colors(steps)
    fig, ax = plt.subplots(figsize=(8, 5))

    for s in steps:
        sub_s = df_sub[df_sub["ckpt_step"] == s]
        ts = sorted(sub_s["t"].unique())
        means, lo, hi = [], [], []
        for t in ts:
            cell = sub_s[sub_s["t"] == t]
            if kind == "per_batch":
                m, lci, hci = bootstrap_mean_ci(cell["cos_sim"].values)
            else:
                m, lci, hci = bootstrap_aggregate_cos_ci(
                    cell["dot_product"].values,
                    cell["g_diff_sq"].values,
                    cell["g_repa_sq"].values,
                )
            means.append(m); lo.append(lci); hi.append(hci)
        ax.plot(ts, means, marker="o", color=colors[s], label=_label_for_step(s), lw=1.8, ms=4)
        ax.fill_between(ts, lo, hi, alpha=0.18, color=colors[s])

    title = (
        f"Gradient compatibility vs t  ({scope}, weights={weights})\n"
        f"{'per-microbatch mean cos' if kind == 'per_batch' else 'aggregate-gradient cos'} "
        f"with 95% bootstrap CI"
    )
    _style_ax(
        ax, title, "Diffusion timestep t",
        r"cos($\nabla L_\mathrm{diff}$, $\nabla \lambda L_\mathrm{REPA}$)",
        show_zero=True,
    )
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ratio(df_sub, out_path, weights, scope):
    steps = sorted(df_sub["ckpt_step"].unique())
    colors = _ckpt_colors(steps)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in steps:
        sub_s = df_sub[df_sub["ckpt_step"] == s]
        ts = sorted(sub_s["t"].unique())
        means, lo, hi = [], [], []
        for t in ts:
            cell = sub_s[sub_s["t"] == t]
            m, lci, hci = bootstrap_log_mean_ci(cell["ratio"].values)
            means.append(m); lo.append(lci); hi.append(hci)
        ax.plot(ts, means, marker="o", color=colors[s], label=_label_for_step(s), lw=1.8, ms=4)
        ax.fill_between(ts, lo, hi, alpha=0.18, color=colors[s])
    ax.set_yscale("log")
    _style_ax(
        ax, f"REPA/diff gradient norm ratio vs t  ({scope}, weights={weights})\n"
            "geometric mean with 95% bootstrap CI",
        "Diffusion timestep t",
        r"$\Vert\nabla \lambda L_\mathrm{REPA}\Vert / \Vert\nabla L_\mathrm{diff}\Vert$",
    )
    ax.grid(True, ls="--", alpha=0.35, which="both")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_useful_work(df_sub, out_path, weights, scope):
    """useful_work = cos × ratio per microbatch, then averaged with bootstrap CI."""
    df_sub = df_sub.copy()
    df_sub["useful_work"] = df_sub["cos_sim"] * df_sub["ratio"]
    steps = sorted(df_sub["ckpt_step"].unique())
    colors = _ckpt_colors(steps)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in steps:
        sub_s = df_sub[df_sub["ckpt_step"] == s]
        ts = sorted(sub_s["t"].unique())
        means, lo, hi = [], [], []
        for t in ts:
            cell = sub_s[sub_s["t"] == t]
            m, lci, hci = bootstrap_mean_ci(cell["useful_work"].values)
            means.append(m); lo.append(lci); hi.append(hci)
        ax.plot(ts, means, marker="o", color=colors[s], label=_label_for_step(s), lw=1.8, ms=4)
        ax.fill_between(ts, lo, hi, alpha=0.18, color=colors[s])
    _style_ax(
        ax,
        f"'Useful work' cos × ratio vs t  ({scope}, weights={weights})\n"
        "per-batch mean with 95% bootstrap CI",
        "Diffusion timestep t",
        r"cos · $\Vert\nabla L_\mathrm{REPA}\Vert / \Vert\nabla L_\mathrm{diff}\Vert$",
        show_zero=True,
    )
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_overlay(df_sub, out_path, weights, scope):
    """Overlay per-batch mean cos and aggregate-grad cos to show Jensen gap."""
    steps = sorted(df_sub["ckpt_step"].unique())
    colors = _ckpt_colors(steps)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in steps:
        sub_s = df_sub[df_sub["ckpt_step"] == s]
        ts = sorted(sub_s["t"].unique())
        pb_means, agg_means = [], []
        for t in ts:
            cell = sub_s[sub_s["t"] == t]
            pb, _, _ = bootstrap_mean_ci(cell["cos_sim"].values)
            ag, _, _ = bootstrap_aggregate_cos_ci(
                cell["dot_product"].values, cell["g_diff_sq"].values, cell["g_repa_sq"].values
            )
            pb_means.append(pb); agg_means.append(ag)
        ax.plot(ts, pb_means, marker="o", ls="-", color=colors[s], lw=1.5, ms=4,
                label=f"{_label_for_step(s)} per-batch")
        ax.plot(ts, agg_means, marker="s", ls="--", color=colors[s], lw=1.5, ms=4,
                label=f"{_label_for_step(s)} aggregate")
    _style_ax(
        ax,
        f"Per-batch vs aggregate-gradient cos  ({scope}, weights={weights})\n"
        "gap reveals minibatch noise in gradient direction",
        "Diffusion timestep t",
        r"cos($\nabla L_\mathrm{diff}$, $\nabla \lambda L_\mathrm{REPA}$)",
        show_zero=True,
    )
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def write_summary(df_sub, out_path, weights, scope):
    """Write per-cell numeric summary to CSV."""
    rows = []
    for (s, t), cell in df_sub.groupby(["ckpt_step", "t"]):
        cos_pb_m, cos_pb_lo, cos_pb_hi = bootstrap_mean_ci(cell["cos_sim"].values)
        cos_ag_m, cos_ag_lo, cos_ag_hi = bootstrap_aggregate_cos_ci(
            cell["dot_product"].values, cell["g_diff_sq"].values, cell["g_repa_sq"].values
        )
        rat_m, rat_lo, rat_hi = bootstrap_log_mean_ci(cell["ratio"].values)
        uw = (cell["cos_sim"] * cell["ratio"]).values
        uw_m, uw_lo, uw_hi = bootstrap_mean_ci(uw)
        rows.append({
            "weights": weights, "scope": scope, "ckpt_step": int(s), "t": float(t),
            "n_microbatches": len(cell),
            "cos_per_batch_mean": cos_pb_m, "cos_per_batch_lo": cos_pb_lo, "cos_per_batch_hi": cos_pb_hi,
            "cos_aggregate":      cos_ag_m, "cos_aggregate_lo": cos_ag_lo, "cos_aggregate_hi": cos_ag_hi,
            "ratio_geom_mean":    rat_m,    "ratio_lo":         rat_lo,    "ratio_hi":         rat_hi,
            "useful_work_mean":   uw_m,     "useful_work_lo":   uw_lo,     "useful_work_hi":   uw_hi,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Default: same as csv-dir")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for weights in ("raw", "ema"):
        csv_path = csv_dir / f"measurements_{weights}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} missing")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["weights"] == weights]
        if df.empty:
            print(f"[skip] no rows for {weights}")
            continue

        summary_rows = []
        for scope in sorted(df["scope"].unique()):
            df_sub = df[df["scope"] == scope]
            print(f"[plot] weights={weights} scope={scope} "
                  f"steps={sorted(df_sub['ckpt_step'].unique())} "
                  f"t={sorted(df_sub['t'].unique())} "
                  f"n_mb_per_cell={df_sub.groupby(['ckpt_step', 't']).size().mean():.0f}")

            plot_cos(df_sub, out_dir / f"cos_vs_t__{scope}__{weights}.png",
                     weights, scope, kind="per_batch")
            plot_cos(df_sub, out_dir / f"cos_agg_vs_t__{scope}__{weights}.png",
                     weights, scope, kind="aggregate")
            plot_ratio(df_sub, out_dir / f"ratio_vs_t__{scope}__{weights}.png",
                       weights, scope)
            plot_useful_work(df_sub, out_dir / f"useful_work_vs_t__{scope}__{weights}.png",
                             weights, scope)
            plot_overlay(df_sub, out_dir / f"cos_overlay__{scope}__{weights}.png",
                         weights, scope)
            summary_rows.append(df_sub.assign(_summary_scope=scope))

        # Single combined summary table per weights mode
        combined = pd.concat(summary_rows, ignore_index=True)
        per_cell = []
        for scope in sorted(df["scope"].unique()):
            sub = combined[combined["_summary_scope"] == scope]
            sub = sub.drop(columns=["_summary_scope"])
            tmp_out = out_dir / f"summary_{weights}_{scope}.csv"
            write_summary(sub, tmp_out, weights, scope)
            per_cell.append(pd.read_csv(tmp_out))
        if per_cell:
            pd.concat(per_cell, ignore_index=True).to_csv(
                out_dir / f"summary_{weights}.csv", index=False
            )
            for scope in sorted(df["scope"].unique()):
                (out_dir / f"summary_{weights}_{scope}.csv").unlink(missing_ok=True)
        print(f"[done] weights={weights} → {out_dir / f'summary_{weights}.csv'}")


if __name__ == "__main__":
    main()
