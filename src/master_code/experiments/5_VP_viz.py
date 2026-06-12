"""Per-run runtime analysis of the Victoria Park benchmark (Global vs Steiner).

``5_VP_run.py`` runs the same config five times per method so the random
wall-clock spikes in the covariance-recovery step can be characterised.  This
script uses a matched 5-run set (runs 1-5) for both methods and:

  * Tables: computes the summary statistics *within each run* (totals, per-step
    max / mean / median / P95, steps-above-215 ms, share-above), then reports the
    *median across runs* with the *min-max range* in each cell.
  * Figures: regenerates the landmarks-and-timing and timing-vs-landmarks/cliques
    plots from the per-step median across runs (spike-robust time series); the
    per-step timing plot stacks the *mean* component (additive -> mean total) and
    overlays a mean-to-max-across-runs ribbon whose peak is the worst-case step.

Run as a script path (the module name starts with a digit, so ``-m`` won't work)::

    python src/master_code/experiments/5_VP_viz.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from master_code.logger import SlamLogger
from master_code.paths import RUNS_ROOT
from master_code.plotting.plotting_funcs import (
    plot_landmarks_and_timing,
    plot_timing_vs_landmarks,
)
from master_code.plotting.thesis_style import apply_thesis_style, save_figure

OLD_BASE = RUNS_ROOT / "experiments" / "VP_BENCHMARKING" / "OLD"  # Global covariance
NEW_BASE = RUNS_ROOT / "experiments" / "VP_BENCHMARKING" / "NEW"  # Steiner tree
RUNS = [1, 2, 3, 4, 5]  # matched 5-run set for both methods (NEW run_6 ignored)
DEADLINE_MS = 215.0  # time between Victoria Park lidar scans -> real-time deadline


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def latest_experiment_dir(base: Path) -> Path:
    """Return the newest timestamped experiment dir under ``base``."""
    candidates = sorted(d for d in base.iterdir() if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No experiment dirs under {base.resolve()}")
    return candidates[-1]


def run_dirs(base: Path) -> list[Path]:
    """Return the ``run_1 .. run_5`` dirs of the latest experiment under ``base``."""
    experiment_dir = latest_experiment_dir(base)
    dirs = [experiment_dir / f"run_{r}" for r in RUNS]
    missing = [d for d in dirs if not (d / "steps.npz").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing steps.npz in: " + ", ".join(str(d.resolve()) for d in missing)
        )
    return dirs


# --------------------------------------------------------------------------- #
# Per-run metrics -> across-run median + range
# --------------------------------------------------------------------------- #
def run_metrics(steps: dict[str, np.ndarray]) -> dict[str, float]:
    """All scalar runtime metrics for a single run."""
    step = np.nan_to_num(np.asarray(steps["duration_step"], dtype=float))
    cov = np.nan_to_num(np.asarray(steps["duration_covariance_extraction"], dtype=float))
    assoc = np.nan_to_num(np.asarray(steps["duration_association"], dtype=float))
    opt = np.nan_to_num(np.asarray(steps["duration_optimization"], dtype=float))
    other = np.maximum(step - (cov + assoc + opt), 0.0)

    step_ms = 1000.0 * step
    cov_ms = 1000.0 * cov
    n = len(step)
    n_above = int(np.sum(step_ms > DEADLINE_MS))
    total = float(step.sum())

    return {
        # cumulative totals [s]
        "total": total,
        "cov": float(cov.sum()),
        "assoc": float(assoc.sum()),
        "opt": float(opt.sum()),
        "other": float(other.sum()),
        "cov_share": 100.0 * cov.sum() / total if total else 0.0,
        # covariance recovery, per step [ms]
        "cov_max": float(cov_ms.max()),
        "cov_mean": float(cov_ms.mean()),
        "cov_median": float(np.median(cov_ms)),
        "cov_p95": float(np.percentile(cov_ms, 95)),
        # total pipeline, per step [ms]
        "step_max": float(step_ms.max()),
        "step_mean": float(step_ms.mean()),
        "step_median": float(np.median(step_ms)),
        "step_p95": float(np.percentile(step_ms, 95)),
        # deadline
        "n_above": float(n_above),
        "share_above": 100.0 * n_above / n if n else 0.0,
    }


def aggregate_metrics(run_dirs: list[Path]) -> dict[str, tuple[float, float, float]]:
    """For each metric, the (median, mean, min, max) of its per-run values."""
    per_run = [run_metrics(SlamLogger.load_steps(d)) for d in run_dirs]
    return {
        key: (
            float(np.median([m[key] for m in per_run])),
            float(np.min([m[key] for m in per_run])),
            float(np.max([m[key] for m in per_run])),
        )
        for key in per_run[0]
    }


def improvement(old: tuple[float, ...], new: tuple[float, ...]) -> str:
    """Global / Steiner speed-up factor of the medians, e.g. ``8.5x``."""
    return f"{old[0] / new[0]:.1f}x" if new[0] else "--"


# --------------------------------------------------------------------------- #
# Tables (plain aligned text: median [min-max] per cell)
# --------------------------------------------------------------------------- #
def cell(stat: tuple[float, float, float], fmt: str = "{:.1f}") -> str:
    """Format one metric as ``median [min-max]`` (or just ``median`` if constant)."""
    med, lo, hi = stat
    if lo == hi:
        return fmt.format(med)
    return f"{fmt.format(med)} [{fmt.format(lo)}-{fmt.format(hi)}]"


def _row(metric: str, old: str, new: str, imp: str) -> str:
    return f"  {metric:<38}{old:>26}{new:>26}{imp:>13}"


_DIVIDER = "  " + "-" * 101


def print_cumulative_table(o: dict, n: dict) -> None:
    print("\nCumulative runtime comparison (per-run totals; median [min-max] over runs 1-5)")
    print(_row("Metric", "Global covariance", "Steiner tree", "Improvement"))
    print(_DIVIDER)
    print(_row("Total pipeline runtime [s]", cell(o["total"]), cell(n["total"]),
               improvement(o["total"], n["total"])))
    print(_row("Covariance recovery runtime [s]", cell(o["cov"]), cell(n["cov"]),
               improvement(o["cov"], n["cov"])))
    print(_row("Covariance share of total runtime [%]", cell(o["cov_share"]),
               cell(n["cov_share"]), "--"))
    print(_row("Association runtime [s]", cell(o["assoc"]), cell(n["assoc"]),
               improvement(o["assoc"], n["assoc"])))
    print(_row("Optimization runtime [s]", cell(o["opt"]), cell(n["opt"]),
               improvement(o["opt"], n["opt"])))
    print(_row("Other runtime [s]", cell(o["other"]), cell(n["other"]),
               improvement(o["other"], n["other"])))


def print_per_step_table(o: dict, n: dict) -> None:
    print(f"\nPer-step runtime comparison (per-run stats; median [min-max] over runs 1-5, "
          f"deadline {DEADLINE_MS:.0f} ms)")
    print(_row("Metric", "Global covariance", "Steiner tree", "Improvement"))
    print(_DIVIDER)
    print("  Covariance recovery, per step")
    for label, key in [("  Maximum [ms]", "cov_max"), ("  Mean [ms]", "cov_mean"),
                       ("  Median [ms]", "cov_median"), ("  95th percentile [ms]", "cov_p95")]:
        print(_row(label, cell(o[key]), cell(n[key]), improvement(o[key], n[key])))
    print("  Total pipeline, per step")
    for label, key in [("  Maximum [ms]", "step_max"), ("  Mean [ms]", "step_mean"),
                       ("  Median [ms]", "step_median"), ("  95th percentile [ms]", "step_p95")]:
        print(_row(label, cell(o[key]), cell(n[key]), improvement(o[key], n[key])))
    print(_DIVIDER)
    print(_row("Real-time requirement [ms]", f"{DEADLINE_MS:.0f}", f"{DEADLINE_MS:.0f}", "--"))
    print(_row("Steps above requirement (total)", cell(o["n_above"], "{:.0f}"),
               cell(n["n_above"], "{:.0f}"), "--"))
    print(_row("Share above requirement (total) [%]", cell(o["share_above"], "{:.2f}"),
               cell(n["share_above"], "{:.2f}"), "--"))


# --------------------------------------------------------------------------- #
# Final covariance comparison
# --------------------------------------------------------------------------- #
def load_final_covariance(run_dir: Path) -> np.ndarray:
    """Load the saved final joint covariance query for one run."""
    return np.load(run_dir / "final_joint_covariance.npy")


def print_covariance_comparison(old_dirs: list[Path], new_dirs: list[Path]) -> None:
    """Compare the final joint covariance recovered by the two methods.

    Each method is deterministic across runs, so run_1 is representative.
    """
    g = load_final_covariance(old_dirs[0])
    s = load_final_covariance(new_dirs[0])
    print("\nFinal joint covariance: Global (reference) vs Steiner")
    if g.shape != s.shape:
        print(f"  shape mismatch: global={g.shape} steiner={s.shape}")
        return

    diff = s - g
    frob = float(np.linalg.norm(diff))
    rel_frob = frob / float(np.linalg.norm(g))
    max_abs = float(np.max(np.abs(diff)))
    print(f"  matrix size                    : {g.shape[0]}x{g.shape[1]}")
    print(f"  Frobenius norm of difference   : {frob:.3e}")
    print(f"  Relative Frobenius norm        : {rel_frob:.3e}")
    print(f"  Max absolute element-wise error: {max_abs:.3e}")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def per_step_stacks(run_dirs: list[Path]) -> dict[str, np.ndarray]:
    """Per-step component/total arrays stacked across runs, shape (n_runs, n_steps)."""
    all_steps = [SlamLogger.load_steps(d) for d in run_dirs]
    n = min(len(s["scan_step"]) for s in all_steps)

    def stack(key: str) -> np.ndarray:
        return np.vstack(
            [np.nan_to_num(np.asarray(s[key], dtype=float)[:n]) for s in all_steps]
        )

    def first(key: str) -> np.ndarray:
        return np.nan_to_num(np.asarray(all_steps[0][key], dtype=float)[:n])

    cov, assoc, opt = stack("duration_covariance_extraction"), stack("duration_association"), stack("duration_optimization")
    step = stack("duration_step")
    return {
        "scan_step": np.asarray(all_steps[0]["scan_step"][:n], dtype=float),
        "cov": cov,
        "assoc": assoc,
        "opt": opt,
        "step": step,
        "other": np.maximum(step - (cov + assoc + opt), 0.0),
        # counts are deterministic across runs -> take them from the first run
        "num_local_landmarks": first("num_local_landmarks"),
        "num_support_cliques": first("num_support_cliques"),
    }


def plot_per_step_timing_mean(stacks: dict[str, np.ndarray]) -> tuple[plt.Figure, plt.Axes]:
    """Per-step timing: mean-component stack (additive -> mean total) with a
    mean-to-max-across-runs ribbon on the total, whose peak is the worst-case step."""
    fig, ax = plt.subplots(figsize=(8, 2.5), tight_layout=True)
    steps = stacks["scan_step"]

    # Mean across runs of each component is additive: the stack top is the mean total.
    parts_ms = [1000.0 * stacks[k].mean(axis=0) for k in ("cov", "assoc", "opt", "other")]
    labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
    ax.stackplot(steps, *parts_ms, labels=labels)

    # Run-to-run spread of the total per-step runtime; the top edge is the worst case.
    total_mean = 1000.0 * stacks["step"].mean(axis=0)
    total_max = 1000.0 * stacks["step"].max(axis=0)
    ax.fill_between(steps, total_mean, total_max, color="0.35", alpha=0.50,
                    linewidth=0, zorder=5, label="Total max across runs")
    # ax.plot(steps, total_max, color="0.3", lw=0.5, alpha=0.7, zorder=6)

    ax.axhline(DEADLINE_MS, color="red", ls="--", lw=1.0, label="Deadline", zorder=7)

    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Scan step")
    ax.set_xlim(steps[0], steps[-1])
    ax.grid(True, lw=0.4)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return fig, ax


def make_plots(tag: str, run_dirs: list[Path]) -> None:
    """Generate and save the three diagnostic figures for one method."""
    stacks = per_step_stacks(run_dirs)
    steps = stacks["scan_step"]
    n_local = stacks["num_local_landmarks"]
    n_support = stacks["num_support_cliques"]

    # Median cov across runs -> spike-robust time series for the median-based plots.
    t_cov = np.median(stacks["cov"], axis=0)

    fig, _ = plot_landmarks_and_timing(steps, t_cov, n_local, n_support)
    save_figure(fig, f"vp_{tag}_landmarks_and_timing")

    fig, _ = plot_per_step_timing_mean(stacks)
    save_figure(fig, f"vp_{tag}_per_step_timing")

    fig, _ = plot_timing_vs_landmarks(n_local, n_support, t_cov)
    save_figure(fig, f"vp_{tag}_timing_vs_landmarks")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(show: bool = True) -> None:
    apply_thesis_style()

    old_dirs = run_dirs(OLD_BASE)
    new_dirs = run_dirs(NEW_BASE)

    old_metrics = aggregate_metrics(old_dirs)
    new_metrics = aggregate_metrics(new_dirs)
    print_cumulative_table(old_metrics, new_metrics)
    print_per_step_table(old_metrics, new_metrics)
    print_covariance_comparison(old_dirs, new_dirs)

    make_plots("global", old_dirs)
    make_plots("steiner", new_dirs)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
