"""Run the simulated SLAM config several times and plot median runtimes.

The covariance-recovery step has random wall-clock spikes that vary from run to
run.  Because the simulated dataset and the iSAM2 path are deterministic, the
*algorithm* is identical every run -- only the measured timing fluctuates.  So
running the same ``sim.yaml`` config N times and taking the per-step median
across runs cancels the measurement noise while keeping the real timing
structure, giving a fair, spike-robust picture for comparison.

Usage::

    python -m master_code.experiments.runtime_multi_run                       # sim, 10 x 1000
    python -m master_code.experiments.runtime_multi_run --dataset real        # real, 10 x 7300
    python -m master_code.experiments.runtime_multi_run --runs 2 --steps 100
    python -m master_code.experiments.runtime_multi_run --reuse <dir>         # plot only
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from master_code.config import SlamConfig
from master_code.logger import SlamLogger
from master_code.paths import RUNS_ROOT
from master_code.plotting.thesis_style import (
    apply_thesis_style,
    save_figure,
    thesis_figsize,
)
from master_code.run_real import run_real
from master_code.run_sim import run_sim

N_RUNS = 10
EXPERIMENT_NAME = "runtime_multi_run"

# Per-dataset: SLAM entry point, config file, and default step count.
DATASETS = {
    "sim": {"run_fn": run_sim, "config": "sim.yaml", "steps": 1000},
    "real": {"run_fn": run_real, "config": "real.yaml", "steps": 7300},
}

# (steps.npz key, legend label, colour) -- order = cumulative stacking order.
TIMING_COMPONENTS = [
    ("duration_covariance_extraction", "Covariance", "tab:blue"),
    ("duration_association", "Association", "tab:orange"),
    ("duration_optimization", "Optimization", "tab:green"),
]
OTHER_COLOR = "tab:red"


# --------------------------------------------------------------------------- #
# Run phase
# --------------------------------------------------------------------------- #
def run_repeated(
    run_fn,
    config: SlamConfig,
    experiment_dir: Path,
    n_runs: int,
    num_steps: int,
) -> list[Path]:
    """Run the SLAM config ``n_runs`` times, return the per-run output dirs."""
    run_dirs: list[Path] = []
    for i in range(n_runs):
        out = experiment_dir / f"run_{i:02d}"
        print(f"[{i + 1}/{n_runs}] running -> {out}")
        run_fn(
            config=config,
            output_dir=out,
            num_steps=num_steps,
            show_plots=False,
            save_plots=False,
        )
        run_dirs.append(out)
    return run_dirs


def find_run_dirs(experiment_dir: Path) -> list[Path]:
    """Return existing ``run_*`` dirs (with steps.npz) inside an experiment dir."""
    run_dirs = sorted(
        d for d in experiment_dir.glob("run_*") if (d / "steps.npz").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_*/steps.npz found under {experiment_dir.resolve()}"
        )
    return run_dirs


# --------------------------------------------------------------------------- #
# Aggregation phase
# --------------------------------------------------------------------------- #
def _get(steps: dict[str, np.ndarray], key: str, n: int) -> np.ndarray:
    """Return a logged array (nan->0), or zeros if the key is missing."""
    values = steps.get(key)
    if values is None:
        return np.zeros(n, dtype=float)
    return np.nan_to_num(np.asarray(values, dtype=float)[:n], nan=0.0)


def aggregate_runs(run_dirs: list[Path]) -> dict[str, np.ndarray]:
    """Median / IQR of per-step timings across runs, plus deterministic counts."""
    all_steps = [SlamLogger.load_steps(d) for d in run_dirs]
    n = min(len(s["scan_step"]) for s in all_steps)

    timing_keys = [key for key, _, _ in TIMING_COMPONENTS] + ["duration_step"]
    median: dict[str, np.ndarray] = {}
    q25: dict[str, np.ndarray] = {}
    q75: dict[str, np.ndarray] = {}
    for key in timing_keys:
        stack = np.vstack([_get(s, key, n) for s in all_steps])  # (n_runs, n)
        median[key] = np.median(stack, axis=0)
        q25[key] = np.percentile(stack, 25, axis=0)
        q75[key] = np.percentile(stack, 75, axis=0)

    cov, assoc, opt = (median[k] for k, _, _ in TIMING_COMPONENTS)
    other = np.maximum(median["duration_step"] - (cov + assoc + opt), 0.0)

    # Counts are deterministic across runs -> take them from the first run.
    first = all_steps[0]
    return {
        "n_runs": len(run_dirs),
        "scan_step": np.asarray(first["scan_step"][:n], dtype=float),
        "num_local_landmarks": _get(first, "num_local_landmarks", n),
        "num_support_cliques": _get(first, "num_support_cliques", n),
        "median": median,
        "q25": q25,
        "q75": q75,
        "other": other,
    }


# --------------------------------------------------------------------------- #
# Plot phase
# --------------------------------------------------------------------------- #
def plot_cov_perstep(agg: dict) -> plt.Figure:
    """Per-step median covariance time (+IQR) vs in-view landmarks / cliques."""
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 4.5), sharex=True, tight_layout=True
    )
    steps = agg["scan_step"]

    cov_med = 1000.0 * agg["median"]["duration_covariance_extraction"]
    cov_q25 = 1000.0 * agg["q25"]["duration_covariance_extraction"]
    cov_q75 = 1000.0 * agg["q75"]["duration_covariance_extraction"]

    axes[0].plot(steps, cov_med, lw=0.8, color="tab:blue", label="Median cov. recovery")
    axes[0].fill_between(
        steps, cov_q25, cov_q75, color="tab:blue", alpha=0.2, linewidth=0, label="IQR"
    )
    axes[0].set_ylabel("Time [ms]")
    axes[0].set_title(f"Covariance Recovery Time (median over {agg['n_runs']} runs)")
    axes[0].grid(True, lw=0.4)
    axes[0].legend(loc="upper left")

    axes[1].plot(steps, agg["num_local_landmarks"], lw=0.8, color="tab:orange")
    axes[1].set_ylabel("# In-view landmarks")
    axes[1].grid(True, lw=0.4)

    axes[2].plot(steps, agg["num_support_cliques"], lw=0.8, color="tab:green")
    axes[2].set_ylabel("# Support cliques")
    axes[2].set_xlabel("Scan step")
    axes[2].grid(True, lw=0.4)

    return fig


def plot_cumulative_components(agg: dict) -> plt.Figure:
    """Stacked cumulative runtime by component (median per-step), with total IQR."""
    fig, ax = plt.subplots(figsize=(8, 2.5), tight_layout=True)
    steps = agg["scan_step"]

    parts = [np.cumsum(agg["median"][key]) for key, _, _ in TIMING_COMPONENTS]
    parts.append(np.cumsum(agg["other"]))
    labels = [label for _, label, _ in TIMING_COMPONENTS] + ["Other"]
    colors = [color for _, _, color in TIMING_COMPONENTS] + [OTHER_COLOR]

    ax.stackplot(steps, *parts, labels=labels, colors=colors)

    # Run-to-run spread of the cumulative total (from per-step step-time IQR).
    # total_q25 = np.cumsum(agg["q25"]["duration_step"])
    # total_q75 = np.cumsum(agg["q75"]["duration_step"])
    # ax.fill_between(
    #     steps, total_q25, total_q75, color="gray", alpha=0.25, linewidth=0,
    #     label="Total IQR",
    # )

    total_med = float(np.sum(agg["median"]["duration_step"]))
    ax.set_ylabel("Cumulative runtime [s]")
    ax.set_xlabel("Scan step")
    ax.set_title(f"Cumulative Runtime by Component (median total: {total_med:.2f} s)")
    ax.grid(True, lw=0.4)
    ax.legend(loc="upper left", fontsize=8)

    return fig


def print_summary(agg: dict) -> None:
    med_step = agg["median"]["duration_step"]
    med_cov = agg["median"]["duration_covariance_extraction"]
    print(f"\nRuntime summary (median over {agg['n_runs']} runs)")
    print(f"  total runtime          : {np.sum(med_step):8.2f} s")
    print(f"  mean per-step time     : {1000.0 * np.mean(med_step):8.2f} ms")
    print(f"  p95 per-step time      : {1000.0 * np.percentile(med_step, 95):8.2f} ms")
    print(f"  p95 covariance time    : {1000.0 * np.percentile(med_cov, 95):8.2f} ms")
    print(f"  max covariance time    : {1000.0 * np.max(med_cov):8.2f} ms")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="sim", help="which dataset to run"
    )
    parser.add_argument("--runs", type=int, default=N_RUNS, help="number of repeated runs")
    parser.add_argument(
        "--steps", type=int, default=None, help="steps per run (default: dataset default)"
    )
    parser.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="reuse an existing experiment dir (skip running, just (re)plot)",
    )
    parser.add_argument("--no-show", action="store_true", help="do not display figures")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    num_steps = args.steps if args.steps is not None else dataset["steps"]

    if args.reuse is not None:
        run_dirs = find_run_dirs(args.reuse)
        print(f"Reusing {len(run_dirs)} runs from {args.reuse.resolve()}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = RUNS_ROOT / "experiments" / EXPERIMENT_NAME / args.dataset / timestamp
        config = SlamConfig.load(dataset["config"])
        run_dirs = run_repeated(
            dataset["run_fn"], config, experiment_dir, args.runs, num_steps
        )
        print(f"Finished runs: {experiment_dir.resolve()}")

    agg = aggregate_runs(run_dirs)

    apply_thesis_style()
    fig_perstep = plot_cov_perstep(agg)
    save_figure(fig_perstep, f"{args.dataset}_runtime_cov_perstep")
    fig_cumulative = plot_cumulative_components(agg)
    save_figure(fig_cumulative, f"{args.dataset}_runtime_cumulative_components")

    print_summary(agg)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
