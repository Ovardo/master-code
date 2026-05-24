from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from master_code.plotter import SlamRunPlotter
from master_code.plotting.thesis_style import apply_thesis_style, save_figure


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: str
    color: str


RUNS = [
    RunSpec("M=1", "runs/20260524_000010_1_4", "steelblue"),
    RunSpec("M=2", "runs/20260524_002432_2_4", "lightcoral"),
    RunSpec("M=3", "runs/20260524_002818_3_4", "green"),
    RunSpec("M=4", "runs/20260524_003427_4_4", "orange"),
]

TIMING_COMPONENTS = [
    ("duration_covariance_extraction", "Covariance", "#4C78A8"),
    ("duration_association", "Association", "#F58518"),
    ("duration_optimization", "Optimization", "#54A24B"),
]


def get_step_array(run: SlamRunPlotter, key: str, default: float = 0.0) -> np.ndarray:
    """Return one logged step array, falling back to a constant array if absent."""
    values = run.steps.get(key)
    if values is None:
        n_steps = len(run.steps["scan_step"])
        return np.full(n_steps, default, dtype=float)
    return np.nan_to_num(np.asarray(values, dtype=float), nan=default)


def get_total_runtime(run: SlamRunPlotter) -> np.ndarray:
    """Return total per-step runtime in seconds."""
    total = run.steps.get("duration_update")
    if total is not None:
        return np.nan_to_num(np.asarray(total, dtype=float), nan=0.0)

    parts = [get_step_array(run, key) for key, _, _ in TIMING_COMPONENTS]
    return np.sum(parts, axis=0)


def binned_median(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 28,
    min_count: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarise noisy per-step timings by median and interquartile range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        empty = np.array([])
        return empty, empty, empty, empty

    if np.min(x) == np.max(x):
        return (
            np.array([np.median(x)]),
            np.array([np.median(y)]),
            np.array([np.percentile(y, 25)]),
            np.array([np.percentile(y, 75)]),
        )

    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    bin_ids = np.digitize(x, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if np.count_nonzero(in_bin) < min_count:
            continue
        rows.append(
            (
                np.median(x[in_bin]),
                np.median(y[in_bin]),
                np.percentile(y[in_bin], 25),
                np.percentile(y[in_bin], 75),
            )
        )

    if not rows:
        return (
            np.array([np.median(x)]),
            np.array([np.median(y)]),
            np.array([np.percentile(y, 25)]),
            np.array([np.percentile(y, 75)]),
        )

    return tuple(np.asarray(rows).T)


def plot_growth(ax: plt.Axes, runs: list[tuple[RunSpec, SlamRunPlotter]]) -> None:
    for spec, run in runs:
        ax.plot(
            run.steps["scan_step"],
            run.steps["num_landmarks"],
            label=spec.label,
            color=spec.color,
        )

    ax.set_title("Landmark Growth")
    ax.set_xlabel("Scan step")
    ax.set_ylabel("# confirmed landmarks")
    ax.grid(True, lw=0.4)
    ax.legend(title="N=4")


def plot_runtime_vs_landmarks(
    ax: plt.Axes,
    runs: list[tuple[RunSpec, SlamRunPlotter]],
) -> None:
    for spec, run in runs:
        x = run.steps["num_landmarks"]
        y_ms = 1000.0 * get_total_runtime(run)
        x_bin, y_med, y_q25, y_q75 = binned_median(x, y_ms)

        ax.plot(x_bin, y_med, marker="o", ms=3, label=spec.label, color=spec.color)
        ax.fill_between(x_bin, y_q25, y_q75, color=spec.color, alpha=0.15, linewidth=0)

    ax.set_title("Step Runtime vs Map Size")
    ax.set_xlabel("# confirmed landmarks")
    ax.set_ylabel("Runtime per step [ms]")
    ax.grid(True, lw=0.4)
    ax.legend(title="N=4")


def plot_total_breakdown(
    ax: plt.Axes,
    runs: list[tuple[RunSpec, SlamRunPlotter]],
) -> None:
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))

    component_totals = []
    for spec, run in runs:
        total = get_total_runtime(run)
        parts = [get_step_array(run, key) for key, _, _ in TIMING_COMPONENTS]
        other = np.maximum(total - np.sum(parts, axis=0), 0.0)
        component_totals.append([np.sum(part) for part in parts] + [np.sum(other)])

    component_totals = np.asarray(component_totals)
    bar_labels = [label for _, label, _ in TIMING_COMPONENTS] + ["Other"]
    bar_colors = [color for _, _, color in TIMING_COMPONENTS] + ["#D45320"]

    for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
        ax.bar(x, component_totals[:, i], bottom=bottom, label=label, color=color, width=0.65)
        bottom += component_totals[:, i]

    for i, (_, run) in enumerate(runs):
        final_landmarks = int(run.steps["num_landmarks"][-1])
        ax.text(i, bottom[i], f"{final_landmarks} lm", ha="center", va="bottom", fontsize=8)

    ax.set_title("Total Runtime Breakdown")
    ax.set_xticks(x, [spec.label for spec, _ in runs])
    ax.set_ylabel("Total runtime [s]")
    ax.grid(True, axis="y", lw=0.4)
    ax.legend()


def plot_component_scaling(
    runs: list[tuple[RunSpec, SlamRunPlotter]],
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharey=False, tight_layout=True)

    for ax, (key, title, _) in zip(axes, TIMING_COMPONENTS):
        for spec, run in runs:
            x = run.steps["num_local_landmarks"]
            y_ms = 1000.0 * get_step_array(run, key)
            x_bin, y_med, y_q25, y_q75 = binned_median(x, y_ms)

            ax.plot(x_bin, y_med, marker="o", ms=2.5, label=spec.label, color=spec.color)
            ax.fill_between(x_bin, y_q25, y_q75, color=spec.color, alpha=0.12, linewidth=0)

        ax.set_title(title)
        ax.set_xlabel("# in-view landmarks")
        ax.set_ylabel("Time [ms]")
        ax.grid(True, lw=0.4)

    axes[0].legend(title="N=4")
    return fig, axes


def print_runtime_summary(runs: list[tuple[RunSpec, SlamRunPlotter]]) -> None:
    print("Runtime summary")
    print("M   final landmarks   total [s]   mean step [ms]   p95 step [ms]")
    for spec, run in runs:
        total = get_total_runtime(run)
        print(
            f"{spec.label:<3} {int(run.steps['num_landmarks'][-1]):>15} "
            f"{np.sum(total):>11.1f} {1000.0 * np.mean(total):>16.2f} "
            f"{1000.0 * np.percentile(total, 95):>15.2f}"
        )


def main(show: bool = True) -> None:
    apply_thesis_style()

    runs = [(spec, SlamRunPlotter.from_run(spec.path)) for spec in RUNS]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    runs[0][1].plot_final_snapshot(axes[0, 0], title="M=1", show_legend=False, equal_aspect=False, show_grid=False, x_label=None, y_label=None)
    runs[1][1].plot_final_snapshot(axes[0, 1], title="M=2", show_legend=False, equal_aspect=False, show_grid=False, x_label=None, y_label=None)
    runs[2][1].plot_final_snapshot(axes[1, 0], title="M=3", show_legend=False, equal_aspect=False, show_grid=False, x_label=None, y_label=None)
    runs[3][1].plot_final_snapshot(axes[1, 1], title="M=4", show_legend=False, equal_aspect=False, show_grid=False, x_label=None, y_label=None)
    save_figure(fig, "compare_mn_estimate")

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), tight_layout=True)
    plot_growth(axes[0], runs)
    plot_runtime_vs_landmarks(axes[1], runs)
    plot_total_breakdown(axes[2], runs)
    save_figure(fig, "compare_mn_runtime")

    fig_components, _ = plot_component_scaling(runs)
    save_figure(fig_components, "compare_mn_runtime_components")

    print_runtime_summary(runs)
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
