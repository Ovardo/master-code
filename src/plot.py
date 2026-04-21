"""
Standalone plotting script for saved SLAM runs.

Run from the project root:
    python plot.py results/vp1_20240417_120000 [--covariances] [--fmt pdf|png|svg] [--no-show]

Or import and call functions from a notebook:
    from logger import SlamLogger
    from plot import plot_trajectory, plot_timing_breakdown, plot_snapshot_animation

    data      = SlamLogger.load("results/vp1_20240417_120000")
    snapshots = SlamLogger.load_snapshots("results/vp1_20240417_120000")
    plot_trajectory(snapshots[-1], data)   # final snapshot + step data overlay
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

from data_loader import VictoriaParkLoader
from logger import SlamLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_ellipse_2d(ax, 
                           center: np.ndarray, 
                           cov: np.ndarray,
                           scale: float = 1, 
                           **kwargs) -> None:
    """Draw a 2-D confidence ellipse for a 2×2 covariance matrix."""
    k = 2.447746830681 # 95% confidence interval for 2 DOF
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)  # clamp floating-point negatives
    angle   = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width   = np.sqrt(eigvals[0]) * 2 * k * scale
    height  = np.sqrt(eigvals[1]) * 2 * k * scale

    ellipse = Ellipse(xy=tuple(center),
                      width=width,
                      height=height,
                      angle=np.degrees(angle),
                      **kwargs)
    
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------

def plot_trajectory(
    snapshot: dict,
    step_data: dict | None = None,
    show_covariances: bool = False,
    show_gnss_overlay: bool = False,
    gps_data: np.ndarray | None = None,
    cov_stride: int = 10,
) -> plt.Figure:
    """
    Estimated trajectory + landmarks + dead-reckoning overlay.

    Parameters
    ----------
    snapshot:
        Dict from ``SlamLogger.load_snapshot()`` — typically the final one.
    step_data:
        Dict from ``SlamLogger.load()`` — used only for metadata in the title.
    show_covariances:
        Overlay 2-σ ellipses on poses and landmarks.
    show_gnss_overlay:
        Overlay raw GPS positions from ``VictoriaParkLoader.gps``.
    gps_data:
        GPS array with shape (K, 3) as [x, y, t]. Used only when
        ``show_gnss_overlay`` is True.
    cov_stride:
        Draw a pose ellipse every this many poses.
    """
    poses     = snapshot["poses"]
    landmarks = snapshot["landmarks"]
    meta      = (step_data or {}).get("metadata", {})

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(poses[:, 0], poses[:, 1],
            color="steelblue", lw=1.5, label="SLAM trajectory", zorder=2)
    ax.scatter(poses[0, 0], poses[0, 1], color="green", s=80, zorder=5,
            label="Start")
    ax.scatter(poses[-1, 0], poses[-1, 1], color="red", s=80, zorder=5,
            label="End")
    ax.scatter(landmarks[:, 0], landmarks[:, 1],
               c="tomato", marker="x", s=40, lw=1.2,
               label=f"Landmarks ({len(landmarks)})", zorder=3)

    if show_gnss_overlay and gps_data is not None and gps_data.size > 0:
        ax.scatter(gps_data[:, 0], gps_data[:, 1],
                   c="gold", marker=".", s=24, alpha=0.6,
                   label="GPS", zorder=1)

    if show_covariances:
        pose_covs = snapshot.get("poses_covariance", np.empty((0,)))
        lm_covs   = snapshot.get("landmarks_covariance", np.empty((0,)))
        if pose_covs.ndim == 3:
            for k in range(0, len(poses), cov_stride):
                _confidence_ellipse_2d(ax, poses[k, :2], pose_covs[k][:2, :2],
                                    fc="steelblue", alpha=0.3,
                                    ec="steelblue", lw=0.5)
        if lm_covs.ndim == 3:
            for j in range(len(landmarks)):
                _confidence_ellipse_2d(ax, landmarks[j], lm_covs[j],
                                    fc="tomato", alpha=0.3,
                                    ec="tomato", lw=0.5)

    title = "SLAM Trajectory"
    if meta.get("num_landmarks"):
        title += f"  |  {meta['num_landmarks']} landmarks"
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig


def plot_timing_breakdown(step_data: dict) -> plt.Figure:
    """
    Two-panel stacked timing figure:
      top    — per-step breakdown
      bottom — cumulative breakdown
    """
    t_cov   = step_data["time_covariance_extraction"]
    t_assoc = step_data["time_association"]
    t_opt   = step_data["time_optimization"]
    t_total = step_data["time_total"]
    other   = np.maximum(t_total - (t_cov + t_assoc + t_opt), 0.0)

    steps   = step_data["steps"]
    labels  = ["Covariance extraction", "Association", "Optimisation", "Other"]
    # colours = ["#4c72b0", "#dd8452", "#55a868", "#ccb974"]
    parts   = [t_cov, t_assoc, t_opt, other]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].stackplot(steps, *parts, labels=labels)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Per-step Processing Time Breakdown")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, lw=0.4)

    axes[1].stackplot(steps, *[np.cumsum(p) for p in parts],
                      labels=labels)
    axes[1].set_ylabel("Cumulative time (s)")
    axes[1].set_xlabel("Scan step")
    axes[1].set_title("Cumulative Processing Time Breakdown")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, lw=0.4)

    fig.tight_layout()
    return fig


def plot_timing_over_time(step_data: dict) -> plt.Figure:
    """
    Two-panel figure:
      top    — total step time (log scale)
      bottom — in-view predicted landmark count
    """
    steps   = step_data["steps"]
    t_total = step_data["time_total"]
    n_local = step_data["count_local_landmarks"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(steps, t_total, lw=0.8, color="steelblue")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_yscale("log")
    axes[0].set_title("SLAM Step Processing Time")
    axes[0].grid(True, which="both", lw=0.4)

    axes[1].plot(steps, n_local, lw=0.8, color="tomato")
    axes[1].set_ylabel("In-view landmarks")
    axes[1].set_xlabel("Scan step")
    axes[1].set_title("In-view Predicted Landmark Count")
    axes[1].grid(True, lw=0.4)

    fig.tight_layout()
    return fig


def plot_timing_vs_landmarks(step_data: dict) -> plt.Figure:
    """Scatter: total step time vs number of in-view landmarks."""
    n_local = step_data["count_local_landmarks"]
    t_total = step_data["time_total"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(n_local, t_total, alpha=0.4, s=12, color="steelblue")
    ax.set_xlabel("In-view landmark count")
    ax.set_ylabel("Step processing time (s)")
    ax.set_title("Processing Time vs. In-view Landmarks")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig


def plot_landmark_growth(step_data: dict) -> plt.Figure:
    """Total confirmed landmark count over time."""
    steps  = step_data["steps"]
    n_lm   = step_data["count_total_landmarks"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, n_lm, lw=1.2, color="steelblue")
    ax.set_xlabel("Scan step")
    ax.set_ylabel("Confirmed landmarks")
    ax.set_title("Landmark Count Over Time")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig


def plot_snapshot_grid(
    snapshots: list[dict],
    max_cols: int = 3,
    show_covariances: bool = False,
) -> plt.Figure:
    """
    Tile multiple snapshots in a grid — useful for showing map growth over
    time in a thesis figure.

    Parameters
    ----------
    snapshots:
        List returned by ``SlamLogger.load_snapshots()``.
    max_cols:
        Maximum columns in the grid.
    show_covariances:
        Overlay 2-σ landmark ellipses.
    """
    n    = len(snapshots)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             squeeze=False)

    for idx, snap in enumerate(snapshots):
        ax     = axes[idx // cols][idx % cols]
        step   = int(snap["step"])
        poses  = snap["poses"]
        lms    = snap["landmarks"]

        ax.plot(poses[:, 0], poses[:, 1], color="steelblue", lw=1.0)
        ax.scatter(lms[:, 0], lms[:, 1], c="tomato", marker="x",
                   s=20, lw=0.8, zorder=3)

        if show_covariances:
            lm_covs = snap.get("landmark_covariances", np.empty((0,)))
            if lm_covs.ndim == 3:
                for j in range(len(lms)):
                    _confidence_ellipse_2d(ax, lms[j], lm_covs[j], n_std=2,
                                        fc="tomato", alpha=0.15,
                                        ec="tomato", lw=0.4)

        ax.set_title(f"Step {step}  |  {len(lms)} landmarks", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("SLAM Map Growth", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save all figures at once
# ---------------------------------------------------------------------------

def save_all_figures(
    step_data: dict,
    snapshots: list[dict],
    out_dir: Path,
    fmt: str = "pdf",
    show_covariances: bool = False,
    show_gnss_overlay: bool = False,
    gps_data: np.ndarray | None = None,
) -> None:
    """
    Produce and save every standard figure to ``out_dir``.

    PDF is recommended for thesis inclusion (vector, lossless at any size).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_snap = snapshots[-1] if snapshots else None

    figures: dict[str, plt.Figure | None] = {}

    if final_snap is not None:
        figures["trajectory"] = plot_trajectory(
            final_snap, step_data, show_covariances=show_covariances,
            show_gnss_overlay=show_gnss_overlay, gps_data=gps_data,
        )

    figures["timing_breakdown"]    = plot_timing_breakdown(step_data)
    figures["timing_over_time"]    = plot_timing_over_time(step_data)
    figures["timing_vs_landmarks"] = plot_timing_vs_landmarks(step_data)
    figures["landmark_growth"]     = plot_landmark_growth(step_data)

    # if len(snapshots) > 1:
    #     # Subsample to at most 9 for the grid
    #     stride = max(1, len(snapshots) // 9)
    #     figures["snapshot_grid"] = plot_snapshot_grid(
    #         snapshots[::stride], show_covariances=show_covariances,
    #     )

    for name, fig in figures.items():
        if fig is None:
            continue
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
        plt.close(fig)

    print(f"[plot] All figures → {out_dir.resolve()}")


def load_and_plot_all(
    run_dir: Path,
    save_dir: Path | None = None,
    fmt: str = "pdf",
    show: bool = True,
    show_covariances: bool = False,
    show_gnss_overlay: bool = False,
) -> tuple[dict, list[dict]]:
    """
    Load a run and produce every standard figure.

    Returns
    -------
    (step_data, snapshots)
        The raw loaded dicts for any further custom analysis.
    """
    run_dir   = Path(run_dir)
    step_data = SlamLogger.load(run_dir)
    snapshots = SlamLogger.load_snapshots(run_dir)
    gps_data  = None

    if show_gnss_overlay:
        gps_data = VictoriaParkLoader().gps

    if save_dir is None:
        save_dir = run_dir / "figures"

    save_all_figures(step_data, snapshots, save_dir, fmt=fmt,
                     show_covariances=show_covariances,
                     show_gnss_overlay=show_gnss_overlay,
                     gps_data=gps_data)

    if show:
        final_snap = snapshots[-1] if snapshots else None
        if final_snap is not None:
            plot_trajectory(final_snap, step_data,
                            show_covariances=show_covariances,
                            show_gnss_overlay=show_gnss_overlay,
                            gps_data=gps_data)
        plot_timing_breakdown(step_data)
        plot_timing_over_time(step_data)
        plot_timing_vs_landmarks(step_data)
        plot_landmark_growth(step_data)
        # if len(snapshots) > 1:
        #     stride = max(1, len(snapshots) // 9)
        #     plot_snapshot_grid(snapshots[::stride],
        #                        show_covariances=show_covariances)
        plt.show()

    return step_data, snapshots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python plot.py <run_dir> [--no-show] [--covariances] [--gps] [--fmt pdf|png|svg]")
    #     sys.exit(1)

    # run_dir = Path(sys.argv[1])
    # show    = "--no-show"     not in sys.argv
    # covs    = "--covariances" in sys.argv
    # gps     = "--gps"         in sys.argv
    # fmt     = "pdf"
    # for arg in sys.argv:
    #     if arg.startswith("--fmt="):
    #         fmt = arg.split("=", 1)[1]
    run_dir = Path("/Users/ovar/Documents/Master/master_code/results/vp1_20260421_142156")
    fmt     = "pdf"

    load_and_plot_all(run_dir, fmt=fmt, show=True,
                      show_covariances=True, show_gnss_overlay=True)
