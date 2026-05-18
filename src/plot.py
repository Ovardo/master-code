"""
Plot saved SLAM runs.

Examples
--------
From a script or notebook:

    from plot import SlamPlotter

    plotter = SlamPlotter.from_run("runs/20260516_205226", load_gps=True)
    plotter.trajectory(covariances=True, gnss=True)
    plotter.save_all(fmt="pdf", covariances=True, gnss=True)

    comparison = SlamRunComparison.from_runs(["runs/run_a", "runs/run_b"])
    comparison.timing_total()
    comparison.save_all()

From the command line:

    python src/plot.py runs/20260516_205226 --covariances --gnss --fmt pdf
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from data_loader import VictoriaParkLoader
from logger import SlamLogger
from utils.utils_math import rotmat2


def _confidence_ellipse_2d(
    ax,
    center: np.ndarray,
    cov: np.ndarray,
    scale: float = 1.0,
    **kwargs,
) -> None:
    """Draw a 95% confidence ellipse for a 2D covariance matrix."""
    chi2_95_2d = 2.447746830681

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width = np.sqrt(eigvals[0]) * 2 * chi2_95_2d * scale
    height = np.sqrt(eigvals[1]) * 2 * chi2_95_2d * scale

    ellipse = Ellipse(
        xy=tuple(center),
        width=width,
        height=height,
        angle=np.degrees(angle),
        **kwargs,
    )
    ax.add_patch(ellipse)


@dataclass(slots=True)
class SlamPlotter:
    """Convenience API for plotting one saved SLAM run."""

    run_dir: Path
    step_data: dict
    snapshots: list[dict]
    gps_data: np.ndarray | None = None

    @classmethod
    def from_run(cls, run_dir: Path | str, load_gps: bool = False) -> "SlamPlotter":
        run_dir = Path(run_dir)
        gps_data = VictoriaParkLoader().gps if load_gps else None
        return cls(
            run_dir=run_dir,
            step_data=SlamLogger.load(run_dir),
            snapshots=SlamLogger.load_snapshots(run_dir),
            gps_data=gps_data,
        )

    @property
    def final_snapshot(self) -> dict:
        return self.snapshots[-1]

    def trajectory(
        self,
        covariances: bool = False,
        gnss: bool = False,
        cov_stride: int = 10,
    ) -> plt.Figure:
        """Plot the final trajectory, landmarks, and optional overlays."""
        snapshot = self.final_snapshot
        poses = snapshot["poses"]
        landmarks = snapshot["landmarks"]
        meta = self.step_data.get("metadata", {})

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.plot(
            poses[:, 0],
            poses[:, 1],
            color="steelblue",
            lw=1.5,
            label="SLAM trajectory",
            zorder=2,
        )
        ax.scatter(
            poses[0, 0],
            poses[0, 1],
            color="green",
            s=80,
            zorder=5,
            label="Start",
        )
        ax.scatter(
            poses[-1, 0],
            poses[-1, 1],
            color="red",
            s=80,
            zorder=5,
            label="End",
        )
        ax.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            c="tomato",
            marker="x",
            s=40,
            lw=1.2,
            label=f"Landmarks ({len(landmarks)})",
            zorder=3,
        )

        if gnss and self.gps_data is not None:
            ax.scatter(
                self.gps_data[:, 1],
                self.gps_data[:, 2],
                c="gold",
                marker=".",
                s=24,
                alpha=0.6,
                label="GPS",
                zorder=1,
            )

        if covariances:
            self._draw_trajectory_covariances(ax, snapshot, cov_stride)

        title = "SLAM Trajectory"
        if meta.get("num_landmarks"):
            title += f" | {meta['num_landmarks']} landmarks"

        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, lw=0.4)
        fig.tight_layout()
        return fig

    def timing_breakdown(self) -> plt.Figure:
        """Plot per-step and cumulative processing time breakdown."""
        t_cov = self.step_data["time_covariance_extraction"]
        t_assoc = self.step_data["time_association"]
        t_opt = self.step_data["time_optimization"]
        t_total = self.step_data["time_total"]
        other = np.maximum(t_total - (t_cov + t_assoc + t_opt), 0.0)

        steps = self.step_data["steps"]
        labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
        parts = [t_cov, t_assoc, t_opt, other]

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        axes[0].stackplot(steps, *parts, labels=labels)
        axes[0].set_ylabel("Time (s)")
        axes[0].set_title("Per-step Processing Time Breakdown")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].grid(True, lw=0.4)

        axes[1].stackplot(steps, *[np.cumsum(p) for p in parts], labels=labels)
        axes[1].set_ylabel("Cumulative time (s)")
        axes[1].set_xlabel("Scan step")
        axes[1].set_title("Cumulative Processing Time Breakdown")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, lw=0.4)

        fig.tight_layout()
        return fig

    def timing_over_time(self) -> plt.Figure:
        """Plot total step time and in-view predicted landmark count."""
        steps = self.step_data["steps"]
        t_total = self.step_data["time_total"]
        n_local = self.step_data["count_local_landmarks"]

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

    def timing_vs_landmarks(self) -> plt.Figure:
        """Plot total step time against in-view landmark count."""
        n_local = self.step_data["count_local_landmarks"]
        t_total = self.step_data["time_total"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(n_local, t_total, alpha=0.4, s=12, color="steelblue")
        ax.set_xlabel("In-view landmark count")
        ax.set_ylabel("Step processing time (s)")
        ax.set_title("Processing Time vs. In-view Landmarks")
        ax.grid(True, lw=0.4)
        fig.tight_layout()
        return fig

    def landmark_growth(self) -> plt.Figure:
        """Plot confirmed landmark count over time."""
        steps = self.step_data["steps"]
        n_landmarks = self.step_data["count_total_landmarks"]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, n_landmarks, lw=1.2, color="steelblue")
        ax.set_xlabel("Scan step")
        ax.set_ylabel("Confirmed landmarks")
        ax.set_title("Landmark Count Over Time")
        ax.grid(True, lw=0.4)
        fig.tight_layout()
        return fig

    def snapshot_grid(
        self,
        max_cols: int = 3,
        covariances: bool = False,
        max_snapshots: int = 9,
    ) -> plt.Figure:
        """Plot a small grid of snapshots to show map growth."""
        stride = max(1, len(self.snapshots) // max_snapshots)
        snapshots = self.snapshots[::stride][:max_snapshots]

        cols = min(len(snapshots), max_cols)
        rows = (len(snapshots) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(5 * cols, 5 * rows),
            squeeze=False,
        )

        for idx, snapshot in enumerate(snapshots):
            ax = axes[idx // cols][idx % cols]
            step = int(snapshot["step"])
            poses = snapshot["poses"]
            landmarks = snapshot["landmarks"]

            ax.plot(poses[:, 0], poses[:, 1], color="steelblue", lw=1.0)
            ax.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                c="tomato",
                marker="x",
                s=20,
                lw=0.8,
                zorder=3,
            )

            if covariances:
                self._draw_landmark_covariances(ax, snapshot, alpha=0.15, lw=0.4)

            ax.set_title(f"Step {step} | {len(landmarks)} landmarks", fontsize=9)
            ax.set_aspect("equal")
            ax.grid(True, lw=0.3)

        for idx in range(len(snapshots), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle("SLAM Map Growth", fontsize=12, y=1.01)
        fig.tight_layout()
        return fig

    def standard_figures(
        self,
        covariances: bool = False,
        gnss: bool = False,
    ) -> dict[str, plt.Figure]:
        """Create the default figure set."""
        return {
            "trajectory": self.trajectory(covariances=covariances, gnss=gnss),
            "timing_breakdown": self.timing_breakdown(),
            "timing_over_time": self.timing_over_time(),
            "timing_vs_landmarks": self.timing_vs_landmarks(),
            "landmark_growth": self.landmark_growth(),
        }

    def save_all(
        self,
        out_dir: Path | str | None = None,
        fmt: str = "pdf",
        covariances: bool = False,
        gnss: bool = False,
    ) -> list[Path]:
        """Save the default figure set and close the figures."""
        out_dir = Path(out_dir) if out_dir is not None else self.run_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        figures = self.standard_figures(covariances=covariances, gnss=gnss)
        for name, fig in figures.items():
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

        print(f"[plot] Saved {len(paths)} figures to {out_dir.resolve()}")
        return paths

    def show_all(self, covariances: bool = False, gnss: bool = False) -> None:
        """Show the default figure set."""
        self.standard_figures(covariances=covariances, gnss=gnss)
        plt.show()

    def _draw_trajectory_covariances(
        self,
        ax,
        snapshot: dict,
        cov_stride: int,
    ) -> None:
        pose_covariances = snapshot.get("poses_covariance", np.empty((0,)))
        if pose_covariances.ndim == 3:
            for k in range(0, len(snapshot["poses"]), cov_stride):
                pose = snapshot["poses"][k]
                cov = pose_covariances[k]
                rotation = rotmat2(pose[2])
                cov_translation = rotation @ cov[:2, :2] @ rotation.T
                _confidence_ellipse_2d(
                    ax,
                    pose[:2],
                    cov_translation,
                    fc="steelblue",
                    alpha=0.3,
                    ec="steelblue",
                    lw=0.5,
                )

        self._draw_landmark_covariances(ax, snapshot, alpha=0.3, lw=0.5)

    def _draw_landmark_covariances(
        self,
        ax,
        snapshot: dict,
        alpha: float,
        lw: float,
    ) -> None:
        landmark_covariances = snapshot.get("landmarks_covariance", np.empty((0,)))
        if landmark_covariances.ndim != 3:
            return

        for landmark, cov in zip(snapshot["landmarks"], landmark_covariances):
            _confidence_ellipse_2d(
                ax,
                landmark,
                cov,
                fc="tomato",
                alpha=alpha,
                ec="tomato",
                lw=lw,
            )


@dataclass(slots=True)
class SlamRunComparison:
    """Convenience API for plotting several saved SLAM runs together."""

    runs: list[SlamPlotter]
    labels: list[str]

    @classmethod
    def from_runs(
        cls,
        run_dirs: Sequence[Path | str],
        labels: Sequence[str] | None = None,
        load_gps: bool = False,
    ) -> "SlamRunComparison":
        runs = [
            SlamPlotter.from_run(run_dir, load_gps=load_gps)
            for run_dir in run_dirs
        ]
        run_labels = (
            list(labels)
            if labels is not None
            else [run.run_dir.name for run in runs]
        )
        return cls(runs=runs, labels=run_labels)

    def __post_init__(self) -> None:
        if len(self.runs) != len(self.labels):
            raise ValueError("runs and labels must have the same length.")

    def timing_total(self, log_scale: bool = True) -> plt.Figure:
        """Plot total step time for each run."""
        fig, ax = plt.subplots(figsize=(10, 5))

        for run, label in zip(self.runs, self.labels):
            ax.plot(
                run.step_data["steps"],
                run.step_data["time_total"],
                lw=1.0,
                label=label,
            )

        ax.set_xlabel("Scan step")
        ax.set_ylabel("Step processing time (s)")
        ax.set_title("Total Step Time Comparison")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", lw=0.4)
        fig.tight_layout()
        return fig

    def standard_figures(self) -> dict[str, plt.Figure]:
        """Create the default comparison figure set."""
        return {
            "timing_total_comparison": self.timing_total(),
        }

    def save_all(
        self,
        out_dir: Path | str | None = None,
        fmt: str = "pdf",
    ) -> list[Path]:
        """Save the default comparison figure set and close the figures."""
        if out_dir is None:
            out_dir = self.runs[0].run_dir.parent / "comparison_figures"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for name, fig in self.standard_figures().items():
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

        print(f"[plot] Saved {len(paths)} comparison figures to {out_dir.resolve()}")
        return paths

    def show_all(self) -> None:
        """Show the default comparison figure set."""
        self.standard_figures()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved SLAM run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--covariances", action="store_true")
    parser.add_argument("--gnss", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plotter = SlamPlotter.from_run(args.run_dir, load_gps=args.gnss)
    plotter.save_all(
        out_dir=args.save_dir,
        fmt=args.fmt,
        covariances=args.covariances,
        gnss=args.gnss,
    )

    if not args.no_show:
        plotter.show_all(covariances=args.covariances, gnss=args.gnss)

    comparison = SlamRunComparison.from_runs(
        run_dirs=["runs/vp1_20260513_175021_forward", "runs/vp1_20260513_181822_backward"],
        labels=["Baseline", "New association"],
    )

    comparison.save_all(fmt="pdf")
    comparison.show_all()



if __name__ == "__main__":
    main()
