from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from association import NIS, individualCompatibility
from config import SlamConfig
from data_loader import VictoriaParkLoader
from logger import SlamLogger
from utils import rotmat2


def confidence_ellipse_2d(center: np.ndarray, cov: np.ndarray, scale: float = 1.0, **kwargs) -> Ellipse:
    """Draw a 95% confidence ellipse for a 2D covariance matrix."""
    chi2_95_2d: float = 2.447746830681

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width = np.sqrt(eigvals[0]) * 2 * chi2_95_2d * scale
    height = np.sqrt(eigvals[1]) * 2 * chi2_95_2d * scale

    return Ellipse(
        xy=tuple(center),
        width=width,
        height=height,
        angle=np.degrees(angle),
        **kwargs,
    )

def draw_pose_covariances(ax, poses: np.ndarray, poses_cov: np.ndarray, cov_stride: int = 10) -> None:
    for pose, cov in zip(poses[::cov_stride], poses_cov[::cov_stride]):
        # Rotate the translational covariance ellipse to world frame
        R = rotmat2(pose[2])
        xy_cov = R @ cov[:2, :2] @ R.T
        
        ax.add_patch(
            confidence_ellipse_2d(pose[:2], xy_cov, fc="steelblue", alpha=0.3, ec="steelblue", lw=0.5)
        )

def draw_landmark_covariances(ax, landmarks: np.ndarray, landmarks_cov: np.ndarray, alpha: float, lw: float) -> None:
    for lm, cov in zip(landmarks, landmarks_cov):
        ax.add_patch(
            confidence_ellipse_2d(lm, cov, fc="tomato", alpha=alpha, ec="tomato", lw=lw)
        )

def plot_estimate(
    poses: np.ndarray | None = None,
    poses_cov: np.ndarray | None = None,
    poses_cov_stride: int = 10,
    landmarks: np.ndarray | None = None,
    landmarks_cov: np.ndarray | None = None,
    gnss: np.ndarray | None = None,
) -> plt.Figure:
    """Plot the trajectory, landmarks, and optional covariances and gnss."""
    
    fig, ax = plt.subplots(figsize=(10, 10))

    if poses is not None:
        ax.plot(poses[:, 0], poses[:, 1], color="steelblue", lw=1.5, label="Trajectory", zorder=2,)
        ax.scatter(poses[0, 0], poses[0, 1], color="green", s=80, zorder=5, label="Start")
        ax.scatter(poses[-1, 0], poses[-1, 1], color="red", s=80, zorder=5, label="End")

        if poses_cov is not None:
            draw_pose_covariances(ax, poses, poses_cov, cov_stride=poses_cov_stride)

    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c="tomato", marker=".", s=40, lw=1.2, label=f"Landmarks ({len(landmarks)})", zorder=3,)

        if landmarks_cov is not None:
            draw_landmark_covariances(ax, landmarks, landmarks_cov, alpha=0.3, lw=0.5)

    if gnss is not None:
        ax.scatter(gnss[:, 1], gnss[:, 2], c="gold", marker=".", s=24, alpha=0.6, label="GNSS", zorder=1)
    
    ax.set_title("MAP Estimate")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig


def plot_timing_breakdown(
    steps: np.ndarray,
    t_cov: np.ndarray,
    t_assoc: np.ndarray,
    t_opt: np.ndarray,
    t_total: np.ndarray,
) -> plt.Figure:
    """Plot per-step and cumulative processing time breakdown."""
    
    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_total = np.nan_to_num(t_total, nan=0.0)

    t_other = np.maximum(t_total - (t_cov + t_assoc + t_opt), 0.0)

    labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
    parts = [t_cov, t_assoc, t_opt, t_other]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    parts_ms = [1000*p for p in parts]
    axes[0].stackplot(steps, *parts_ms, labels=labels)
    axes[0].set_ylabel("Time (ms)")
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

def plot_timing_over_time(
    steps: np.ndarray,
    t_cov: np.ndarray,
    n_local: np.ndarray
) -> plt.Figure:
    """Plot covariance recovery step time and in-view predicted landmark count."""
   
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(steps, t_cov, lw=0.8, color="steelblue")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_yscale("log")
    axes[0].set_title("Covariance Recovery Step Processing Time")
    axes[0].grid(True, which="both", lw=0.4)

    axes[1].plot(steps, n_local, lw=0.8, color="tomato")
    axes[1].set_ylabel("In-view landmarks")
    axes[1].set_xlabel("Scan step")
    axes[1].set_title("In-view Predicted Landmark Count")
    axes[1].grid(True, lw=0.4)

    fig.tight_layout()
    return fig

def plot_timing_vs_landmarks(
    n_local: np.ndarray,
    t_cov: np.ndarray,
) -> plt.Figure:
    """Plot covariance recovery step time against in-view landmark count."""

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(n_local, t_cov, alpha=0.4, s=12, color="steelblue")
    ax.set_xlabel("In-view landmark count")
    ax.set_ylabel("Covariance recovery time (s)")
    ax.set_title("Covariance Recovery Time vs. In-view Landmarks")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig

def plot_landmark_growth(
    steps: np.ndarray, 
    n_landmarks: np.ndarray
) -> plt.Figure:
    """Plot confirmed landmark count over time."""

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, n_landmarks, lw=1.2, color="steelblue")
    ax.set_xlabel("Scan step")
    ax.set_ylabel("Confirmed landmarks")
    ax.set_title("Landmark Count Over Time")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig

def plot_error(
    steps: np.ndarray,
    error: np.ndarray,
    n_factors: np.ndarray,
) -> plt.Figure:
    """Plot the graph error over steps."""
    mask = ~np.isnan(error)
    error_avg = error[mask] / n_factors[mask] 

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps[mask], error_avg, lw=1.2, color="steelblue")
    ax.set_xlabel("Scan step")
    ax.set_ylabel("Average Factor Graph Error (mahalanobis / num_factors)")
    ax.set_title("SLAM Error Over Time")
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    return fig


def _nearest_indices(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """Return index of the nearest reference time for each query time."""
    insertion_indices = np.searchsorted(reference_times, query_times)
    right_indices = np.clip(insertion_indices, 0, len(reference_times) - 1)
    left_indices = np.clip(insertion_indices - 1, 0, len(reference_times) - 1)

    left_dt = np.abs(query_times - reference_times[left_indices])
    right_dt = np.abs(query_times - reference_times[right_indices])

    return np.where(left_dt <= right_dt, left_indices, right_indices)


def plot_gnss_pose_error(
    gnss: np.ndarray,
    poses: np.ndarray,
    pose_times: np.ndarray,
) -> plt.Figure:
    """
    Plot position error between GNSS samples and nearest-in-time pose estimates.

    Parameters
    ----------
    gnss
        Array with columns [time, x, y].
    poses
        Pose estimates with columns [x, y, theta].
    pose_times
        Timestamp for each pose estimate.
    """
    pose_times = np.asarray(pose_times, dtype=float)
    pose_xy = np.asarray(poses[:, :2], dtype=float)
    gnss = np.asarray(gnss, dtype=float)

    if len(pose_times) != len(pose_xy):
        raise ValueError("pose_times must have the same length as poses.")
    if len(pose_times) == 0:
        raise ValueError("At least one pose is required.")
    if len(gnss) == 0:
        raise ValueError("At least one GNSS measurement is required.")

    order = np.argsort(pose_times)
    pose_times = pose_times[order]
    pose_xy = pose_xy[order]

    valid_gnss = np.isfinite(gnss[:, 0]) & np.isfinite(gnss[:, 1]) & np.isfinite(gnss[:, 2])
    in_pose_interval = (pose_times[0] <= gnss[:, 0]) & (gnss[:, 0] <= pose_times[-1])
    gnss = gnss[valid_gnss & in_pose_interval]

    if len(gnss) == 0:
        raise ValueError("No GNSS measurements overlap the pose time interval.")

    nearest_pose_indices = _nearest_indices(gnss[:, 0], pose_times)
    matched_pose_xy = pose_xy[nearest_pose_indices]

    error_xy = matched_pose_xy - gnss[:, 1:3]
    error_norm = np.linalg.norm(error_xy, axis=1)
    time_offset = pose_times[nearest_pose_indices] - gnss[:, 0]
    elapsed_time = gnss[:, 0] - pose_times[0]

    rmse = np.sqrt(np.mean(error_norm**2))

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].scatter(elapsed_time, error_norm, s=14, color="steelblue")
    axes[0].axhline(rmse, color="black", ls="--", lw=0.9, label=f"RMSE: {rmse:.2f} m")
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title("GNSS vs. Nearest Pose Error")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, lw=0.4)

    axes[1].scatter(elapsed_time, error_xy[:, 0], s=14, color="steelblue", label="x error")
    axes[1].scatter(elapsed_time, error_xy[:, 1], s=14, color="tomato", label="y error")
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel("Component error (m)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, lw=0.4)

    axes[2].scatter(elapsed_time, time_offset, s=14, color="dimgray")
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_xlabel("Time since start (s)")
    axes[2].set_ylabel("Time offset (s)")
    axes[2].grid(True, lw=0.4)

    fig.tight_layout()
    return fig


def _measurement_points_world(pose: np.ndarray, measurements: np.ndarray) -> np.ndarray:
    """Transform range-bearing measurements to world-frame endpoints."""
    measurements = np.asarray(measurements, dtype=float).reshape(-1, 2)
    if len(measurements) == 0:
        return np.empty((0, 2))

    ranges = measurements[:, 0]
    bearings = measurements[:, 1]
    local_points = np.column_stack((ranges * np.cos(bearings), ranges * np.sin(bearings)))
    return pose[:2] + local_points @ rotmat2(pose[2]).T


def _bearing_range_plot_points(measurements: np.ndarray) -> np.ndarray:
    """Return points as [bearing deg, range] for innovation-space plots."""
    measurements = np.asarray(measurements, dtype=float).reshape(-1, 2)
    if len(measurements) == 0:
        return np.empty((0, 2))
    return np.column_stack((np.rad2deg(measurements[:, 1]), measurements[:, 0]))


def compute_association_statistics(
    diagnostics: dict[str, np.ndarray],
    alpha_individual: float = 0.999,
    alpha_joint: float = 0.9999,
) -> dict[str, float | int | np.ndarray]:
    """Compute derived association statistics from raw saved association data."""
    measurements   = diagnostics.get("measurements")
    predicted      = diagnostics.get("predicted_measurements")
    association    = diagnostics.get("association")
    innovation_cov = diagnostics.get("innovation_covariance")
    
    n_associated = int(np.sum(association >= 0))
    dof = 2 * n_associated

    individual_compatibility = individualCompatibility(measurements, predicted, innovation_cov)
    individual_gate = float(chi2.isf(1 - alpha_individual, 2))

    if dof == 0:
        joint_nis = np.nan
        joint_expected = np.nan
        joint_upper = np.nan
        joint_lower = np.nan
        joint_per_dof = np.nan
    else:
        joint_nis      = float(NIS(measurements, predicted, innovation_cov, association))
        joint_expected = float(dof)
        joint_upper    = float(chi2.isf(1 - alpha_joint, dof))
        joint_lower    = float(chi2.isf(alpha_joint, dof))
        joint_per_dof  = joint_nis / dof

    return {
        "individual_compatibility": individual_compatibility,
        "individual_gate": individual_gate,
        "joint_nis": joint_nis,
        "joint_nis_dof": dof,
        "joint_nis_expected": joint_expected,
        "joint_nis_upper": joint_upper,
        "joint_nis_lower": joint_lower,
        "joint_nis_per_dof": joint_per_dof,
    }


def plot_association(
    diagnostics: dict[str, np.ndarray],
    show_covariances: bool = True,
    alpha_individual: float = 0.999,
    alpha_joint: float = 0.9999,
) -> plt.Figure:
    """Plot one saved association diagnostic in world space and innovation space."""
    scan_step = int(diagnostics["scan_step"][0])
    
    pose            = diagnostics.get("pose")
    measurements    = diagnostics.get("measurements")
    predicted       = diagnostics.get("predicted_measurements")
    association     = diagnostics.get("association")
    local_landmarks = diagnostics.get("local_landmarks")
    innovation_cov  = diagnostics.get("innovation_covariance")
    
    stats = compute_association_statistics(
        diagnostics,
        alpha_individual=alpha_individual,
        alpha_joint=alpha_joint,
    )

    measurement_world = _measurement_points_world(pose, measurements)
    measurement_br = _bearing_range_plot_points(measurements)
    predicted_br = _bearing_range_plot_points(predicted)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    world_ax, innovation_ax = axes

    world_ax.scatter(pose[0], pose[1], c="black", marker="o", s=45, label="Robot")
    heading = rotmat2(pose[2]) @ np.array([2.0, 0.0])
    world_ax.arrow(
        pose[0],
        pose[1],
        heading[0],
        heading[1],
        width=0.05,
        head_width=0.5,
        length_includes_head=True,
        color="black",
        alpha=0.8,
    )

    if len(local_landmarks) > 0:
        world_ax.scatter(
            local_landmarks[:, 0],
            local_landmarks[:, 1],
            c="tomato",
            marker="x",
            s=45,
            label="Predicted landmarks",
        )

    if len(measurement_world) > 0:
        is_associated = association >= 0
        world_ax.scatter(
            measurement_world[~is_associated, 0],
            measurement_world[~is_associated, 1],
            c="dimgray",
            marker=".",
            s=45,
            label="Unassociated measurements",
        )
        world_ax.scatter(
            measurement_world[is_associated, 0],
            measurement_world[is_associated, 1],
            c="steelblue",
            marker=".",
            s=55,
            label="Associated measurements",
        )

        for i, predicted_idx in enumerate(association):
            if predicted_idx < 0:
                continue
            world_ax.plot(
                [measurement_world[i, 0], local_landmarks[predicted_idx, 0]],
                [measurement_world[i, 1], local_landmarks[predicted_idx, 1]],
                color="seagreen",
                lw=0.9,
                alpha=0.7,
            )

    world_ax.set_title(f"Association Result, Scan {scan_step}")
    world_ax.set_xlabel("X (m)")
    world_ax.set_ylabel("Y (m)")
    world_ax.set_aspect("equal")
    world_ax.grid(True, lw=0.4)
    world_ax.legend(fontsize=8)

    if len(predicted_br) > 0:
        innovation_ax.scatter(
            predicted_br[:, 0],
            predicted_br[:, 1],
            c="tomato",
            marker="x",
            s=45,
            label="Predicted",
        )

    if len(measurement_br) > 0:
        is_associated = association >= 0
        innovation_ax.scatter(
            measurement_br[~is_associated, 0],
            measurement_br[~is_associated, 1],
            c="dimgray",
            marker=".",
            s=45,
            label="Unassociated",
        )
        innovation_ax.scatter(
            measurement_br[is_associated, 0],
            measurement_br[is_associated, 1],
            c="steelblue",
            marker=".",
            s=55,
            label="Associated",
        )

        for i, predicted_idx in enumerate(association):
            if predicted_idx < 0:
                continue
            innovation_ax.plot(
                [measurement_br[i, 0], predicted_br[predicted_idx, 0]],
                [measurement_br[i, 1], predicted_br[predicted_idx, 1]],
                color="seagreen",
                lw=0.9,
                alpha=0.7,
            )

    if show_covariances and innovation_cov.shape == (2 * len(predicted), 2 * len(predicted)):
        transform = np.array([[0.0, 180.0 / np.pi], [1.0, 0.0]])
        for j, center in enumerate(predicted_br):
            cov_rb = innovation_cov[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
            cov_br = transform @ cov_rb @ transform.T
            innovation_ax.add_patch(
                confidence_ellipse_2d(
                    center,
                    cov_br,
                    fc="none",
                    ec="tomato",
                    alpha=0.35,
                    lw=0.7,
                )
            )

    joint_nis = float(stats["joint_nis"])
    joint_expected = float(stats["joint_nis_expected"])
    joint_upper = float(stats["joint_nis_upper"])
    nis_title = "Innovation Space"
    if np.isfinite(joint_nis):
        nis_title += f" (NIS {joint_nis:.1f}, E {joint_expected:.1f}, upper {joint_upper:.1f})"

    innovation_ax.set_title(nis_title)
    innovation_ax.set_xlabel("Bearing (deg)")
    innovation_ax.set_ylabel("Range (m)")
    innovation_ax.grid(True, lw=0.4)
    handles, _ = innovation_ax.get_legend_handles_labels()
    if handles:
        innovation_ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_association_nis(
    diagnostics: list[dict[str, np.ndarray]],
    alpha_individual: float = 0.999,
    alpha_joint: float = 0.9999,
) -> plt.Figure:
    """Plot joint association NIS over saved diagnostic scans."""
    scan_steps = np.array([int(item["scan_step"][0]) for item in diagnostics], dtype=int) 
    
    stats = [
        compute_association_statistics(
            item,
            alpha_individual=alpha_individual,
            alpha_joint=alpha_joint,
        )
        for item in diagnostics
    ]
    joint_nis = np.array([float(item["joint_nis"]) for item in stats], dtype=float)
    expected = np.array([float(item["joint_nis_expected"]) for item in stats], dtype=float)
    upper = np.array([float(item["joint_nis_upper"]) for item in stats], dtype=float)
    lower = np.array([float(item["joint_nis_lower"]) for item in stats], dtype=float)
    dof = np.array([int(item["joint_nis_dof"]) for item in stats], dtype=int)
    valid = (dof > 0) & np.isfinite(joint_nis)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    if np.any(valid):   
        axes[0].plot(scan_steps[valid], joint_nis[valid], lw=1.2, color="steelblue", label="NIS")
        axes[0].plot(scan_steps[valid], expected[valid], lw=1.0, ls=":", color="tomato", label="E[NIS] = dof")
        axes[0].plot(scan_steps[valid], upper[valid], lw=1.0, ls=":", color="green", label=r"$\chi^2_{{dof},\alpha_{joint}}$")
        axes[0].plot(scan_steps[valid], lower[valid], lw=1.0, ls=":", color="orange", label=r"$\chi^2_{{dof},1-\alpha_{joint}}$")

        axes[1].plot(scan_steps[valid], joint_nis[valid] / dof[valid], lw=1.2, color="steelblue", label="NIS / dof")
        axes[1].axhline(1.0, lw=1.0, ls=":", color="tomato", label="E[NIS / dof] = 1")
        axes[1].plot(scan_steps[valid], upper[valid] / dof[valid], lw=1.0, ls=":", color="green", label=r"$\chi^2_{{dof},\alpha_{joint}}$ / dof")
        axes[1].plot(scan_steps[valid], lower[valid] / dof[valid], lw=1.0, ls=":", color="orange", label=r"$\chi^2_{{dof},1-\alpha_{joint}}$ / dof")
    else:
        axes[0].text(0.5, 0.5, "No associated measurements in saved diagnostics", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No normalized NIS values", ha="center", va="center")

    axes[0].set_ylabel("NIS")
    axes[0].set_title("Joint NIS")
    if np.any(valid):
        axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, lw=0.4)

    axes[1].set_xlabel("Scan step")
    axes[1].set_ylabel("NIS / DOF")
    axes[1].set_title("Joint NIS per Degree of Freedom")
    if np.any(valid):
        axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, lw=0.4)

    fig.tight_layout()
    return fig


@dataclass
class SlamRunPlotter:
    """Utility class for plotting SLAM runs from saved logs and snapshots."""
    run_dir: Path
    steps: dict[str, np.ndarray]
    snapshots: list[dict[str, np.ndarray]]
    association: list[dict[str, np.ndarray]]
    config: SlamConfig
    
    @classmethod
    def from_run(cls, run_dir: Path | str) -> SlamRunPlotter:
        steps       = SlamLogger.load_steps(run_dir)
        snapshots   = SlamLogger.load_all_snapshots(run_dir)
        association = SlamLogger.load_all_association_diagnostics(run_dir)
        config      = SlamConfig.load(run_dir / "config.yaml")
        
        return cls(
            run_dir=run_dir,
            steps=steps,
            snapshots=snapshots,
            association=association,
            config=config
        )
    
    @property
    def figure_dir(self) -> Path:
        path = self.run_dir / "figures"
        path.mkdir(exist_ok=True)
        return path
    
    def _finish_figure(self, fig, name: str, save: bool, show: bool = True, fmt: str = "pdf") -> None:
        fig.tight_layout()
        if save:
            path = self.figure_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
        if not show:
            plt.close(fig)

    
    def plot_final_snapshot(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_estimate(
            poses         = self.snapshots[-1].get("poses"),
            poses_cov     = self.snapshots[-1].get("poses_covariance"),
            landmarks     = self.snapshots[-1].get("landmarks"),
            landmarks_cov = self.snapshots[-1].get("landmarks_covariance"), 
            gnss = VictoriaParkLoader().gnss_filtered,
            poses_cov_stride = 20
        )
        self._finish_figure(fig, "estimate", save, fmt)
    
    def plot_timing_breakdown(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_timing_breakdown(
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            t_assoc = self.steps.get("duration_association"),
            t_opt   = self.steps.get("duration_optimization"),
            t_total = self.steps.get("duration_update"),
        )
        self._finish_figure(fig, "timing_breakdown", save, fmt)
    
    def plot_timing_over_time(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_timing_over_time(
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction") + self.steps.get("duration_association") + self.steps.get("duration_optimization"),
            n_local = self.steps.get("num_local_landmarks"),
        )
        self._finish_figure(fig, "timing_over_time", save, fmt)
    
    def plot_timing_vs_landmarks(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_timing_vs_landmarks(
            n_local = self.steps.get("num_local_landmarks"),
            t_cov = self.steps.get("duration_covariance_extraction") + self.steps.get("duration_association") + self.steps.get("duration_optimization"),
        )
        self._finish_figure(fig, "timing_vs_landmarks", save, fmt)
    
    def plot_landmark_growth(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_landmark_growth(
            steps       = self.steps.get("scan_step"),
            n_landmarks = self.steps.get("num_landmarks"),
        )
        self._finish_figure(fig, "landmark_growth", save, fmt)
    
    def plot_error_over_time(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_error(
            steps     = self.steps.get("scan_step"),
            error     = self.steps.get("factor_graph_error"),
            n_factors = self.steps.get("num_factors"),
        )
        self._finish_figure(fig, "error_over_time", save, fmt)
    
    def plot_gnss_pose_error(self, save: bool = True, fmt: str = "pdf") -> None:
        fig = plot_gnss_pose_error(
            gnss=VictoriaParkLoader().gnss_filtered,
            poses=self.snapshots[-1].get("poses"),
            pose_times=self.steps.get("scan_time"),
        )
        self._finish_figure(fig, "gnss_pose_error", save, fmt)
    
    def plot_association_nis(self, save: bool = True, fmt: str = "pdf") -> None:
        if not self.association:
            print("No association diagnostics found, skipping NIS plot.")
            return
        
        fig = plot_association_nis(
            diagnostics=self.association,
            alpha_individual=self.config.association.alpha_individual,
            alpha_joint=self.config.association.alpha_joint,
        )
        self._finish_figure(fig, "association_nis", save, fmt)
    
    def plot_association(self, save: bool = True, fmt: str = "pdf") -> None:
        if not self.association:
            print("No association diagnostics found, skipping association plot.")
            return
        
        for diag in self.association:
            scan_step = int(diag["scan_step"][0])
            fig = plot_association(
                diagnostics=diag,
                show_covariances=True,
                alpha_individual=self.config.association.alpha_individual,
                alpha_joint=self.config.association.alpha_joint,
            )
            self._finish_figure(fig, f"association_scan_{scan_step:03d}", save, show=False)

    
    def plot_all(self, save: bool = True, show: bool = True):
        self.plot_final_snapshot(save=save)
        self.plot_timing_breakdown(save=save)
        self.plot_timing_over_time(save=save)
        self.plot_timing_vs_landmarks(save=save)
        self.plot_landmark_growth(save=save)
        self.plot_error_over_time(save=save)  
        self.plot_gnss_pose_error(save=save)
        self.plot_association_nis(save=save) 
        # self.plot_association(save=False)
        plt.show()




def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved SLAM run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--gnss", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--alpha-individual", type=float, default=0.999)
    parser.add_argument("--alpha-joint", type=float, default=0.9999)
    args = parser.parse_args()

    run_dir = args.run_dir
    plotter = SlamRunPlotter.from_run(run_dir)
    plotter.plot_all(save=True, show=args.show)
    

if __name__ == "__main__":
    plotter = SlamRunPlotter.from_run(Path('runs/20260521_203732_all'))
    plotter.plot_all(save=False, show=True)

    # main()
    
