from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2, spearmanr

from master_code.data_association import NIS, individualCompatibility
from master_code.plotting.thesis_style import thesis_figsize
from master_code.utils import rotmat2, ssa


def _ensure_ax(
    ax: plt.Axes | np.ndarray | None,
    figsize: tuple[float, float],
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kwargs,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    if ax is not None:
        fig = ax.figure if isinstance(ax, plt.Axes) else ax.flat[0].figure
        return fig, ax
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True, **subplot_kwargs)
    return fig, ax


def confidence_ellipse_2d(center: np.ndarray, cov: np.ndarray, scale: float = 1.0, **kwargs) -> Ellipse:
    """Draw a 95% confidence ellipse for a 2D covariance matrix."""
    chi2_95_2d: float = 2.447746830681

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width = np.sqrt(eigvals[0]) * 2 * chi2_95_2d * scale
    height = np.sqrt(eigvals[1]) * 2 * chi2_95_2d * scale

    return Ellipse(xy=tuple(center), width=width, height=height, angle=np.degrees(angle), **kwargs)

def draw_pose_covariances(ax, poses: np.ndarray, poses_cov: np.ndarray, cov_stride: int = 10, **kwargs) -> None:
    R_W_B = rotmat2(poses[:, 2])
    B_xy_cov = poses_cov[:, :2, :2]
    W_xy_covs = R_W_B @ B_xy_cov @ R_W_B.transpose((0, 2, 1))  # Rotate to world frame
    
    for pose, W_xy_cov in zip(poses[::cov_stride], W_xy_covs[::cov_stride]):
        ax.add_patch(
            confidence_ellipse_2d(pose[:2], W_xy_cov, fc="steelblue", alpha=0.3, ec="steelblue", **kwargs)
        )

def draw_landmark_covariances(ax, landmarks: np.ndarray, landmarks_cov: np.ndarray, alpha: float, **kwargs) -> None:
    for lm, cov in zip(landmarks, landmarks_cov):
        ax.add_patch(
            confidence_ellipse_2d(lm, cov, fc="tomato", alpha=alpha, ec="tomato", **kwargs)
        )


def pose2_tangent_errors(poses: np.ndarray, poses_gt: np.ndarray) -> np.ndarray:
    """Return Logmap(T_est.between(T_gt)) for aligned Pose2 arrays."""
    import gtsam

    errors = np.zeros_like(poses, dtype=float)
    for i, (pose, pose_gt) in enumerate(zip(poses, poses_gt)):
        T = gtsam.Pose2(*pose)
        T_gt = gtsam.Pose2(*pose_gt)
        errors[i, :] = gtsam.Pose2.Logmap(T.between(T_gt))
    return errors

def plot_estimate(
    poses: np.ndarray | None = None,
    poses_cov: np.ndarray | None = None,
    poses_cov_stride: int = 10,
    landmarks: np.ndarray | None = None,
    landmarks_cov: np.ndarray | None = None,
    gnss: np.ndarray | None = None,
    poses_gt: np.ndarray | None = None,
    landmarks_gt: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the trajectory, landmarks, and optional reference data."""
    
    fig, ax = _ensure_ax(ax, figsize=(8, 8))

    kwargs.setdefault("show_legend", True)
    kwargs.setdefault("show_grid", True)
    kwargs.setdefault("equal_aspect", True)
    kwargs.setdefault("title", "MAP Estimate")
    kwargs.setdefault("traj_label", "Trajectory")
    kwargs.setdefault("x_label", "x [m]")
    kwargs.setdefault("y_label", "y [m]")

    if poses_gt is not None and len(poses_gt) > 0:
        ax.plot(poses_gt[:, 0], poses_gt[:, 1], color="black", linestyle="--", alpha=0.8, label="Ground truth trajectory", zorder=1)

    if landmarks_gt is not None and len(landmarks_gt) > 0:
        ax.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], c="0.25", marker="x", s=18, alpha=0.7, label=f"Ground truth landmarks ({len(landmarks_gt)})",zorder=2)

    if poses is not None and len(poses) > 0:
        ax.plot(poses[:, 0], poses[:, 1], color="steelblue", label=kwargs.get("traj_label"), zorder=4)
        ax.scatter(poses[0, 0], poses[0, 1], color="green", s=50, zorder=5, label="Start", marker="o", facecolors="white", edgecolors="green")
        ax.scatter(poses[-1, 0], poses[-1, 1], color="red", s=50, zorder=5, label="End", marker="s", facecolors="white", edgecolors="red")

        if poses_cov is not None:
            draw_pose_covariances(ax, poses, poses_cov, cov_stride=poses_cov_stride, zorder=2)

    if landmarks is not None and len(landmarks) > 0:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c="tomato", marker=".", label=f"Landmarks ({len(landmarks)})", zorder=3)

        if landmarks_cov is not None:
            draw_landmark_covariances(ax, landmarks, landmarks_cov, alpha=0.3, zorder=3)

    if gnss is not None and len(gnss) > 0:
        ax.scatter(gnss[:, 1], gnss[:, 2], c="gold", marker=".", alpha=0.6, label="GNSS", zorder=1)

    ax.set_title(kwargs["title"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    if kwargs["show_grid"]:   
        ax.grid(lw=0.4, alpha=0.7)
    if kwargs["equal_aspect"]:
        ax.set_aspect("equal", adjustable="box") 
    if kwargs["show_legend"]:
        ax.legend()

    return fig, ax

def plot_cumulative_timing(
    steps: np.ndarray,
    t_cov: np.ndarray,
    t_assoc: np.ndarray,
    t_opt: np.ndarray,
    t_step: np.ndarray,  
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-step and cumulative processing time breakdown."""
    fig, ax = _ensure_ax(ax, figsize=(6, 3), sharex=True)

    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_step  = np.nan_to_num(t_step,  nan=0.0)
    t_other = np.maximum(t_step - (t_cov + t_assoc + t_opt), 0.0)

    total_step = float(np.sum(t_step))
    total_cov = float(np.sum(t_cov))
    total_assoc = float(np.sum(t_assoc))
    total_opt = float(np.sum(t_opt))
    print(
        f"Total processing times [s] -> "
        f"step: {total_step:.3f}, cov: {total_cov:.3f}, assoc: {total_assoc:.3f}, opt: {total_opt:.3f}"
    )

    t_cov_cum = np.cumsum(t_cov)
    t_assoc_cum = np.cumsum(t_assoc)
    t_opt_cum = np.cumsum(t_opt)
    t_other_cum = np.cumsum(t_other)

    labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
    parts = [t_cov_cum, t_assoc_cum, t_opt_cum, t_other_cum]

    parts = [p for p in parts]
    ax.stackplot(steps, *parts, labels=labels)
    ax.set_ylabel("Time (S)")
    ax.set_title(f"Cumulative Processing Time (Total: {total_step:.2f} s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, lw=0.4)

    return fig, ax
        
        
def plot_per_step_timing(
    steps: np.ndarray,
    t_cov: np.ndarray,
    t_assoc: np.ndarray,
    t_opt: np.ndarray,
    t_step: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-step and cumulative processing time breakdown."""
    fig, ax = _ensure_ax(ax, figsize=(6, 2.5))

    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_step  = np.nan_to_num(t_step,  nan=0.0)
    
    t_other = np.maximum(t_step - (t_cov + t_assoc + t_opt), 0.0)

    labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
    parts = [t_cov, t_assoc, t_opt, t_other]

    parts_ms = [1000*p for p in parts]
    ax.stackplot(steps, *parts_ms, labels=labels)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-step Processing Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, lw=0.4)

    # Print per-step processing time statistics
    per_step_ms = t_step * 1000
    n_above_214 = int(np.sum(per_step_ms > 214)) # 214 ms is time between lidar scans in VP
    share_above_214 = (n_above_214 / len(per_step_ms) * 100.0) if len(per_step_ms) else 0.0
    print(f"Per-step Processing Time - Max: {np.max(per_step_ms):.2f} ms, "
        f"95th Quantile: {np.quantile(per_step_ms, 0.95):.2f} ms, "
        f"Average: {np.mean(per_step_ms):.2f} ms, "
        f"Median: {np.median(per_step_ms):.2f} ms, "
        f"Steps > 214 ms: {n_above_214} ({share_above_214:.2f}%)")

    return fig, ax


def plot_landmarks_and_timing(
    steps: np.ndarray,
    t_cov: np.ndarray,
    n_local: np.ndarray,
    n_support: np.ndarray,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot covariance recovery step time and in-view predicted landmark count."""

    fig, axes = _ensure_ax(axes, figsize=(6, 6), nrows=3, ncols=1, sharex=True)

    axes[0].plot(steps, t_cov, lw=0.8, color="tab:blue", label="Covariance recovery time")
    axes[0].set_ylabel("Time (s)")
    # axes[0].set_yscale("log")
    axes[0].grid(True, which="both", axis="both", lw=0.4)
    axes[0].legend(loc="upper left")
    axes[0].tick_params(axis="x", which="both", labelbottom=False)

    axes[1].plot(steps, n_local, lw=0.8, color="tab:orange", label="# In-view landmarks")
    axes[1].set_ylabel("# In-view landmarks")
    axes[1].grid(True, lw=0.4)
    axes[1].legend()

    axes[2].plot(steps, n_support, lw=0.8, color="tab:green", label="# Support cliques")
    axes[2].set_ylabel("# Support cliques")
    axes[2].set_xlabel("Scan step")
    axes[2].grid(True, lw=0.4)
    axes[2].legend()

    return fig, axes


def plot_timing_vs_landmarks(
    n_local: np.ndarray,
    n_support: np.ndarray,
    t_cov: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot covariance recovery step time against in-view landmark count."""
    fig, axes = _ensure_ax(ax, figsize=(8, 4), nrows=1, ncols=2)

    t_cov_ms = 1000.0 * t_cov
    t_support_ms = 1000.0 * t_cov
    
    rho_support, _ = spearmanr(n_support, t_support_ms)
    rho_local, _ = spearmanr(n_local, t_cov_ms)

    axes[0].scatter(n_local, t_cov_ms, alpha=0.4, s=12, color="steelblue")
    axes[0].set_xlabel("# In-view landmarks")
    axes[0].set_ylabel("Cov recovery time (ms)")
    axes[0].set_title("Spearman rho: " + r"$\rho$" + f" = {rho_local:.2f}")
    axes[0].grid(True, lw=0.4)

    axes[1].scatter(n_support, t_support_ms, alpha=0.4, s=12, color="steelblue")
    axes[1].set_xlabel("# Support cliques")
    axes[1].set_ylabel("Cov recovery time (ms)")
    axes[1].set_title("Spearman rho: " + r"$\rho$" + f" = {rho_support:.2f}")
    axes[1].grid(True, lw=0.4)

    return fig, axes


def plot_landmark_growth(
    steps: np.ndarray, 
    n_landmarks: np.ndarray,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot confirmed landmark count over time."""
    fig, ax = _ensure_ax(ax, figsize=(8, 2.5))

    ax.plot(steps, n_landmarks, **kwargs)
    ax.set_xlabel("Scan step")
    ax.set_ylabel("# landmarks")
    ax.set_title("Landmark Count Over Time")
    ax.grid(True, lw=0.4)
    return fig, ax 



def plot_pose_diagnostics(
    poses: np.ndarray | None = None,
    poses_gt: np.ndarray | None = None,
    poses_covs: np.ndarray | None = None,
    axes: plt.Axes | None = None,
    cov_frame: Literal["body", "world"] = "body",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot pose-component errors and/or pose covariance sigma envelopes.

    This function plots x, y, and heading diagnostics over time.

    If poses and poses_gt are provided, component errors are plotted. For
    cov_frame == "body", the error is the Pose2 tangent vector
    Logmap(T_est.between(T_gt)); for cov_frame == "world", x/y are direct
    world-frame component differences and yaw is wrapped.

    If poses_covs is provided, marginal 1, 2, and 3 sigma covariance envelopes
    are plotted.

    If cov_frame == "body", the xy covariance is used as-is.
    If cov_frame == "world", poses must be provided, and the xy covariance is
    rotated into the world frame before extracting x/y standard deviations.
    """

    fig, axes = _ensure_ax(axes, figsize=(8, 5.2), nrows=3, ncols=1, sharex=True)

    
    if poses is None and poses_gt is None and poses_covs is None:
        raise ValueError("At least one diagnostic source must be provided.")

    plot_error = poses is not None and poses_gt is not None
    plot_cov = poses_covs is not None

    if poses_gt is not None and poses is None:
        raise ValueError("poses_gt was provided, but poses is None.")

    if cov_frame == "world" and poses_covs is not None and poses is None:
        raise ValueError("poses must be provided when cov_frame='world'.")

    # Determine common length
    lengths = []
    if poses is not None:
        lengths.append(len(poses))
    if poses_gt is not None:
        lengths.append(len(poses_gt))
    if poses_covs is not None:
        lengths.append(len(poses_covs))


    n = min(lengths)
    steps = np.arange(n)  # Assuming poses, poses_gt, and poses_covs are all aligned in time and have the same length. Adjust if needed.

    if poses is not None:
        poses = poses[:n]
    if poses_gt is not None:
        poses_gt = poses_gt[:n]
    if poses_covs is not None:
        poses_covs = poses_covs[:n]


    labels = ["x [m]", "y [m]", r"$\theta$ [deg]"]

    for ax, ylabel in zip(axes, labels):
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4, alpha=0.7)

    # -------------------------------------------------------------------------
    # Covariance envelopes
    # -------------------------------------------------------------------------
    if plot_cov:
        xy_covs = poses_covs[:, :2, :2]

        if cov_frame == "world":
            R = rotmat2(poses[:, 2])
            xy_covs = R @ xy_covs @ R.transpose((0, 2, 1))

        sigmas_xy = np.sqrt(
            np.maximum(np.diagonal(xy_covs, axis1=1, axis2=2), 0.0)
        )
        sigma_x = sigmas_xy[:, 0]
        sigma_y = sigmas_xy[:, 1]
        sigma_theta = np.rad2deg(
            np.sqrt(np.maximum(poses_covs[:, 2, 2], 0.0))
        )

        sigma_series = [sigma_x, sigma_y, sigma_theta]

        colors = {1: "steelblue", 2: "seagreen", 3: "tomato"}
        alphas = {1: 0.28, 2: 0.18, 3: 0.10}
        sigma_handles = []
        sigma_labels = []

        for ax_idx, (ax, sigma) in enumerate(zip(axes, sigma_series)):
            for k in (3, 2, 1):
                bound = k * sigma

                label = rf"$\pm {k}\sigma$" if ax_idx == 0 else None
                fill = ax.fill_between(
                    steps,
                    -bound,
                    bound,
                    color=colors[k],
                    alpha=alphas[k],
                    label=label,
                    linewidth=0.0,
                )
                ax.plot(bound, color=colors[k], lw=0.7, alpha=0.5)
                ax.plot(-bound, color=colors[k], lw=0.7, alpha=0.5)

                if ax_idx == 0:
                    sigma_handles.append(fill)
                    sigma_labels.append(rf"$\pm {k}\sigma$")

    # -------------------------------------------------------------------------
    # Component errors
    # -------------------------------------------------------------------------
    if plot_error:
        if cov_frame == "body":
            error = pose2_tangent_errors(poses, poses_gt)
        else:
            error = poses - poses_gt
            error[:, 2] = ssa(error[:, 2])
        
        x_error = error[:, 0]
        y_error = error[:, 1]
        yaw_error = error[:, 2]

        x_rmse = float(np.sqrt(np.mean(x_error**2)))
        y_rmse = float(np.sqrt(np.mean(y_error**2)))
        yaw_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(yaw_error**2))))

        error_series = [
            (x_error, f"x error, RMSE={x_rmse:.2f} m"),
            (y_error, f"y error, RMSE={y_rmse:.2f} m"),
            (np.rad2deg(yaw_error), f"yaw error, RMSE={yaw_rmse_deg:.2f} deg"),
        ]

        for ax, (err, label) in zip(axes, error_series):
            ax.plot(
                err,
                color="tab:orange",
                lw=0.9,
                label=label,
                zorder=10,
            )

    # -------------------------------------------------------------------------
    # Titles and legends
    # -------------------------------------------------------------------------
    if plot_cov and plot_error:
        title = f"Pose errors with {cov_frame}-frame covariance envelopes"
    elif plot_cov:
        title = f"{cov_frame.capitalize()}-frame pose covariance envelopes"
    else:
        title = "Pose component errors"

    axes[0].set_title(title)
    axes[-1].set_xlabel("Scan step")

    handles, labels = axes[0].get_legend_handles_labels()
    error_handles = [h for h, l in zip(handles, labels) if "error" in l.lower()]
    error_labels = [l for l in labels if "error" in l.lower()]
    if error_handles:
        error_legend = axes[0].legend(
            error_handles,
            error_labels,
            loc="upper left",
            fontsize=9,
        )
        axes[0].add_artist(error_legend)

    if plot_cov and sigma_handles:
        axes[0].legend(
            sigma_handles,
            sigma_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
            frameon=True,
            ncol=3,
        )

    for ax in axes[1:]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=9)

    return fig, axes


def plot_pose_nees(
    poses: np.ndarray,
    poses_gt: np.ndarray,
    poses_covs: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot NEES over time for pose estimates."""

    fig, ax = _ensure_ax(ax, figsize=(8, 3))

    n = len(poses)

    poses_gt = poses_gt[:n]

    poses_error = pose2_tangent_errors(poses, poses_gt)

    nees = np.einsum("ij,ijk,ik->i", poses_error, np.linalg.inv(poses_covs), poses_error)
    anees = np.sum(nees) / n

    upper_bound = chi2.ppf(0.95, 3) 
    lower_bound = chi2.ppf(0.05, 3)
    inlier_mask = (nees >= lower_bound) & (nees <= upper_bound)
    inlier_pct = 100.0 * np.mean(inlier_mask) 

    ax.plot(nees, color="steelblue", label=f"NEES (ANEES={anees:.2f})")
    ax.axhline(lower_bound, ls="--", c="tomato", label=r"$\chi^2_{3,0.95}$")
    ax.axhline(upper_bound, ls="--", c="orange", label=r"$\chi^2_{3,0.05}$")
    ax.set_xlabel("Scan step")
    ax.set_ylabel("NEES")
    ax.set_title(f"Pose NEES (Inliers: {inlier_pct:.1f}%)")
    ax.grid(True, lw=0.4)
    ax.legend(loc="upper center", ncol=3, frameon=True)

    return fig, ax


def plot_position_nis(
    gnss: np.ndarray,
    poses: np.ndarray,
    poses_covs: np.ndarray,
    poses_times: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot NIS between GNSS samples and nearest-in-time pose estimates."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    else:
        fig = ax.figure

    poses = poses
    poses_covs = poses_covs
 
    pose_xy = poses[:, :2]
    covs_xy = poses_covs[:, :2, :2]
    covs_xy = rotmat2(poses[:, 2]) @ covs_xy @ rotmat2(poses[:, 2]).transpose((0, 2, 1))  # Rotate to world frame

    if len(poses_times) != len(pose_xy):
        raise ValueError("pose_times must have the same length as poses.")
    if len(poses_times) == 0:
        raise ValueError("At least one pose is required.")
    if len(gnss) == 0:
        raise ValueError("At least one GNSS measurement is required.")

    nearest_pose_indices = nearest_indices(gnss[:, 0], poses_times)
    pose_xy = pose_xy[nearest_pose_indices]
    innovation_covs_xy = covs_xy[nearest_pose_indices] + np.eye(2) * (1.0**2)   # Hardcoding 1m GNSS sigma for NIS calculation

    innovation_xy = pose_xy - gnss[:, 1:3]
    nis_xy = np.einsum("ij,ijk,ik->i", innovation_xy, np.linalg.inv(innovation_covs_xy), innovation_xy)
    anis_xy = np.sum(nis_xy) / len(nis_xy)

    nis_lower = chi2.isf(1-0.05, 2)
    nis_upper = chi2.isf(1-0.95, 2)
    inliers = np.count_nonzero((nis_xy >= nis_lower) & (nis_xy <= nis_upper))
    inlier_frac = inliers / len(nis_xy)

    ax.scatter(nearest_pose_indices, nis_xy, s=5, color="steelblue", label=r"$\mathrm{NIS}_{xy}$")
    ax.axhline(nis_upper, ls="--", c="tomato", lw=1, label=r"$\chi^2_{2,0.95}$")
    ax.axhline(nis_lower, ls="--", c="orange", lw=1, label=r"$\chi^2_{2,0.05}$")
    ax.set_xlabel("Scan step")
    ax.set_ylabel(r"NIS")
    ax.set_title(f"NIS-position | ANIS = {anis_xy:.2f} | inliers = {inlier_frac:.0%}")
    ax.grid(True, lw=0.4)
    ax.legend()

    return fig, ax

def nearest_indices(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """Return index of the nearest reference time for each query time."""
    insertion_indices = np.searchsorted(reference_times, query_times)
    right_indices = np.clip(insertion_indices, 0, len(reference_times) - 1)
    left_indices = np.clip(insertion_indices - 1, 0, len(reference_times) - 1)

    left_dt = np.abs(query_times - reference_times[left_indices])
    right_dt = np.abs(query_times - reference_times[right_indices])

    return np.where(left_dt <= right_dt, left_indices, right_indices)




def plot_association(
    diagnostics: dict[str, np.ndarray],
    show_covariances: bool = True,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one saved association diagnostic in world space and measurement space."""
    
    fig, axes = _ensure_ax(axes, figsize=(8, 6), nrows=1, ncols=2)
    world_ax, meas_ax = axes
    
    scan_step = int(diagnostics.get("scan_step")[0])
    
    pose      = diagnostics.get("pose")
    z         = diagnostics.get("measurements")
    z_pred    = diagnostics.get("predicted_measurements")
    assoc     = diagnostics.get("association")
    local_lms = diagnostics.get("local_landmarks")
    S         = diagnostics.get("innovation_covariance")
    
    z_world = pose[:2] + z[:, 0] * np.column_stack((np.cos(z[:, 1]), np.sin(z[:, 1]))) @ rotmat2(pose[2]).T
    z_deg = z.copy()
    z_pred_deg = z_pred.copy()
    
    z_deg[:, 1]  = np.rad2deg(z[:, 1])
    z_pred_deg[:, 1] = np.rad2deg(z_pred[:, 1])

    ### WORLD SPACE
    world_ax.scatter(pose[0], pose[1], c="black", marker="o", s=45, label="Robot")
    heading = rotmat2(pose[2]) @ np.array([2.0, 0.0])
    world_ax.arrow(pose[0], pose[1], heading[0], heading[1], width=0.05, head_width=0.5, length_includes_head=True, color="black", alpha=0.8)

    if len(local_lms) > 0:
        world_ax.scatter(local_lms[:, 0], local_lms[:, 1], c="tomato", marker="x", s=45, label="Predicted landmarks")

    if len(z_world) > 0:
        is_ass = assoc >= 0
        world_ax.scatter(z_world[~is_ass, 0], z_world[~is_ass, 1], c="dimgray", marker=".", s=45,label="Unassociated measurements")
        world_ax.scatter(z_world[is_ass, 0], z_world[is_ass, 1], c="steelblue", marker=".", s=55, label="Associated measurements")

        for i, ass_idx in enumerate(assoc):
            if ass_idx < 0:
                continue
            world_ax.plot([z_world[i, 0], local_lms[ass_idx, 0]], [z_world[i, 1], local_lms[ass_idx, 1]], color="seagreen", lw=0.9, alpha=0.7)

    world_ax.set_title(f"Association Result, Scan {scan_step}")
    world_ax.set_xlabel("X (m)")
    world_ax.set_ylabel("Y (m)")
    world_ax.set_aspect("equal")
    world_ax.grid(True, lw=0.4)
    world_ax.legend(fontsize=8)
   
    ### MEAUSREMENT SPACE
    if len(z_pred_deg) > 0:
        meas_ax.scatter(z_pred_deg[:, 0], z_pred_deg[:, 1], c="tomato", marker="x", s=45, label="Predicted")

    if len(z_deg) > 0:
        is_ass = assoc >= 0
        meas_ax.scatter(z_deg[~is_ass, 0], z_deg[~is_ass, 1], c="dimgray", marker=".", s=45, label="Unassociated")
        meas_ax.scatter(z_deg[is_ass, 0], z_deg[is_ass, 1], c="steelblue", marker=".", s=55, label="Associated")

        for i, ass_idx in enumerate(assoc):
            if ass_idx < 0:
                continue
            meas_ax.plot([z_deg[i, 0], z_pred_deg[ass_idx, 0]], [z_deg[i, 1], z_pred_deg[ass_idx, 1]], color="seagreen", lw=0.9, alpha=0.7)

    if show_covariances and S.shape == (2 * len(z_pred), 2 * len(z_pred)):
        transform = np.array([[0.0, 180.0 / np.pi], [1.0, 0.0]])
        for j, center in enumerate(z_pred_deg):
            cov_rb = S[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
            cov_br = transform @ cov_rb @ transform.T
            meas_ax.add_patch(confidence_ellipse_2d(center, cov_br, fc="none", ec="tomato", alpha=0.35, lw=0.7))

    meas_ax.set_title("Measurement Space")
    meas_ax.set_xlabel("Bearing (deg)")
    meas_ax.set_ylabel("Range (m)")
    meas_ax.grid(True, lw=0.4)
    handles, _ = meas_ax.get_legend_handles_labels()
    if handles:
        meas_ax.legend(fontsize=8)

    return fig, axes


def plot_landmark_nis(
    diagnostics: list[dict[str, np.ndarray]],
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot landmark measurement NIS per degree of freedom."""

    fig, ax = _ensure_ax(ax, figsize=(8, 3))

    num_steps = len(diagnostics)

    nis = np.empty(num_steps, dtype=float)
    dofs = np.empty(num_steps, dtype=int)
    steps = np.empty(num_steps, dtype=int)

    for i, step_diag in enumerate(diagnostics):
        step   = step_diag.get("scan_step")[0]
        z      = step_diag.get("measurements")
        z_pred = step_diag.get("predicted_measurements")
        assoc  = step_diag.get("association")
        S      = step_diag.get("innovation_covariance")

        nis[i] = NIS(z, z_pred, S, assoc)
        dofs[i] = 2 * np.sum(assoc >= 0)
        steps[i] = step

    # Remove scans with no associated range-bearing measurements
    valid = dofs > 0
    nis = nis[valid]
    dofs = dofs[valid]
    steps = steps[valid]

    # Per-step NIS per degree of freedom
    nis_per_dof = nis / dofs

    nis_lower = chi2.ppf(0.025, dofs) / dofs
    nis_upper = chi2.ppf(0.975, dofs) / dofs

    # Aggregate NIS per degree of freedom
    D = np.sum(dofs)
    anis_per_dof = np.sum(nis) / D

    anis_lower = chi2.ppf(0.025, D) / D
    anis_upper = chi2.ppf(0.975, D) / D

    ax.plot(steps, nis_per_dof, lw=1.2, color="steelblue", label=r"$\bar{\epsilon}_{k,\mathrm{lm}}$")
    ax.axhline(1.0, lw=1.0, ls=":", color="tomato", label=r"$\mathbb{E}[\bar{\epsilon}_{k,\mathrm{lm}}]=1$")
    ax.plot(steps, nis_upper, lw=1.0, ls=":", color="tab:green", label=r"$\chi^2_{d_{k,\mathrm{lm}},\,0.975}/d_{k,\mathrm{lm}}$")
    ax.plot(steps, nis_lower, lw=1.0, ls=":", color="orange", label=r"$\chi^2_{d_{k,\mathrm{lm}},\,0.025}/d_{k,\mathrm{lm}}$")
    anis_text = (
        rf"$\bar{{\epsilon}}_{{\mathrm{{ANIS,lm}}}} = {anis_per_dof:.2f}$"
        "\n"
        rf"$95\%$ interval: $[{anis_lower:.2f},\,{anis_upper:.2f}]$"
    )
    ax.text(0.98, 0.95, anis_text, transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_title(r"Landmark measurement NIS per degree of freedom")
    ax.set_xlabel(r"Scan step $k$")
    ax.set_ylabel(r"$\bar{\epsilon}_{k,\mathrm{lm}}$")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, lw=0.4)

    return fig, ax
