import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from master_code.association import NIS, individualCompatibility
from master_code.plotting.thesis_style import thesis_figsize
from master_code.utils import rotmat2, ssa


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

def draw_pose_covariances(ax, poses: np.ndarray, poses_cov: np.ndarray, cov_stride: int = 10, **kwargs) -> None:
    for pose, cov in zip(poses[::cov_stride], poses_cov[::cov_stride]):
        # Rotate the translational covariance ellipse to world frame
        R = rotmat2(pose[2])
        xy_cov = R @ cov[:2, :2] @ R.T
        
        ax.add_patch(
            confidence_ellipse_2d(pose[:2], xy_cov, fc="steelblue", alpha=0.3, ec="steelblue", **kwargs)
        )

def draw_landmark_covariances(ax, landmarks: np.ndarray, landmarks_cov: np.ndarray, alpha: float, **kwargs) -> None:
    for lm, cov in zip(landmarks, landmarks_cov):
        ax.add_patch(
            confidence_ellipse_2d(lm, cov, fc="tomato", alpha=alpha, ec="tomato", **kwargs)
        )


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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    else:
        fig = ax.figure

    kwargs.setdefault("show_legend", True)
    kwargs.setdefault("show_grid", True)
    kwargs.setdefault("equal_aspect", True)
    kwargs.setdefault("title", "MAP Estimate")
    kwargs.setdefault("traj_label", "Trajectory")
    kwargs.setdefault("x_label", "x [m]")
    kwargs.setdefault("y_label", "y [m]")

    if poses_gt is not None and len(poses_gt) > 0:
        ax.plot(
            poses_gt[:, 0],
            poses_gt[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="Ground truth trajectory",
            zorder=1,
        )

    if landmarks_gt is not None and len(landmarks_gt) > 0:
        ax.scatter(
            landmarks_gt[:, 0],
            landmarks_gt[:, 1],
            c="0.25",
            marker="x",
            s=18,
            alpha=0.7,
            label=f"Ground truth landmarks ({len(landmarks_gt)})",
            zorder=2,
        )

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
    t_lmap: np.ndarray,
    t_total: np.ndarray,  
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-step and cumulative processing time breakdown."""
    if axes is None:
        fig, ax = plt.subplots(figsize=(8, 3), sharex=True, tight_layout=True)
    else:
        fig = ax.figure

    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_lmap  = np.nan_to_num(t_lmap,  nan=0.0)
    t_total = np.nan_to_num(t_total, nan=0.0)

    t_other = np.maximum(t_total - (t_cov + t_assoc + t_opt + t_lmap), 0.0)

    labels = ["Covariance extraction", "Association", "Optimisation", "Local Landmark Extraction", "Other"]
    parts = [t_cov, t_assoc, t_opt, t_lmap, t_other]

    parts_ms = [1000*p for p in parts]
    ax.stackplot(steps, *parts_ms, labels=labels)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-step Processing Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, lw=0.4)

    return fig, axes
        

def plot_timing_breakdown(
    steps: np.ndarray,
    t_cov: np.ndarray,
    t_assoc: np.ndarray,
    t_opt: np.ndarray,
    t_lmap: np.ndarray,
    t_total: np.ndarray,
    # t_tent: np.ndarray,
    # t_inno: np.ndarray,
    # t_scan: np.ndarray,
    
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-step and cumulative processing time breakdown."""
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True, tight_layout=True)
    else:
        fig = axes[0].figure

    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_lmap  = np.nan_to_num(t_lmap,  nan=0.0)
    t_total = np.nan_to_num(t_total, nan=0.0)
    # t_tent  = np.nan_to_num(t_tent,  nan=0.0)
    # t_inno  = np.nan_to_num(t_inno,  nan=0.0)
    # t_scan  = np.nan_to_num(t_scan,  nan=0.0)
    
    t_other = np.maximum(t_total - (t_cov + t_assoc + t_opt + t_lmap), 0.0)

    labels = ["Covariance extraction", "Association", "Optimisation", "Local Landmark Extraction", "Other"]
    parts = [t_cov, t_assoc, t_opt, t_lmap, t_other]

    parts_ms = [1000*p for p in parts]
    axes[0].stackplot(steps, *parts_ms, labels=labels)
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Per-step Processing Time")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, lw=0.4)

    axes[1].stackplot(steps, *[np.cumsum(p) for p in parts], labels=labels)
    axes[1].set_ylabel("Cumulative time (s)")
    axes[1].set_xlabel("Scan step")
    axes[1].set_title("Cumulative Processing Time")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, lw=0.4)

    return fig, axes


def plot_timing_over_time(
    steps: np.ndarray,
    t_cov: np.ndarray,
    n_local: np.ndarray,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot covariance recovery step time and in-view predicted landmark count."""
    cov_color = "tab:blue"
    local_color = "tab:orange"

    if axes is None:
        fig, axes = plt.subplots(2, 1,figsize=(8, 3), tight_layout=True)
    else:
        fig = axes[0].figure

    axes[0].plot(steps, t_cov, lw=0.8, color=cov_color, label="Covariance recovery time")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", axis="both", lw=0.4)
    axes[0].legend(loc="upper left")
    axes[0].tick_params(axis="x", which="both", labelbottom=False)

    axes[1].plot(steps, n_local, lw=0.8, color=local_color, label="# In-view landmarks")
    axes[1].set_ylabel("# In-view landmarks")
    axes[1].set_xlabel("Scan step")
    axes[1].grid(True, lw=0.4)
    axes[1].legend(loc="upper left")

    return fig, axes


def plot_timing_vs_landmarks(
    n_local: np.ndarray,
    t_cov: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot covariance recovery step time against in-view landmark count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    else:
        fig = ax.figure

    ax.scatter(n_local, t_cov, alpha=0.4, s=12, color="steelblue")
    ax.set_xlabel("# In-view landmarks")
    ax.set_ylabel("Cov recovery time (s)")
    ax.set_title("Covariance Recovery Time vs. In-view Landmarks")
    ax.grid(True, lw=0.4)

    return fig, ax 


def plot_landmark_growth(
    steps: np.ndarray, 
    n_landmarks: np.ndarray,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot confirmed landmark count over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), tight_layout=True)
    else:
        fig = ax.figure


    ax.plot(steps, n_landmarks, **kwargs)
    ax.set_xlabel("Scan step")
    ax.set_ylabel("# landmarks")
    ax.set_title("Landmark Count Over Time")
    ax.grid(True, lw=0.4)
    return fig, ax 


def plot_error(
    scan_steps: np.ndarray,
    error: np.ndarray,
    n_factors: np.ndarray,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the graph error over steps."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2.5), tight_layout=True)
    else:
        fig = ax.figure
    
    mask = ~np.isnan(error)
    
    n_factors_poses = scan_steps + 1 # (+1 for prior factor)
    n_factors_landmarks = n_factors - n_factors_poses
    n_scalar_factors = 3*n_factors_poses + 2*n_factors_landmarks

    # Average normalized factor errror (multiply by 2 first as the gtsam returns 1/2 sum_i r_i^T * Sigma_i^-1 * r_i)
    ANFE = 2*error[mask] / n_scalar_factors[mask] 
    
    kwargs.setdefault("lw", 1.2)
    kwargs.setdefault("color", "steelblue")
    ax.plot(scan_steps[mask], ANFE, **kwargs)
    ax.set_xlabel("Scan step")
    ax.set_ylabel("ANFE")
    ax.set_title("Average Normalized Factor Error")
    ax.grid(True, lw=0.4)

    return fig, ax

def plot_pose_covariance_evolution(
    covs: np.ndarray,
    poses: np.ndarray | None = None,
    steps: np.ndarray | None = None,
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot 1, 2, and 3 sigma pose covariance envelopes around zero.

    When poses are provided, the translational covariance is rotated to the
    world frame before extracting the x/y standard deviations.
    """
    covs = np.asarray(covs, dtype=float)
    if covs.ndim != 3 or covs.shape[1:] != (3, 3):
        raise ValueError("covs must have shape (N, 3, 3).")

    use_world_frame_xy = poses is not None
    if poses is not None:
        poses = np.asarray(poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError("poses must have shape (N, 3).")
        if len(poses) != len(covs):
            raise ValueError("poses must have the same length as covs.")

    if steps is None:
        steps = np.arange(len(covs))
    else:
        steps = np.asarray(steps)
        if len(steps) != len(covs):
            raise ValueError("steps must have the same length as covs.")

    if axes is None:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 4.8), sharex=True, tight_layout=True)
    else:
        fig = axes[0].figure

    xy_covs = covs[:, :2, :2]
    if poses is not None:
        xy_covs_world = np.empty_like(xy_covs)
        for i, (pose, xy_cov) in enumerate(zip(poses, xy_covs)):
            R = rotmat2(pose[2])
            xy_covs_world[i] = R @ xy_cov @ R.T
        xy_covs = xy_covs_world

    sigmas_xy = np.sqrt(np.maximum(np.diagonal(xy_covs, axis1=1, axis2=2), 0.0))
    sigma_x = sigmas_xy[:, 0]
    sigma_y = sigmas_xy[:, 1]
    sigma_theta = np.rad2deg(np.sqrt(np.maximum(covs[:, 2, 2], 0.0)))

    labels = ["x [m]", "y [m]", r"$\theta$ [deg]"]
    sigma_series = [sigma_x, sigma_y, sigma_theta]
    colors = {1: "steelblue", 2: "seagreen", 3: "tomato"}
    alphas = {1: 0.28, 2: 0.18, 3: 0.10}

    for ax, sigma, ylabel in zip(axes, sigma_series, labels):
        for k in (3, 2, 1):
            bound = k * sigma
            ax.fill_between(
                steps,
                -bound,
                bound,
                color=colors[k],
                alpha=alphas[k],
                label=rf"$\pm {k}\sigma$",
                linewidth=0.0,
            )
            ax.plot(steps, bound, color=colors[k], lw=0.7, alpha=0.9)
            ax.plot(steps, -bound, color=colors[k], lw=0.7, alpha=0.9)

        ax.axhline(0.0, color="black", lw=0.8, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4, alpha=0.7)

    title = "World-frame Pose Covariance Envelopes" if use_world_frame_xy else "Pose Covariance Envelopes"
    axes[0].set_title(title)
    axes[-1].set_xlabel("Scan step")
    axes[0].legend(loc="upper left", ncols=3, fontsize=8)

    return fig, axes


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
 
    pose_xy = poses[:, :2]
    covs_xy = poses_covs[:, :2, :2]

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
   
    ax.scatter(nearest_pose_indices, nis_xy, s=5, color="steelblue", label=r"$\mathrm{NIS}_{xy}$")
    ax.axhline(chi2.isf(1-0.95, 2), ls="--", c="tomato", lw=1, label=r"$\chi^2_{2,0.95}$")
    ax.axhline(chi2.isf(1-0.05, 2), ls="--", c="orange", lw=1, label=r"$\chi^2_{2,0.05}$")
    ax.set_xlabel("Scan step")
    ax.set_ylabel(r"NIS")
    ax.set_title(f"NIS-position | ANIS = {anis_xy:.2f}")
    ax.grid(True, lw=0.4)
    ax.legend()

    return fig, ax


def plot_trajectory_error(
    poses: np.ndarray,
    poses_gt: np.ndarray,
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot estimated trajectory error against simulated ground truth."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    poses_gt = np.asarray(poses_gt, dtype=float).reshape(-1, 3)

    if len(poses_gt) >= len(poses) + 1:
        poses_gt = poses_gt[1:len(poses) + 1]

    n = min(len(poses), len(poses_gt))
    if n == 0:
        raise ValueError("At least one estimated and ground-truth pose is required.")

    poses = poses[:n]
    poses_gt = poses_gt[:n]
    steps = np.arange(n)

    position_error = np.linalg.norm(poses[:, :2] - poses_gt[:, :2], axis=1)
    heading_error = ssa(poses[:, 2] - poses_gt[:, 2])
    position_rmse = float(np.sqrt(np.mean(position_error**2)))
    heading_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(heading_error**2))))

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, tight_layout=True)
    else:
        fig = axes[0].figure

    axes[0].plot(steps, position_error, color="steelblue", lw=0.9)
    axes[0].set_ylabel("Position error [m]")
    axes[0].set_title(f"Trajectory error | position RMSE = {position_rmse:.2f} m")
    axes[0].grid(True, lw=0.4)

    axes[1].plot(steps, np.rad2deg(heading_error), color="tomato", lw=0.9)
    axes[1].set_xlabel("Scan step")
    axes[1].set_ylabel("Heading error [deg]")
    axes[1].set_title(f"Heading RMSE = {heading_rmse_deg:.2f} deg")
    axes[1].grid(True, lw=0.4)

    return fig, axes


def nearest_indices(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """Return index of the nearest reference time for each query time."""
    insertion_indices = np.searchsorted(reference_times, query_times)
    right_indices = np.clip(insertion_indices, 0, len(reference_times) - 1)
    left_indices = np.clip(insertion_indices - 1, 0, len(reference_times) - 1)

    left_dt = np.abs(query_times - reference_times[left_indices])
    right_dt = np.abs(query_times - reference_times[right_indices])

    return np.where(left_dt <= right_dt, left_indices, right_indices)





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
) -> tuple[plt.Figure, plt.Axes]:
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

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
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
    return fig, axes


def plot_landmark_nis(
    diagnostics: list[dict[str, np.ndarray]],
    alpha_individual: float = 0.999,
    alpha_joint: float = 0.9999,
    axes : plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot joint association NIS over saved diagnostic scans."""
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    else:
        fig = axes[0].figure
    
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
    return fig, axes
