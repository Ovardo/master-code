from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import gtsam
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Ellipse

from config import SlamConfig
from data_loader import VictoriaParkLoader
from preprocessing import relative_pose

matplotlib.use('qtagg')


def covariance_ellipse_params(
    pose: gtsam.Pose2,
    cov: np.ndarray,
    scale: float = 1,
) -> tuple[tuple[float, float], float, float, float]:
    """Return center, width, height, and angle for the pose xy covariance ellipse."""
    k = 2.447746830681  # 95% confidence interval for 2 DOF

    R = pose.rotation().matrix()
    xy_cov = R @ cov[:2, :2] @ R.T
    eigvals, eigvecs = np.linalg.eigh(xy_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)  # clamp floating-point negatives
    eigvecs = eigvecs[:, order]

    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width = np.sqrt(eigvals[0]) * 2 * k * scale
    height = np.sqrt(eigvals[1]) * 2 * k * scale

    return tuple(pose.translation()), width, height, np.degrees(angle)



def propagate_dead_reckoning(frame_stride: int = 1):
    cfg = SlamConfig.load("dead_reckoning.yaml")
    loader = VictoriaParkLoader()
    
    odometry = loader.odometry
    dt = 0.025

    pose = gtsam.Pose2(*loader.initial_pose)
    cov = cfg.noise.init_pose_cov_matrix.copy()
    cov_odom = cfg.noise.odom_cov_matrix 

    poses = [pose]
    covs = [cov.copy()]
    times = [0.0]

    for i, odom in enumerate(odometry, start=1):
        t = odom[0]
        velocity = odom[1]
        steering = odom[2]
        
        odo, _ = relative_pose(velocity, steering, dt)
        
        H1 = np.zeros((3, 3), order="F")
        H2 = np.zeros((3, 3), order="F")
        
        pose = pose.compose(odo, H1, H2)
        cov = H1 @ cov @ H1.T + cov_odom 

        if i % frame_stride == 0:
            poses.append(pose)
            covs.append(cov.copy())
            times.append(t)

    return poses, covs, times


def animate_covariance(
    poses: list[gtsam.Pose2],
    covs: list[np.ndarray],
    times: list[float],
    output_path: Path,
    fps: int = 30,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = np.array([p.x() for p in poses])
    ys = np.array([p.y() for p in poses])
    ellipse_params = [covariance_ellipse_params(pose, cov) for pose, cov in zip(poses, covs)]

    max_radius = max(max(width, height) for _, width, height, _ in ellipse_params) / 2
    margin = max(10.0, max_radius * 1.1)

    fig, ax = plt.subplots(figsize=(8, 8))
    (path_line,) = ax.plot([], [], color="tab:blue", linewidth=1.8, label="Dead reckoning")
    (current_pose,) = ax.plot([], [], marker="o", color="tab:blue", markersize=4)
    ellipse = Ellipse(
        xy=ellipse_params[0][0],
        width=ellipse_params[0][1],
        height=ellipse_params[0][2],
        angle=ellipse_params[0][3],
        edgecolor="tab:red",
        facecolor="none",
        linewidth=1.5,
        label="Last pose covariance",
    )
    ax.add_patch(ellipse)
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax.set_title("Dead Reckoning: Covariance of Last Pose")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    def update(frame: int):
        center, width, height, angle = ellipse_params[frame]
        path_line.set_data(xs[: frame + 1], ys[: frame + 1])
        current_pose.set_data([xs[frame]], [ys[frame]])
        ellipse.center = center
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle
        time_text.set_text(f"t = {times[frame]:.2f} s")
        return path_line, current_pose, ellipse, time_text

    animation = FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    animation.save(output_path, writer=writer, dpi=150)
    plt.close(fig)


    

def main():
    loader = VictoriaParkLoader()
    gps = loader.gps
    
    poses, covs, times = propagate_dead_reckoning(frame_stride=25)
    
    # animate_covariance(poses, covs, times, Path("videos/dead_reckoning.mp4"), fps=60)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot([p.x() for p in poses], [p.y() for p in poses], label="Dead reckoning path", color="tab:blue", linewidth=1.8)
    ax.scatter(gps[:, 1], gps[:, 2], marker='.', s=18, c="tab:orange", label="GPS measurements")

    ax.legend(loc="lower right")

    if len(poses) >= 1:
        start = poses[0]
        ax.plot(start.x(), start.y(), marker='o', color='green', markersize=8, label='Start')
        end = poses[-1]
        ax.plot(end.x(), end.y(), marker='o', color='red', markersize=8, label='End')
    
    ax.set_title("Dead Reckoning Path vs GPS Measurements")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__": 
    main()
    
