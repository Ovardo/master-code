"""Inspect simulated landmark initialization using ground-truth poses.

This script ignores data association and optimization. It simply back-projects
each simulated range-bearing measurement through the corresponding ground-truth
vehicle pose and plots the resulting initialized landmark cloud against the
ground-truth landmarks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from master_code.data_loader import SimulatedDataLoader
from master_code.plotting.plotting_funcs import plot_estimate


MAX_STEPS: int | None = None
MEASUREMENT_STRIDE = 1
POSE_OFFSET = 0
SAVE_FIGURE = False
FIGURE_PATH = Path("simulated_gt_landmark_init.pdf")


def initialize_landmarks_from_gt(
    loader: SimulatedDataLoader,
    max_steps: int | None = MAX_STEPS,
    measurement_stride: int = MEASUREMENT_STRIDE,
    pose_offset: int = POSE_OFFSET,
) -> np.ndarray:
    stop = len(loader.measurements)
    if max_steps is not None:
        stop = min(stop, max_steps)

    initialized = []

    for k, measurements in enumerate(loader.measurements[:stop]):
        measurements = measurements[::measurement_stride]
        if len(measurements) == 0:
            continue

        x, y, theta = loader.poses_gt[k + pose_offset]
        ranges = measurements[:, 0]
        bearings = measurements[:, 1]
        angles = theta + bearings

        landmarks = np.column_stack(
            [
                x + ranges * np.cos(angles),
                y + ranges * np.sin(angles),
            ]
        )
        initialized.append(landmarks)

    if not initialized:
        return np.zeros((0, 2))

    return np.vstack(initialized)


def main() -> None:
    loader = SimulatedDataLoader()
    initialized_landmarks = initialize_landmarks_from_gt(loader)

    print(f"Initialized {len(initialized_landmarks)} landmark observations")
    print(f"Ground-truth landmarks: {len(loader.landmarks_gt)}")

    stop_pose = len(loader.poses_gt) if MAX_STEPS is None else min(len(loader.poses_gt), MAX_STEPS + 1)
    fig, ax = plot_estimate(
        poses_gt=loader.poses_gt[:stop_pose],
        landmarks=initialized_landmarks,
        landmarks_gt=loader.landmarks_gt,
        title="Simulated landmark initialization from ground-truth poses",
        traj_label="Ground-truth trajectory",
    )
    ax.set_title("Simulated landmark initialization from ground-truth poses")

    if SAVE_FIGURE:
        fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {FIGURE_PATH.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
