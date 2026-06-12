from __future__ import annotations

import gtsam
import matplotlib.pyplot as plt

from master_code.loaders.victoria_park import VictoriaParkLoader
from master_code.plotting.thesis_style import apply_thesis_style, save_figure, thesis_figsize
from master_code.preprocessing import relative_pose
from master_code.plotter import plot_estimate

import numpy as np

# plt.style.use(['science'])


# matplotlib.use('qtagg')


def dead_reckoned_poses(frame_stride: int = 1) -> list[gtsam.Pose2]:
    loader = VictoriaParkLoader()
    odometry = loader.odometry
    
    dt = 0.025
    
    pose = gtsam.Pose2(*loader.initial_pose)
    poses = [pose]

    for i, odom in enumerate(odometry, start=1):
        velocity = odom[1]
        steering = odom[2]
        
        odo, _ = relative_pose(velocity, steering, dt)
        pose = pose.compose(odo)

        if i % frame_stride == 0:
            poses.append(pose)
    
    return poses



def main():
    loader = VictoriaParkLoader()
    gnss = loader.gnss_filtered
    poses = dead_reckoned_poses(frame_stride=10)
    poses = np.array([[p.x(), p.y()] for p in poses])

    fig, ax = plt.subplots(figsize=(5.5,4))
    plot_estimate(
        ax=ax,
        poses=poses,
        gnss=gnss,
        traj_label="Trajectory",
        equal_aspect=False,
        show_grid=True,
        title=None,
        show_legend=False,
    )
    fig.subplots_adjust(right=0.72)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    save_figure(fig, "dead_reckoning")
    # plt.show()


if __name__ == "__main__": 
    main()
    
