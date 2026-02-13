# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import load_config
from data_loader import VictoriaParkLoader
from factor_graph_slam import FactorGraphSLAM
from visualization import SLAMHistory, SLAMVisualizer


def main():
    # sensorOffset = np.array([car.a + car.L, car.b])

    data_folder = Path(__file__).parents[1].joinpath('data/victoria_park')
    data_loader = VictoriaParkLoader(data_folder=data_folder)

    config_path = Path(__file__).parents[0].joinpath('conf/victoria_park_config.yaml')
    cfg = load_config(config_path)

    initial_pose = np.array(data_loader.initial_position)

    slam = FactorGraphSLAM(cfg.inference, initial_pose)

    history = SLAMHistory()

    N = 5000 # max number of steps to process

    # %% Run dead reckoning
    poses_dead_reckoning = []
    x_prev = gtsam.Pose2(initial_pose) # IMPORTANT: gtsanm.Pose2(initial_pose) is different from gtsam.Pose2(*initial_pose)
    poses_dead_reckoning.append(x_prev)
    # Deac reckoning
    for data_k in data_loader.iterate_steps(max_steps=N):
        x_pred = x_prev.compose(gtsam.Pose2(*data_k.odometry))
        poses_dead_reckoning.append(x_pred)
        x_prev = x_pred

    fig, ax = plt.subplots(figsize=(13, 8))
    x_coords = [pose.x() for pose in poses_dead_reckoning]
    y_coords = [pose.y() for pose in poses_dead_reckoning]
    ax.plot(x_coords, y_coords, 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$')

    # %% Run SLAM 

    # Accumulated odometry since last measurement, reset after each measurement step
    odometry_integrated = gtsam.Pose2() 

    for data_k in tqdm(data_loader.iterate_steps(max_steps=N), total=N - 1, desc="SLAM"):
        odometry_integrated = odometry_integrated.compose(gtsam.Pose2(*data_k.odometry))

        if data_k.has_laser:
            meas_gtsam = [(r, gtsam.Rot2(b)) for r, b in data_k.measurements]
            step_result = slam.process_step(odometry_integrated, meas_gtsam)
            history.add(step_result)
            
            # Reset accumulated odometry
            odometry_integrated = gtsam.Pose2()  

    # %% Visualize final result
    visualizer = SLAMVisualizer(history, cfg.visualization)

    marginals = slam.get_marginals()
    poses_est = slam.get_estimated_poses()
    landmarks_est = slam.get_estimated_landmarks()

    plt.figure(figsize=(13, 8))
    x_coords = [pose.x() for pose in poses_est]
    y_coords = [pose.y() for pose in poses_est]
    plt.plot(x_coords, y_coords, 'b-', alpha=0.7, label=r'$\hat{x}_{SLAM}$')
    x_coords = [pose.x() for pose in poses_dead_reckoning]
    y_coords = [pose.y() for pose in poses_dead_reckoning]
    plt.plot(x_coords, y_coords, 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$')
    x_coords = [lm[0] for lm in landmarks_est]
    y_coords = [lm[1] for lm in landmarks_est]
    plt.scatter(x_coords, y_coords, c='r', marker='x', label='estimated landmarks')
    plt.legend()
    plt.show()


    # fig, ax = visualizer.plot_final_result(slam, marginals, poses_dead_reckoning=poses_dead_reckoning, ax=ax)
    # fig.savefig("figures/jcbb_vp.pdf", bbox_inches="tight")
    # plt.show()

    # fig, ax = visualizer.plot_NIS()
    # fig.savefig("figures/jcbb_vp_nis.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
