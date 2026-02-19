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
from utils.utils_gtsam import pose2_to_array


def main():
    # sensorOffset = np.array([car.a + car.L, car.b])

    data_folder = Path(__file__).parents[1].joinpath('data/victoria_park')
    data_loader = VictoriaParkLoader(data_folder=data_folder)

    config_path = Path(__file__).parents[0].joinpath('conf/victoria_park_config.yaml')
    cfg = load_config(config_path)

    initial_pose = np.array(data_loader.initial_position)

    slam = FactorGraphSLAM(cfg.inference, initial_pose)

    history = SLAMHistory()

    K = 5000 # max number of steps to process

    # %% Run dead reckoning
    x_prev = gtsam.Pose2(*initial_pose) # IMPORTANT: gtsanm.Pose2(initial_pose) is different from gtsam.Pose2(*initial_pose)
    poses_dead_reckoning = [pose2_to_array(x_prev)]

    for data_k in data_loader.iterate_steps(max_steps=K):
        x_pred = x_prev.compose(gtsam.Pose2(*data_k.odometry))
        poses_dead_reckoning.append(pose2_to_array(x_pred))
        x_prev = x_pred

    poses_dead_reckoning = np.array(poses_dead_reckoning)

    # %% Run SLAM 

    # Accumulated odometry since last measurement, reset after each measurement step
    odometry_integrated = gtsam.Pose2() 

    for data_k in tqdm(data_loader.iterate_steps(max_steps=K), total=K-1, desc="SLAM"):
        odometry = data_k.odometry
        measurements = data_k.measurements
        
        odometry_integrated = odometry_integrated.compose(gtsam.Pose2(*odometry))

        if data_k.has_laser:
            step_result = slam.process_step(odometry_integrated, measurements)
            history.add(step_result)
            
            # Reset accumulated odometry
            odometry_integrated = gtsam.Pose2()  

    # %% Visualize final result
    marginals = slam.get_marginals()
    
    visualizer = SLAMVisualizer(cfg.visualization, history)
    visualizer.plot_estimates_np(step=-1, dead_reckoning_poses=poses_dead_reckoning)
    visualizer.plot_measurements_polar_np(step=-1)
    visualizer.plot_measurements_cartesian_np(step=-1)
    visualizer.create_measurement_video_cartesian('videos/measurements_cartesian.mp4', fps=5)
    visualizer.create_estimates_video('videos/estimates.mp4', fps=5, dead_reckoning_poses=poses_dead_reckoning) 
    # visualizer.create_measurement_video_polar('measurements_polar.mp4', fps=5)
    # visualizer.create_dashboard_video('dashboard.mp4', fps=5, dead_reckoning_poses=poses_dead_reckoning)


    # fig, ax = visualizer.plot_final_result(slam, marginals, poses_dead_reckoning=poses_dead_reckoning, ax=ax)
    # fig.savefig("figures/jcbb_vp.pdf", bbox_inches="tight")
    # plt.show()

    # fig, ax = visualizer.plot_NIS()
    # fig.savefig("figures/jcbb_vp_nis.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
