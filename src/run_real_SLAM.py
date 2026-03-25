# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import load_config
from data_loader import VictoriaParkLoader
from factor_graph_slam import FactorGraphSLAM
from utils.utils_gtsam import pose2_to_array
from utils.utils_victoria_park import odometry_func
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

    K = 10000 # max number of steps to process

    # %% Run dead reckoning
    x_prev = gtsam.Pose2(*initial_pose) # IMPORTANT: gtsanm.Pose2(initial_pose) is different from gtsam.Pose2(*initial_pose)
    poses_dead_reckoning = [pose2_to_array(x_prev)]

    for data_k in data_loader.iterate_steps(max_steps=K):
        odo = data_k.odometry
        
        # x_pred = x_prev.compose(gtsam.Pose2(*data_k.odometry))
        x_pred = x_prev.compose(odo)
        poses_dead_reckoning.append(pose2_to_array(x_pred))
        x_prev = x_pred

    poses_dead_reckoning = np.array(poses_dead_reckoning)

    # from scipy.io import loadmat
    # data_folder = Path(__file__).parents[1]
    # mat_path = data_folder.joinpath('data/victoriaParkDataset.mat')
    # data = loadmat(mat_path)
    # U = data['controllerInput']
    # X_dr = data['deadReckoning']
    # X_gps = data['gpsLatLong']
    # Z = data['measurements'][:,0]

    # %% Run SLAM 
    k = 0
    for data_k in tqdm(data_loader.iterate_steps(max_steps=K), total=K-1, desc="SLAM"):
        
        
        # u = U[k]
        # dt = 0.025
        # odometery = gtsam.Pose2(odometry_func(u[0], u[1], dt))
        # measurements = Z[k]

        # odometery = data_k.odometry
        # measurements = data_k.measurements  
        
        # result = slam.process_step(odometery, measurements)

        result = slam.process_step(data_k)
        
        if result is not None:
            history.add(result)
        
        k += 1


    # %% Visualize final result


    # from utils.utils_plot import plot_result, MultivariateNormalParameters
    # poses = slam.get_estimated_poses()
    # poses_cov = slam.get_estimated_pose_covariances()
    # landmark = slam.get_estimated_landmarks()
    # landmark_cov = slam.get_estimated_landmark_covariances()

    # poses_dist = [MultivariateNormalParameters(pose, cov) for pose, cov in zip(poses, poses_cov)]
    # landmarks_dist = [MultivariateNormalParameters(lm, cov) for lm, cov in zip(landmark, landmark_cov)]
        
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plot_result(ax, poses_dist, landmarks_dist, exact_map=False)

    
    visualizer = SLAMVisualizer(cfg.visualization, history)
    visualizer.plot_estimates(step=-1, plot_dead_reckoning=True, plot_covariances=False, plot_predicted_measurements=True)
    # visualizer.plot_measurements_polar(step=-1)
    # visualizer.plot_measurements_cartesian(step=-1)
    # visualizer.create_measurement_video_cartesian('videos/measurements_cartesian.mp4', fps=5)
    # visualizer.create_estimates_video('videos/estimates.mp4', fps=5, dead_reckoning_poses=poses_dead_reckoning) 

    # visualizer.create_measurement_video_polar('measurements_polar.mp4', fps=5)
    visualizer.create_dashboard_video('videos/dashboard.mp4', fps=10, plot_covariance=False)


    # fig, ax = visualizer.plot_final_result(slam, marginals, poses_dead_reckoning=poses_dead_reckoning, ax=ax)
    # fig.savefig("figures/jcbb_vp.pdf", bbox_inches="tight")
    # plt.show()

    # fig, ax = visualizer.plot_NIS()
    # fig.savefig("figures/jcbb_vp_nis.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
