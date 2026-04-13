# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import load_config
from data_loader import VictoriaParkLoader
from factor_graph_slam import FactorGraphSLAM


def main():

    config = load_config("vp1.yaml")
    
    data_loader = VictoriaParkLoader()
    
    slam = FactorGraphSLAM(
        cfg = config, 
        pose0 = data_loader.initial_pose,
    )


    K = 7300 # max number of scan steps to process
    # %% Run SLAM 
    for step in tqdm(data_loader.iter_lidar_steps(max_steps=K), total=K-1, desc="SLAM"):
        
        slam.update(step)
        

    # Plotting
    poses = slam.get_estimated_poses()
    landmarks = slam.get_estimated_landmarks()
    poses_dr = slam.get_dead_reckoning_poses()


    plt.figure(figsize=(10, 10))
    plt.plot(poses[:, 0], poses[:, 1], label="Estimated trajectory")
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='x', label="Estimated landmarks")
    plt.title("SLAM Result (#landmarks: {})".format(len(landmarks)))
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    plt.figure(figsize=(10, 10))
    plt.plot(poses_dr[:, 0], poses_dr[:, 1], label="Odometry trajectory")
    plt.title("Odometry Result")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    # Timing
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 1, 1)
    plt.plot(slam._counts_local_landmark)
    plt.title("Number of Predicted Landmarks Over Time")
    plt.xlabel("Step")
    plt.ylabel("Number of Predicted Landmarks")
    plt.grid() 

    plt.subplot(2, 1, 2)
    plt.plot(slam._times_update)
    plt.title("SLAM Step Processing Time")
    plt.xlabel("Step")
    plt.ylabel("Time (s)")
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()

    plt.figure(figsize=(7, 7))
    plt.scatter(slam._counts_local_landmark, slam._times_update)
    plt.title("Processing Time vs Number of Predicted Landmarks")
    plt.xlabel("Number of Predicted Landmarks")
    plt.ylabel("Time (s)")
    plt.grid()
    plt.tight_layout()

    # Stacked time plot
    steps = np.arange(len(slam._times_update))
    cov_times = np.asarray(slam._times_covariance_extraction)
    ass_times = np.asarray(slam._times_association)
    opt_times = np.asarray(slam._times_optimization)
    opt2_times = np.asarray(slam._times_optimization2)
    other_times = np.asarray(slam._times_update) - (cov_times + ass_times + opt_times)

    plt.figure(figsize=(10, 5))
    plt.stackplot(
        steps,
        cov_times,
        ass_times,
        opt_times,
        opt2_times,
        other_times,
        labels=["Covariance extraction", "Association", "Optimization", "Opt 2",  "Other"],
    )
    plt.title("SLAM Processing Time Breakdown")
    plt.xlabel("Step")
    plt.ylabel("Time (s)")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()


    cov_cum_times = np.cumsum(cov_times)
    ass_cum_times = np.cumsum(ass_times)
    opt_cum_times = np.cumsum(opt_times)
    opt2_cum_times = np.cumsum(opt2_times)
    other_cum_times = np.cumsum(other_times)

    plt.figure(figsize=(10, 5))
    plt.stackplot(
        steps,
        cov_cum_times,
        ass_cum_times,
        opt_cum_times,
        opt2_cum_times, 
        other_cum_times,
        labels=["Covariance extraction", "Association", "Optimization", "Opt 2", "Other"],
    )
    plt.title("SLAM Cumulative Processing Time Breakdown")
    plt.xlabel("Step")
    plt.ylabel("Time (s)")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()
