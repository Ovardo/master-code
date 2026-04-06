# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
from pathlib import Path

from tqdm import tqdm

from config import load_config
from data_loader import VictoriaParkLoader
from factor_graph_slam import FactorGraphSLAM

from matplotlib import pyplot as plt


def main():

    config = load_config("vp1.yaml")
    
    data_loader = VictoriaParkLoader(None)

    slam = FactorGraphSLAM(
        cfg = config, 
        pose0 = data_loader.initial_position,
    )

    K = 5000 # max number of steps to process

    # %% Run SLAM 
    for input in tqdm(data_loader.iterate_steps(max_steps=K), total=K-1, desc="SLAM"):
        z_lsr = input.z_lsr
        z_odo = (input.ve_dr, input.alpha_dr, input.dt_dr)
        
        slam.register_odometry(z_odo)
        
        if z_lsr is None:
            continue

        slam.register_scan(z_lsr) 


    poses = slam.get_estimated_poses()
    landmarks = slam.get_estimated_landmarks()

    plt.figure(figsize=(10, 10))
    plt.plot(poses[:, 0], poses[:, 1], label="Estimated trajectory")
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='x', label="Estimated landmarks")
    plt.title("SLAM Result")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
