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

    K = 73000 # max number of steps to process

    lsr_index = 0 
    # %% Run SLAM 
    for input in tqdm(data_loader.iterate_steps(max_steps=K), total=K-1, desc="SLAM"):
        z_lsr = input.z_lsr
        z_odo = (input.ve_dr, input.alpha_dr, input.dt_dr)
        
        slam.register_odometry(z_odo)
        # print(f"{lsr_index }. ODOM: {input.timestamp:.3f}, {input.ve_dr:.3f}, {input.alpha_dr:.3f}, {input.dt_dr:.3f}")
        
        if z_lsr is None:
            continue

        slam.register_scan(z_lsr) 
        # print(f"{lsr_index}. LSR: {input.timestamp}")
        lsr_index += 1


    poses = slam.get_estimated_poses()
    landmarks = slam.get_estimated_landmarks()
    poses_dr = slam.get_poses_dr()

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

    plt.figure(figsize=(10, 10))
    plt.plot(poses_dr[:, 0], poses_dr[:, 1], label="Odometry trajectory")
    plt.title("Odometry Result")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
