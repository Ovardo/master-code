# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
import numpy as np
import matplotlib.pyplot as plt
import gtsam 

from pathlib import Path

from tqdm import tqdm

from data_loader import VictoriaParkLoader
from utils.utils_victoria_park import Car
from factor_graph_slam import FactorGraphSLAM, SLAMVisualizer
from config import load_config

def main():

    # sensorOffset = np.array([car.a + car.L, car.b])

    data_folder = Path(__file__).parents[1].joinpath("data/victoria_park")
    data_loader = VictoriaParkLoader(data_folder=data_folder)

    x0 = np.array(data_loader.initial_position)


    config_path = Path(__file__).parents[0].joinpath("conf/victoria_park_config.yaml")
    cfg = load_config(config_path)

    slam = FactorGraphSLAM(cfg.inference, gtsam.Pose2(*x0))
    slam.current_step = 1 # quick fix as we do nt assume measurement at step 0


    # %% Run SLAM (dead reckoning for odometry only)
    N = 1500

    poses_dead_reckoning = []
    x_prev = gtsam.Pose2(*x0)
    poses_dead_reckoning.append(x_prev)
    # Deac reckoning
    for data_k in data_loader.iterate_steps(max_steps=N): 
        x_pred = x_prev.compose(gtsam.Pose2(*data_k.odometry))
        poses_dead_reckoning.append(x_pred)
        x_prev = x_pred
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    x_coords = [pose.x() for pose in poses_dead_reckoning]
    y_coords = [pose.y() for pose in poses_dead_reckoning]
    ax.plot(x_coords, y_coords, 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$'),
    
        
    # %% Run SLAM
    odometry_integrated = gtsam.Pose2()  # integrated odometry since last SLAM update, reset after each SLAM update

    for data_k in tqdm(data_loader.iterate_steps(max_steps=N), total=N-1, desc="SLAM"):

        odometry_integrated = odometry_integrated.compose(gtsam.Pose2(*data_k.odometry)) 
        
        if data_k.has_laser:
            meas_gtsam = [(r, gtsam.Rot2(b)) for r, b in data_k.measurements]
            slam.process_step(odometry_integrated, meas_gtsam)
            odometry_integrated = gtsam.Pose2() # reset accumulated odometry
    
    
    # %% Visualize final result
    marginals = slam.get_marginals()
    fig, ax = SLAMVisualizer.plot_final_result(slam, marginals, poses_dead_reckoning=poses_dead_reckoning, ax=ax)
    fig.savefig("figures/jcbb_vp.pdf", bbox_inches="tight")
    plt.show()
    

    # %%
    fig, ax = SLAMVisualizer.plot_NIS(slam)
    fig.savefig("figures/jcbb_vp_nis.pdf", bbox_inches="tight")
    plt.show()


 

if __name__ == "__main__":
    main()



