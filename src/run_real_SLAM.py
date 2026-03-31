# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
from pathlib import Path

import gtsam
import numpy as np
from tqdm import tqdm

from config import load_config
from data_loader import VictoriaParkLoader
from experiment_results import ExperimentReferenceData, save_result
from factor_graph_slam import FactorGraphSLAM
from history import SLAMHistory
from slam_types import SLAMHistoryEntry
from timing_profiler import TimingProfiler
from utils.utils_victoria_park import odometry_func
from visualization import SLAMVisualizer
from landmark_manager import TentativeLandmarkManager
from association import Associator



def main():
    # sensorOffset = np.array([car.a + car.L, car.b])

    # config_path = Path(__file__).parents[0].joinpath('conf/victoria_park_config.yaml')
    # config_path = Path(__file__).parents[0].joinpath('conf/default_config.yaml')
    config_path = Path(__file__).parents[0].joinpath('conf/victoria_park.yaml')
    cfg = load_config(config_path)

    data_folder = Path(__file__).parents[1].joinpath('data/victoria_park')
    data_loader = VictoriaParkLoader(data_folder=data_folder)

    profiler = TimingProfiler(enabled=cfg.profilinfg_enabled)

    slam = FactorGraphSLAM(
        cfg = cfg.inference, 
        initial_pose = np.array(data_loader.initial_position),
        profiler = profiler,
        tentative_manager = TentativeLandmarkManager(cfg.inference.landmark_manager),
        associator = Associator(cfg.inference.association),
    )

    history = SLAMHistory()

    K = 30000 # max number of steps to process

    # %% Run SLAM 
    for step_input in tqdm(data_loader.iterate_steps(max_steps=K), total=K-1, desc="SLAM"):

        step_output = slam.process_step(step_input)
        
        if step_output is not None:
            history_entry = SLAMHistoryEntry(
                step_index=step_input.step_index,
                step_input=step_input,
                step_output=step_output,
                reference=None,
                dead_reckoning_poses=slam.get_poses_dr(),
            )
            history.add(history_entry)
        

    # %% Visualize final result
    save_result(
        config=cfg,
        history=history,
        profiler=profiler,
        reference_data=ExperimentReferenceData(
            gps_track=data_loader.gps,
        ),
    )
    
    
    visualizer = SLAMVisualizer(cfg.visualization, history)
    visualizer.plot_estimates(step=-1, plot_dead_reckoning=True, plot_covariances=False, plot_predicted_measurements=True)
    # visualizer.plot_measurements_polar(step=-1)
    # visualizer.plot_measurements_cartesian(step=-1)
    # visualizer.create_measurement_video_cartesian('videos/measurements_cartesian.mp4', fps=5)
    # visualizer.create_estimates_video('videos/estimates.mp4', fps=5, dead_reckoning_poses=poses_dead_reckoning) 

    # visualizer.create_measurement_video_polar('measurements_polar.mp4', fps=5)
    # visualizer.create_dashboard_video('videos/dashboard.mp4', fps=10, plot_covariance=False)

    # fig, ax = visualizer.plot_final_result(slam, marginals, poses_dead_reckoning=poses_dead_reckoning, ax=ax)
    # fig.savefig("figures/jcbb_vp.pdf", bbox_inches="tight")
    # plt.show()

    # fig, ax = visualizer.plot_NIS()
    # fig.savefig("figures/jcbb_vp_nis.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
