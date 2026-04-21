# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import SlamConfig

# from plot import SlamPlotter
from data_loader import VictoriaParkLoader
from factor_graph_slam import FactorGraphSLAM
from logger import SlamLogger


def main():
    config_filename = "vp1.yaml"
    config_path = SlamConfig.resolve_path(config_filename)
    config = SlamConfig.load(config_path)

    # ── output paths ─────────────────────────────────────────────────────────
    run_filename = datetime.now().strftime("vp1_%Y%m%d_%H%M%S")
    run_path  = Path("results") / run_filename
    logger   = SlamLogger(run_path, snapshot_every=500)   
    shutil.copy2(config_path, run_path / config_path.name)
    
    # ── data & SLAM ───────────────────────────────────────────────────────────
    data_loader  = VictoriaParkLoader()
    initial_pose = data_loader.initial_pose
    
    slam = FactorGraphSLAM(cfg=config, logger=logger)
    slam.add_prior_factor(initial_pose)

    K = 7300 # max number of scan steps to process
    for step in tqdm(data_loader.iter_lidar_steps(max_steps=K), total=K-1, desc="SLAM"):
        slam.update(step)


    # ── save all logged data + final snapshot ─────────────────────────────────
    logger.save(slam.get_snapshot())
 
    # ── produce + save figures (set show=False for headless/batch runs) ───────
    from plot import load_and_plot_all
    load_and_plot_all(
        run_path,
        fmt="pdf",          # vector PDF — ideal for thesis inclusion
        show=True,
        show_covariances=True,
    )



    # # Plotting
    # poses = slam.get_poses()
    # landmarks = slam.get_landmarks()
    # slam._counts_local_landmark
    # slam._times_update
    # slam._times_covariance_extraction
    # slam._times_association
    # slam._times_optimization
    # slam._times_optimization2


    # plt.figure(figsize=(10, 10))
    # plt.plot(poses[:, 0], poses[:, 1], label="Estimated trajectory")
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='x', label="Estimated landmarks")
    # plt.title("SLAM Result (#landmarks: {})".format(len(landmarks)))
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.legend()
    # plt.grid()
    # plt.axis('equal')

    # # Timing
    # plt.figure(figsize=(10, 5))
    
    # plt.subplot(2, 1, 1)
    # plt.plot(slam._counts_local_landmark)
    # plt.title("Number of Predicted Landmarks Over Time")
    # plt.xlabel("Step")
    # plt.ylabel("Number of Predicted Landmarks")
    # plt.grid() 

    # plt.subplot(2, 1, 2)
    # plt.plot(slam._times_update)
    # plt.title("SLAM Step Processing Time")
    # plt.xlabel("Step")
    # plt.ylabel("Time (s)")
    # plt.yscale('log')
    # plt.grid()
    # plt.tight_layout()

    # plt.figure(figsize=(7, 7))
    # plt.scatter(slam._counts_local_landmark, slam._times_update)
    # plt.title("Processing Time vs Number of Predicted Landmarks")
    # plt.xlabel("Number of Predicted Landmarks")
    # plt.ylabel("Time (s)")
    # plt.grid()
    # plt.tight_layout()

    # # Stacked time plot
    # steps = np.arange(len(slam._times_update))
    # cov_times = np.asarray(slam._times_covariance_extraction)
    # ass_times = np.asarray(slam._times_association)
    # opt_times = np.asarray(slam._times_optimization)
    # opt2_times = np.asarray(slam._times_optimization2)
    # other_times = np.asarray(slam._times_update) - (cov_times + ass_times + opt_times)

    # plt.figure(figsize=(10, 5))
    # plt.stackplot(
    #     steps,
    #     cov_times,
    #     ass_times,
    #     opt_times,
    #     opt2_times,
    #     other_times,
    #     labels=["Covariance extraction", "Association", "Optimization", "Opt 2",  "Other"],
    # )
    # plt.title("SLAM Processing Time Breakdown")
    # plt.xlabel("Step")
    # plt.ylabel("Time (s)")
    # plt.legend(loc="upper left")
    # plt.grid()
    # plt.tight_layout()


    # cov_cum_times = np.cumsum(cov_times)
    # ass_cum_times = np.cumsum(ass_times)
    # opt_cum_times = np.cumsum(opt_times)
    # opt2_cum_times = np.cumsum(opt2_times)
    # other_cum_times = np.cumsum(other_times)

    # plt.figure(figsize=(10, 5))
    # plt.stackplot(
    #     steps,
    #     cov_cum_times,
    #     ass_cum_times,
    #     opt_cum_times,
    #     opt2_cum_times, 
    #     other_cum_times,
    #     labels=["Covariance extraction", "Association", "Optimization", "Opt 2", "Other"],
    # )
    # plt.title("SLAM Cumulative Processing Time Breakdown")
    # plt.xlabel("Step")
    # plt.ylabel("Time (s)")
    # plt.legend(loc="upper left")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()




# %%
if __name__ == "__main__":
    main()

# %%
