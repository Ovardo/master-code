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
    logger   = SlamLogger(run_path, snapshot_every=1000)   
    shutil.copy2(config_path, run_path / config_path.name) # copy config to results for record-keeping
    
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
        fmt="pdf", # vector PDF — ideal for thesis inclusion
        show=True,
        show_covariances=True,
    )

# %%
if __name__ == "__main__":
    main()

