# Adapted from Odin Aleksander Severinsen Graded Assignment 2 code in TTK4250.
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import SlamConfig
from logger import SlamLogger
from plotting import SlamPlotter
from data_loader import VictoriaParkLoader
from slam import FactorGraphSLAM


def main() -> None:
    out_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SlamLogger(out_dir, snapshot_every=None)

    config_name = "vp1.yaml"
    config = SlamConfig.load(config_name)
    config.save(out_dir / config_name)

    # Data-loader
    loader = VictoriaParkLoader()

    # SLAM system
    slam = FactorGraphSLAM(
        config=config,
        logger=logger,
        initial_pose=loader.initial_pose,
    )

    # Num lidar scan steps
    K = 1000 # Max is 7250

    # Main loop
    for step in tqdm(loader.iterate(max_steps=K), total=K-1, desc="SLAM"):
        slam.update(step)

    # Save all logged data and the final snapshot and error.
    logger.save(
        snapshot=slam.get_snapshot(),
        error=slam.get_error(),
    )

    # Produce + save figures.
    plotter = SlamPlotter.from_run(out_dir, load_gps=True)
    plotter.save_all(fmt="pdf", covariances=True, gnss=True)
    plotter.show_all(covariances=True, gnss=True)


if __name__ == "__main__":
    main()
