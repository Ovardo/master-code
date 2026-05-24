# Adapted from Odin Aleksander Severinsen Graded Assignment 2 code in TTK4250.
import time
from datetime import datetime
from tqdm import tqdm

from master_code.config import SlamConfig
from master_code.data_loader import VictoriaParkLoader
from master_code.logger import SlamLogger
from master_code.plotter import SlamRunPlotter
from master_code.slam import FactorGraphSLAM
from master_code.paths import RUNS_ROOT


def main() -> None:
    
    output_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_4_6")

    config_name = "vp1.yaml"
    config = SlamConfig.load(config_name)
    logger = SlamLogger(output_dir, config.logging)
    config.save(output_dir / "config.yaml")

    # Data-loader
    loader = VictoriaParkLoader()

    # SLAM system
    slam = FactorGraphSLAM(
        config=config,
        logger=logger,
        initial_pose=loader.initial_pose,
    )

    # Num lidar scan steps
    K = 7250 # max is 7248

    steps_diagnostics_list = []

    # Main loop
    t0 = time.perf_counter()
    for k, meas in tqdm(enumerate(loader.iterate(K)), total=min(K, 7248), desc="SLAM"):

        diagnostics = slam.update(meas)
        steps_diagnostics_list.append(diagnostics)

    total_time = time.perf_counter() - t0

    logger.save_snapshot(k, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(steps_diagnostics_list)
    logger.save_metadata(steps_diagnostics_list[-1], total_time)

    # Produce + save figures.
    plotter = SlamRunPlotter.from_run(output_dir)
    plotter.plot_all(save=True, show=True)


if __name__ == "__main__":
    main()
