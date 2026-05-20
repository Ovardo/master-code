# Adapted from Odin Aleksander Severinsen Graded Assignment 2 code in TTK4250.
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import SlamConfig
from data_loader import VictoriaParkLoader
from logger import SlamLogger
from plotting import save_run_figures
from slam import FactorGraphSLAM


def main() -> None:
    
    output_dir = Path("runs") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logger = SlamLogger(output_dir)

    config_name = "vp1.yaml"
    config = SlamConfig.load(config_name)
    config.save(output_dir / config_name)

    # Data-loader
    loader = VictoriaParkLoader()

    # SLAM system
    slam = FactorGraphSLAM(
        config=config,
        logger=logger,
        initial_pose=loader.initial_pose,
    )

    # Num lidar scan steps
    K = 7300 # max is 7248

    records = []

    # Main loop
    for k, meas in tqdm(enumerate(loader.iterate(K)), total=min(K, 7248), desc="SLAM"):
        
        record = slam.update(meas)
 
        if k % 200 == 0 and k > 0:
            snapshot = slam.get_snapshot()
            logger.save_snapshot(k, snapshot)
            
            record['fg_error'] = slam.get_error()
            record['n_factors'] = slam.get_num_factors()

        records.append(record)


    snapshot = slam.get_snapshot()
    logger.save_snapshot(k, snapshot, final=True)
    records[-1]['fg_error'] = slam.get_error()
    records[-1]['n_factors'] = slam.get_num_factors()
    
    steps = logger.convert_records_to_steps(records)
    logger.save_steps(steps)
    logger.save_metadata(steps, snapshot)

    # Produce + save figures.
    save_run_figures(
        run_dir=output_dir,
        steps=steps,
        snapshot=snapshot,
        gnss=loader.gnss,
        fmt="pdf",
        show=True,
    )


if __name__ == "__main__":
    main()
