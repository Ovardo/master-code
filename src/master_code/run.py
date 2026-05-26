# Adapted from Odin Aleksander Severinsen Graded Assignment 2 code in TTK4250.
from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from master_code.config import SlamConfig
from master_code.data_loader import SimulatedDataLoader, VictoriaParkLoader
from master_code.logger import SlamLogger
from master_code.measurements import SlamStepInput
from master_code.plotter import SlamRunPlotter
from master_code.preprocessing import preprocess_victoria_park_step
from master_code.slam import FactorGraphSLAM
from master_code.paths import RUNS_ROOT


DEFAULT_CONFIGS = {
    "victoria_park": "vp1.yaml",
    "simulated": "simulated.yaml",
}


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    path = Path(output_dir)
    if path.is_absolute():
        return path
    return RUNS_ROOT / path


def count_victoria_park_steps(loader: VictoriaParkLoader, max_steps: int | None) -> int:
    available_steps = max(0, loader.lsr_timestamps.size - 2)
    if max_steps is None:
        return available_steps
    return min(max_steps, available_steps)


def count_simulated_steps(loader: SimulatedDataLoader, max_steps: int | None) -> int:
    available_steps = len(loader.odometry)
    if max_steps is None:
        return available_steps
    return min(max_steps, available_steps)


def load_dataset(
    dataset: str,
    config: SlamConfig,
    max_steps: int | None,
) -> tuple[Iterator[SlamStepInput], int, object, dict]:
    if dataset == "victoria_park":
        loader = VictoriaParkLoader()
        raw_steps = loader.iterate(max_steps)
        steps = (
            preprocess_victoria_park_step(
                raw_step,
                odometry_covariance=config.noise.odom_cov_matrix,
                max_range=config.sensor.max_range,
            )
            for raw_step in raw_steps
        )
        return (
            steps,
            count_victoria_park_steps(loader, max_steps),
            loader.initial_pose,
            {"gnss": loader.gnss_filtered},
        )

    if dataset == "simulated":
        loader = SimulatedDataLoader()
        steps = loader.iterate(
            max_steps,
            odometry_covariance=config.noise.odom_cov_matrix,
            max_range=config.sensor.max_range,
        )
        return (
            steps,
            count_simulated_steps(loader, max_steps),
            loader.initial_pose,
            loader.reference,
        )

    raise ValueError(f"Unknown dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the factor-graph SLAM pipeline.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_CONFIGS),
        default="victoria_park",
        help="Dataset to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file name or path. Defaults depend on --dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=7250,
        help="Maximum number of SLAM steps. Use -1 for all available steps.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory. Relative paths are placed under runs/.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only save logs and metadata, skipping figures.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show generated figures while saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be positive, or -1 for all available steps.")

    max_steps = None if args.max_steps < 0 else args.max_steps
    output_dir = resolve_output_dir(args.output_dir)

    config_name = args.config or DEFAULT_CONFIGS[args.dataset]
    config = SlamConfig.load(config_name)
    logger = SlamLogger(output_dir, config.logging)
    config.save(output_dir / "config.yaml")

    steps, total_steps, initial_pose, reference = load_dataset(args.dataset, config, max_steps)
    logger.save_reference(**reference)

    # SLAM system
    slam = FactorGraphSLAM(
        config=config,
        logger=logger,
        initial_pose=initial_pose,
    )

    diagnostics_steps = []
    last_step = None

    # Main loop
    t0 = time.perf_counter()
    for meas in tqdm(steps, total=total_steps, desc="SLAM"):
        diagnostics = slam.update(meas)
        diagnostics_steps.append(diagnostics)
        last_step = meas.scan_step

    total_time = time.perf_counter() - t0

    if last_step is None or not diagnostics_steps:
        raise RuntimeError("No SLAM steps were processed.")

    logger.save_snapshot(last_step, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(diagnostics_steps)
    logger.save_metadata(diagnostics_steps[-1], total_time)

    # Produce + save figures.
    if not args.no_plots:
        plotter = SlamRunPlotter.from_run(output_dir)
        plotter.plot_all(save=True, show=args.show_plots)


if __name__ == "__main__":
    main()
