from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from master_code.config import SlamConfig
from master_code.loaders.simulated import SimulatedDataLoader
from master_code.slam import run_slam


def run_sim(
    config: SlamConfig,
    output_dir: Path,
    num_steps: int | None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    dataset = SimulatedDataLoader()
    return run_slam(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        num_steps=num_steps,
        show_plots=show_plots,
        save_plots=save_plots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLAM on the simulated dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_sim.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Do not display plots after the run.",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Do not save plots to the run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("runs/sim") / datetime.now().strftime("%Y%m%d_%H%M%S")

    config = SlamConfig.load(args.config)

    run_sim(
        config=config,
        output_dir=output_dir,
        num_steps=args.steps,
        show_plots=not args.no_show_plots,
        save_plots=not args.no_save_plots,
    )


if __name__ == "__main__":
    main()
