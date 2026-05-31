from __future__ import annotations

from datetime import datetime
from pathlib import Path

from master_code.config import SlamConfig
from master_code.loaders.victoria_park import VictoriaParkLoader
from master_code.paths import RUNS_ROOT
from master_code.slam import run_slam


def run_real(
    config: SlamConfig,
    output_dir: Path,
    num_steps: int | None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    dataset = VictoriaParkLoader()
    return run_slam(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        num_steps=num_steps,
        show_plots=show_plots,
        save_plots=save_plots,
    )


def main() -> None:
    NUM_STEPS = 500
    OUTPUT_DIR = RUNS_ROOT / "real" / datetime.now().strftime("%Y%m%d_%H%M%S")
    CONFIG_NAME = "real.yaml"

    config = SlamConfig.load(CONFIG_NAME)

    run_real(
        config=config,
        output_dir=OUTPUT_DIR,
        num_steps=NUM_STEPS,
        show_plots=True,
        save_plots=True,
    )


if __name__ == "__main__":
    main()
