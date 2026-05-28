from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from master_code.config import SlamConfig
from master_code.loaders.victoria_park import VictoriaParkLoader
from master_code.logger import SlamLogger, StepDiagnostics
from master_code.paths import RUNS_ROOT
from master_code.preprocessing import preprocess_victoria_park_step
from master_code.slam import GraphSLAM


@dataclass(frozen=True)
class RunSpec:
    """One experiment to run.

    ``run_dir`` is interpreted relative to ``runs/`` unless it is absolute.
    ``config_file`` can be a file in ``src/master_code/config/files`` or an
    explicit path, matching ``SlamConfig.load``.
    """

    run_dir: str | Path
    config_file: str | Path


# Edit this list when you want to rerun a set of thesis experiments.
RUNS: list[RunSpec] = [
    RunSpec(run_dir="M1_N4", config_file="M1_N4.yaml"),
    RunSpec(run_dir="M2_N4", config_file="M2_N4.yaml"),
    RunSpec(run_dir="M3_N4", config_file="M3_N4.yaml"),
    RunSpec(run_dir="M4_N4", config_file="M4_N4.yaml"),
]


MAX_STEPS = 7250
SAVE_PLOTS = True
SHOW_PLOTS = False


def resolve_run_dir(run_dir: str | Path) -> Path:
    """Return the output path for a run directory name."""
    path = Path(run_dir)
    if path.is_absolute():
        return path
    return RUNS_ROOT / path


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create a clean output directory for one run."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)


def count_lidar_steps(loader: VictoriaParkLoader, max_steps: int | None) -> int:
    """Return the number of LiDAR-to-LiDAR updates that will be processed."""
    available_steps = max(0, loader.lsr_timestamps.size - 2)
    if max_steps is None:
        return available_steps
    return min(max_steps, available_steps)


def run_single_experiment(
    spec: RunSpec,
    loader: VictoriaParkLoader,
    *,
    max_steps: int | None = MAX_STEPS,
    overwrite: bool = False,
    save_plots: bool = SAVE_PLOTS,
    show_plots: bool = SHOW_PLOTS,
) -> Path:
    """Run one SLAM experiment and save logs, config, metadata, and figures."""
    output_dir = resolve_run_dir(spec.run_dir)
    print(f"\n[run_multiple] Running {spec.run_dir} with {spec.config_file}")

    config = SlamConfig.load(spec.config_file)
    prepare_output_dir(output_dir, overwrite=overwrite)
    config.save(output_dir / "config.yaml")

    logger = SlamLogger(output_dir, config.logging)
    logger.save_reference(gnss=loader.gnss_filtered)
    slam = GraphSLAM(
        config=config,
        logger=logger,
        initial_pose=loader.initial_pose,
    )

    steps_diagnostics: list[StepDiagnostics] = []
    last_step: int | None = None
    total_steps = count_lidar_steps(loader, max_steps)

    t0 = time.perf_counter()
    for raw_meas in tqdm(
        loader.iterate(max_steps),
        total=total_steps,
        desc=str(spec.run_dir),
    ):
        meas = preprocess_victoria_park_step(
            raw_meas,
            odometry_covariance=config.noise.odom_cov_matrix,
            max_range=config.sensor.max_range,
        )
        diagnostics = slam.update(meas)
        steps_diagnostics.append(diagnostics)
        last_step = meas.scan_step

    total_time = time.perf_counter() - t0
    if last_step is None or not steps_diagnostics:
        raise RuntimeError(f"No SLAM steps were processed for {spec.run_dir}")

    logger.save_snapshot(last_step, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(steps_diagnostics)
    logger.save_metadata(steps_diagnostics[-1], total_time)

    if save_plots:
        from master_code.plotter import SlamRunPlotter

        plotter = SlamRunPlotter.from_run(output_dir)
        plotter.plot_all(save=True, show=show_plots)

    return output_dir


def validate_runs(runs: Sequence[RunSpec]) -> None:
    """Fail early for empty or ambiguous experiment lists."""
    if not runs:
        raise ValueError("No runs configured. Add RunSpec entries to RUNS.")

    resolved_dirs = [resolve_run_dir(spec.run_dir) for spec in runs]
    duplicates = {
        path for path in resolved_dirs if resolved_dirs.count(path) > 1
    }
    if duplicates:
        names = ", ".join(str(path) for path in sorted(duplicates))
        raise ValueError(f"Duplicate output directories in RUNS: {names}")


def select_runs(runs: Sequence[RunSpec], names: Sequence[str] | None) -> list[RunSpec]:
    """Optionally select a subset of configured runs by run directory name."""
    if not names:
        return list(runs)

    selected_names = set(names)
    selected = [
        spec for spec in runs
        if Path(spec.run_dir).name in selected_names or str(spec.run_dir) in selected_names
    ]
    found_names = {Path(spec.run_dir).name for spec in selected} | {
        str(spec.run_dir) for spec in selected
    }
    missing = selected_names - found_names
    if missing:
        raise ValueError(f"Unknown run name(s): {', '.join(sorted(missing))}")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run several SLAM experiments sequentially."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing run directories before writing new logs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum number of LiDAR steps per run. Use -1 for all available steps.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only save logs and metadata, skipping per-run figures.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show generated figures while saving them.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only the listed run_dir names from RUNS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be positive, or -1 for all available steps.")

    max_steps = None if args.max_steps < 0 else args.max_steps

    runs = select_runs(RUNS, args.only)
    validate_runs(runs)

    loader = VictoriaParkLoader()
    output_dirs = []

    for spec in runs:
        output_dir = run_single_experiment(
            spec,
            loader,
            max_steps=max_steps,
            overwrite=args.overwrite,
            save_plots=not args.no_plots,
            show_plots=args.show_plots,
        )
        output_dirs.append(output_dir)

    print("\n[run_multiple] Finished all runs:")
    for output_dir in output_dirs:
        print(f"  {output_dir}")


if __name__ == "__main__":
    main()
