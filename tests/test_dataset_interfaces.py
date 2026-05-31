from __future__ import annotations

import gtsam
import numpy as np

from master_code.config import SlamConfig
from master_code.loaders.simulated import SimulatedDataLoader
from master_code.loaders.victoria_park import VictoriaParkLoader
from master_code.logger import SlamLogger
from master_code.run_real import run_real
from master_code.run_sim import run_sim
from master_code.slam import SlamStepInput


def _load_quiet_config(name: str) -> SlamConfig:
    config = SlamConfig.load(name)
    config.logging.log_association_diagnostics = False
    config.logging.log_snapshot = False
    return config


def _assert_slam_step(step: SlamStepInput) -> None:
    assert isinstance(step, SlamStepInput)
    assert isinstance(step.relative_pose, gtsam.Pose2)
    assert step.relative_pose_cov.shape == (3, 3)
    assert step.measurements.ndim == 2
    assert step.measurements.shape[1] == 2
    assert np.isfinite(step.scan_time)


def test_victoria_park_loader_yields_normalized_slam_steps() -> None:
    config = _load_quiet_config("real.yaml")
    dataset = VictoriaParkLoader()

    steps = list(dataset.iterate_slam(config, 2))

    assert len(steps) <= 2
    assert len(steps) > 0
    for step in steps:
        _assert_slam_step(step)


def test_simulated_loader_yields_normalized_slam_steps() -> None:
    config = _load_quiet_config("sim.yaml")
    dataset = SimulatedDataLoader()

    steps = list(dataset.iterate_slam(config, 2))

    assert len(steps) == 2
    for step in steps:
        _assert_slam_step(step)


def test_simulated_slam_iteration_hides_loader_offset() -> None:
    config = _load_quiet_config("sim.yaml")
    dataset = SimulatedDataLoader()

    assert len(list(dataset.iterate_slam(config, 3))) == 3


def test_one_step_runners_smoke(tmp_path) -> None:
    run_real(
        config=_load_quiet_config("real.yaml"),
        output_dir=tmp_path / "real",
        num_steps=1,
        show_plots=False,
        save_plots=False,
    )
    run_sim(
        config=_load_quiet_config("sim.yaml"),
        output_dir=tmp_path / "sim",
        num_steps=1,
        show_plots=False,
        save_plots=False,
    )

    assert SlamLogger.load_metadata(tmp_path / "real")["dataset"] == "victoria_park"
    assert SlamLogger.load_metadata(tmp_path / "sim")["dataset"] == "sim"
