from pathlib import Path

import numpy as np

from master_code.config import SlamConfig
from master_code.loaders.simulated import SimulatedDataLoader


def test_simulated_loader_produces_slam_input() -> None:
    loader = SimulatedDataLoader(Path("data/simulated/simulatedSLAM.mat"))
    config = SlamConfig.load(Path("configs/default_sim.yaml"))

    first_step = next(loader.iterate_slam(config, max_steps=1))

    assert first_step.relative_pose is None
    assert first_step.relative_pose_cov.shape == (3, 3)
    assert first_step.measurements.ndim == 2
    assert first_step.measurements.shape[1] == 2
    assert np.isfinite(first_step.measurements).all()
