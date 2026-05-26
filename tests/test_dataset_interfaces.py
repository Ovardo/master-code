import numpy as np

from master_code.data_loader import RawLidarStepInput, SimulatedDataLoader, WheelOdometry
from master_code.measurements import SlamStepInput
from master_code.preprocessing import preprocess_victoria_park_step


def test_victoria_park_raw_step_preprocesses_to_slam_step() -> None:
    raw_step = RawLidarStepInput(
        odometry=[WheelOdometry(velocity=1.0, steering=0.0, dt=0.1)],
        scan=np.full(361, 100.0),
        scan_time=12.0,
        scan_step=7,
    )

    step = preprocess_victoria_park_step(
        raw_step,
        odometry_covariance=np.diag([0.1, 0.1, 0.01]),
        max_range=45.0,
    )

    assert isinstance(step, SlamStepInput)
    assert step.scan_step == 7
    assert step.scan_time == 12.0
    assert step.measurements.shape == (0, 2)
    assert step.odometry.covariance.shape == (3, 3)


def test_simulated_loader_yields_processed_steps_and_reference() -> None:
    loader = SimulatedDataLoader()
    covariance = np.diag([0.3, 0.3, 0.01])

    steps = list(
        loader.iterate(
            max_steps=2,
            odometry_covariance=covariance,
            max_range=80.0,
        )
    )

    assert len(loader.odometry) == 1000
    assert loader.poses_gt.shape == (1001, 3)
    assert loader.landmarks_gt.shape[1] == 2
    assert len(steps) == 2
    assert all(isinstance(step, SlamStepInput) for step in steps)
    assert steps[0].measurements.shape[1] == 2
    assert np.allclose(steps[0].measurements, loader.measurements[1])
    assert steps[0].scan_time == 1.0
    assert np.allclose(steps[0].odometry.covariance, covariance)
    assert set(loader.reference) == {"poses_gt", "landmarks_gt"}
