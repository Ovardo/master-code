from __future__ import annotations

import numpy as np
import pytest

from master_code.logger import AssociationDiagnostics, SlamLogger


def test_association_diagnostics_serializes_prior_state_order() -> None:
    diagnostics = AssociationDiagnostics(
        scan_step=7,
        scan_time=1.5,
        pose_index=7,
        pose=np.zeros(3),
        measurements=np.empty((0, 2)),
        predicted_measurements=np.zeros((2, 2)),
        association=np.empty(0, dtype=int),
        local_landmarks=np.zeros((2, 2)),
        local_landmark_keys=np.array([11, 12]),
        prior_joint_covariance=np.eye(7),
        innovation_covariance=np.eye(4),
    )

    arrays = diagnostics.as_npz_dict()

    np.testing.assert_array_equal(arrays["local_landmark_keys"], [11, 12])
    np.testing.assert_array_equal(arrays["prior_joint_covariance"], np.eye(7))


def test_load_snapshot_by_step(tmp_path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    np.savez(snapshot_dir / "snap_00357.npz", step=np.array([357]))

    snapshot = SlamLogger.load_snapshot(tmp_path, 357)

    assert snapshot["step"][0] == 357


def test_load_association_diagnostics_by_step(tmp_path) -> None:
    association_dir = tmp_path / "associations"
    association_dir.mkdir()
    np.savez(association_dir / "assoc_00357.npz", scan_step=np.array([357]))

    diagnostics = SlamLogger.load_association_diagnostics(tmp_path, 357)

    assert diagnostics["scan_step"][0] == 357


@pytest.mark.parametrize(
    ("load", "description"),
    [
        (SlamLogger.load_snapshot, "snapshot"),
        (SlamLogger.load_association_diagnostics, "association diagnostics"),
    ],
)
def test_load_step_reports_missing_file(tmp_path, load, description) -> None:
    with pytest.raises(
        FileNotFoundError,
        match=rf"No {description} saved for step 12",
    ):
        load(tmp_path, 12)


@pytest.mark.parametrize(
    "step",
    [1.5, "12", True],
)
def test_load_step_requires_integer(tmp_path, step) -> None:
    with pytest.raises(TypeError, match="step must be an int"):
        SlamLogger.load_snapshot(tmp_path, step)


def test_load_step_rejects_negative_integer(tmp_path) -> None:
    with pytest.raises(ValueError, match="step must be non-negative"):
        SlamLogger.load_snapshot(tmp_path, -1)
