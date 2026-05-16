from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logger import SlamLogger


def _snapshot(num_poses: int = 1, num_landmarks: int = 0) -> dict[str, np.ndarray]:
    return {
        "poses": np.zeros((num_poses, 3)),
        "poses_covariance": np.zeros((num_poses, 3, 3)),
        "landmarks": np.zeros((num_landmarks, 2)),
        "landmarks_covariance": np.zeros((num_landmarks, 2, 2)),
    }


def test_pending_metrics_flush_to_compatible_step_data(tmp_path: Path) -> None:
    logger = SlamLogger(tmp_path, snapshot_every=0)

    logger.log_time("total", 1.2)
    logger.log_time("association", 0.3)
    logger.log_count("local_landmarks", 4)
    logger.log_count("total_landmarks", 7)
    logger.log_value("innovation_norm", 2.5)
    logger.log_step(step=2)
    logger.save(_snapshot(num_poses=2, num_landmarks=7))

    data = SlamLogger.load(tmp_path)

    assert data["steps"].tolist() == [2]
    assert data["time_total"].tolist() == [1.2]
    assert data["time_association"].tolist() == [0.3]
    assert data["count_local_landmarks"].tolist() == [4]
    assert data["count_total_landmarks"].tolist() == [7]
    assert data["value_innovation_norm"].tolist() == [2.5]


def test_accumulated_timing_is_summed_within_step(tmp_path: Path) -> None:
    logger = SlamLogger(tmp_path)

    logger.log_time("optimization", 0.1, accumulate=True)
    logger.log_time("optimization", 0.25, accumulate=True)
    logger.log_step(step=1)

    assert np.isclose(logger._times["optimization"][0], 0.35)


def test_pending_buffer_clears_after_log_step(tmp_path: Path) -> None:
    logger = SlamLogger(tmp_path)

    logger.log_time("total", 1.0)
    logger.log_count("local_landmarks", 3)
    logger.log_step(step=1)
    logger.log_step(step=2)

    assert logger._pending_times == {}
    assert logger._pending_counts == {}
    assert logger._times["total"][0] == 1.0
    assert np.isnan(logger._times["total"][1])
    assert logger._counts["local_landmarks"][0] == 3
    assert np.isnan(logger._counts["local_landmarks"][1])


def test_missing_metrics_keep_columns_step_aligned(tmp_path: Path) -> None:
    logger = SlamLogger(tmp_path, snapshot_every=0)

    logger.log_time("total", 1.0)
    logger.log_step(step=1)
    logger.log_time("association", 0.2)
    logger.log_step(step=2)
    logger.save(_snapshot(num_poses=2))

    data = SlamLogger.load(tmp_path)

    assert data["steps"].tolist() == [1, 2]
    assert data["time_total"].shape == (2,)
    assert data["time_association"].shape == (2,)
    assert data["time_total"][0] == 1.0
    assert np.isnan(data["time_total"][1])
    assert np.isnan(data["time_association"][0])
    assert data["time_association"][1] == 0.2


def test_maybe_log_snapshot_only_writes_configured_steps(tmp_path: Path) -> None:
    logger = SlamLogger(tmp_path, snapshot_every=3)
    calls = 0

    def snapshot_fn() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return _snapshot(num_poses=3)

    logger.maybe_log_snapshot(1, snapshot_fn)
    logger.maybe_log_snapshot(2, snapshot_fn)
    logger.maybe_log_snapshot(3, snapshot_fn)

    assert calls == 1
    assert not (tmp_path / "snapshots" / "step_000001.npz").exists()
    assert (tmp_path / "snapshots" / "step_000003.npz").exists()
