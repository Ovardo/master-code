from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import SLAMConfig, VisualizationConfig
from experiment_results import (
    CURRENT_SCHEMA_VERSION,
    ExperimentReferenceData,
    ExperimentResult,
    load_result,
    save_result,
)
from history import SLAMHistory
from slam_types import (
    UNASSOCIATED,
    DataAssociationResult,
    MeasurementPrediction,
    SLAMHistoryEntry,
    SLAMStepInput,
    SLAMStepOutput,
    StateEstimate,
    StepDiagnostics,
    StepReference,
)
from timing_profiler import TimingProfiler


def make_step_input(step_index: int) -> SLAMStepInput:
    return SLAMStepInput(
        step_index=step_index,
        timestamp=100.0 + step_index,
        ve_dr=1.2,
        alpha_dr=0.05,
        dt_dr=0.1,
        z_lsr=np.array([1.0, 2.0, 3.0], dtype=float),
        odometry=np.array([0.1, 0.0, 0.01], dtype=float),
        measurements=np.array([[2.0, 0.1], [3.0, -0.2]], dtype=float),
    )


def make_step_output() -> SLAMStepOutput:
    return SLAMStepOutput(
        estimate=StateEstimate(
            robot_poses=np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]], dtype=float),
            landmark_positions=np.array([[2.0, 1.0], [4.0, -1.0]], dtype=float),
            current_robot_pose=np.array([1.0, 0.2, 0.1], dtype=float),
            predicted_robot_pose=np.array([1.1, 0.25, 0.12], dtype=float),
        ),
        measurement_prediction=MeasurementPrediction(
            observed_measurements=np.array([[2.0, 0.1], [3.0, -0.2]], dtype=float),
            predicted_measurements=np.array([[1.9, 0.12]], dtype=float),
            predicted_landmark_ids=np.array([5], dtype=int),
        ),
        associations=DataAssociationResult(
            landmark_ids_by_measurement=np.array([5, UNASSOCIATED], dtype=int),
            prediction_indices_by_measurement=np.array([0, UNASSOCIATED], dtype=int),
        ),
        diagnostics=StepDiagnostics(
            innovation_covariance=np.eye(2),
            current_pose_covariance=np.eye(3) * 4.0,
        ),
    )


def make_history_entry(step_index: int) -> SLAMHistoryEntry:
    return SLAMHistoryEntry(
        step_index=step_index,
        step_input=make_step_input(step_index),
        step_output=make_step_output(),
        reference=StepReference(
            ground_truth_pose=np.array([1.0, 2.0, 0.5], dtype=float),
            ground_truth_landmarks=np.array([[5.0, 6.0]], dtype=float),
            landmark_ids_by_measurement_gt=np.array([5, UNASSOCIATED], dtype=int),
            metadata={"source": "simulation"},
        ),
        dead_reckoning_poses=np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.08]], dtype=float),
    )


def make_history() -> SLAMHistory:
    history = SLAMHistory()
    history.add(make_history_entry(step_index=3))
    return history


class ExperimentResultsTests(unittest.TestCase):
    def test_save_and_load_round_trip_with_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "results_root"
            config = SLAMConfig(
                name="Experiment Demo",
                visualization=VisualizationConfig(output_dir=str(output_dir)),
            )
            history = make_history()
            profiler = TimingProfiler(enabled=True)
            profiler.record("process_step", 15.0, iteration=8)
            reference_data = ExperimentReferenceData(
                ground_truth_poses=np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.1]], dtype=float),
                ground_truth_landmarks=np.array([[5.0, 6.0]], dtype=float),
                gps_track=np.array([[10.0, 11.0, 12.0]], dtype=float),
                metadata={"dataset": "victoria_park"},
            )

            path = save_result(
                config=config,
                history=history,
                profiler=profiler,
                reference_data=reference_data,
            )

            self.assertTrue(path.exists())
            self.assertEqual(path.parent, output_dir / "experiments")
            self.assertEqual(path.suffix, ".pkl")
            self.assertRegex(path.name, r"^\d{8}T\d{6}Z_Experiment_Demo\.pkl$")

            loaded = load_result(path)

            self.assertEqual(loaded.schema_version, CURRENT_SCHEMA_VERSION)
            self.assertTrue(loaded.created_at_utc.endswith("Z"))
            self.assertEqual(loaded.config.name, "Experiment Demo")
            self.assertEqual(len(loaded.history), 1)
            self.assertIsNotNone(loaded.profiler)
            self.assertEqual(len(loaded.profiler.events), 1)
            self.assertEqual(loaded.profiler.events[0].name, "process_step")
            self.assertEqual(loaded.profiler.events[0].iteration, 8)
            self.assertIsNotNone(loaded.reference_data)
            self.assertEqual(loaded.reference_data.metadata["dataset"], "victoria_park")
            np.testing.assert_array_equal(
                loaded.reference_data.gps_track,
                np.array([[10.0, 11.0, 12.0]], dtype=float),
            )
            np.testing.assert_array_equal(
                loaded.history.latest_or_raise().reference.ground_truth_pose,
                np.array([1.0, 2.0, 0.5], dtype=float),
            )

    def test_save_result_omits_disabled_or_missing_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "results_root"
            config = SLAMConfig(
                name="Profiler Test",
                visualization=VisualizationConfig(output_dir=str(output_dir)),
            )
            history = make_history()

            disabled_path = save_result(
                config=config,
                history=history,
                profiler=TimingProfiler(enabled=False),
                path=output_dir / "disabled_profiler.pkl",
            )
            missing_path = save_result(
                config=config,
                history=history,
                profiler=None,
                path=output_dir / "missing_profiler.pkl",
            )

            self.assertIsNone(load_result(disabled_path).profiler)
            self.assertIsNone(load_result(missing_path).profiler)

    def test_load_result_rejects_wrong_payload_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid_payload.pkl"
            with path.open("wb") as handle:
                pickle.dump({"payload": "not-an-experiment-result"}, handle)

            with self.assertRaisesRegex(TypeError, "Expected ExperimentResult payload"):
                load_result(path)

    def test_load_result_rejects_unsupported_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "unsupported_schema.pkl"
            payload = ExperimentResult(
                created_at_utc="2026-03-30T12:00:00Z",
                config=SLAMConfig(),
                history=make_history(),
                profiler=None,
                reference_data=None,
                schema_version=CURRENT_SCHEMA_VERSION + 1,
            )
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with self.assertRaisesRegex(ValueError, "Unsupported experiment result schema_version"):
                load_result(path)


if __name__ == "__main__":
    unittest.main()
