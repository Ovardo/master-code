from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import VisualizationConfig
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
from visualization import SLAMVisualizer


def make_step_input(step_index: int) -> SLAMStepInput:
    return SLAMStepInput(
        step_index=step_index,
        timestamp=12.5 + step_index,
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
            robot_pose_covariances=[np.eye(3), np.eye(3) * 2.0],
            landmark_positions=np.array([[2.0, 1.0], [4.0, -1.0]], dtype=float),
            landmark_covariances=[np.eye(2), np.eye(2) * 3.0],
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
        ),
        dead_reckoning_poses=np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.08]], dtype=float),
    )


class SLAMContractTests(unittest.TestCase):
    def test_step_input_and_reference_construction(self) -> None:
        step_input = make_step_input(step_index=7)
        reference = StepReference(
            ground_truth_pose=np.array([1.0, 2.0, 0.5], dtype=float),
            metadata={"source": "simulation"},
        )

        self.assertEqual(step_input.step_index, 7)
        self.assertEqual(step_input.timestamp, 19.5)
        np.testing.assert_array_equal(step_input.odometry, np.array([0.1, 0.0, 0.01], dtype=float))
        np.testing.assert_array_equal(reference.ground_truth_pose, np.array([1.0, 2.0, 0.5], dtype=float))
        self.assertEqual(reference.metadata["source"], "simulation")

    def test_step_output_and_history_entry_construction(self) -> None:
        entry = make_history_entry(step_index=3)

        self.assertEqual(entry.step_index, 3)
        np.testing.assert_array_equal(
            entry.step_output.estimate.predicted_robot_pose,
            np.array([1.1, 0.25, 0.12], dtype=float),
        )
        np.testing.assert_array_equal(
            entry.step_output.measurement_prediction.predicted_landmark_ids,
            np.array([5], dtype=int),
        )
        np.testing.assert_array_equal(
            entry.step_output.associations.landmark_ids_by_measurement,
            np.array([5, UNASSOCIATED], dtype=int),
        )
        np.testing.assert_array_equal(
            entry.dead_reckoning_poses,
            np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.08]], dtype=float),
        )


class SLAMHistoryTests(unittest.TestCase):
    def test_add_order_and_latest_lookup(self) -> None:
        history = SLAMHistory()
        later_entry = make_history_entry(step_index=4)
        earlier_entry = make_history_entry(step_index=2)

        history.add(later_entry)
        history.add(earlier_entry)

        self.assertEqual(history.step_indices, [2, 4])
        self.assertIs(history.require(2), earlier_entry)
        self.assertIs(history.latest(), later_entry)
        self.assertIs(history.latest_or_raise(), later_entry)
        self.assertEqual([entry.step_index for entry in history.iter_entries()], [2, 4])
        self.assertEqual([entry.step_index for entry in history.all_entries()], [2, 4])
        self.assertIsNone(history.get(-1))

    def test_add_rejects_duplicates_without_overwrite(self) -> None:
        history = SLAMHistory()
        history.add(make_history_entry(step_index=3))

        with self.assertRaises(KeyError):
            history.add(make_history_entry(step_index=3))

        replacement_entry = make_history_entry(step_index=3)
        history.add(replacement_entry, overwrite=True)
        self.assertIs(history.require(3), replacement_entry)

    def test_empty_history_failure_paths(self) -> None:
        history = SLAMHistory()

        self.assertIsNone(history.latest())
        with self.assertRaises(KeyError):
            history.latest_or_raise()
        with self.assertRaises(KeyError):
            history.require(0)


class SLAMVisualizerSmokeTests(unittest.TestCase):
    def test_plot_methods_accept_history_entries(self) -> None:
        history = SLAMHistory()
        history.add(make_history_entry(step_index=3))
        visualizer = SLAMVisualizer(VisualizationConfig(), history)

        fig_est, ax_est = plt.subplots()
        returned_ax_est = visualizer.plot_estimates(step=-1, ax=ax_est, plot_dead_reckoning=True)
        self.assertIs(returned_ax_est, ax_est)
        plt.close(fig_est)

        fig_polar, ax_polar = plt.subplots()
        returned_ax_polar = visualizer.plot_measurements_polar(step=-1, ax=ax_polar)
        self.assertIs(returned_ax_polar, ax_polar)
        plt.close(fig_polar)

        fig_cart, ax_cart = plt.subplots()
        returned_ax_cart = visualizer.plot_measurements_cartesian(step=-1, ax=ax_cart)
        self.assertIs(returned_ax_cart, ax_cart)
        plt.close(fig_cart)


if __name__ == "__main__":
    unittest.main()
