from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


UNASSOCIATED = -1
AMBIGUOUS = -2


@dataclass(slots=True)
class SLAMStepInput:
    """Runtime input consumed by a single SLAM update."""

    step_index: int
    timestamp: float | None
    ve_dr: float
    alpha_dr: float
    dt_dr: float
    z_lsr: np.ndarray | None
    odometry: np.ndarray
    measurements: np.ndarray


@dataclass(slots=True)
class StepReference:
    """Optional reference data attached to a step for evaluation or simulation."""

    ground_truth_pose: np.ndarray | None = None
    ground_truth_landmarks: np.ndarray | None = None
    landmark_ids_by_measurement_gt: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateEstimate:
    """Estimated SLAM state produced by one update."""

    robot_poses: np.ndarray | None = None  # (K, 3)
    robot_pose_covariances: list[np.ndarray] | None = None  # (K, 3, 3)
    landmark_positions: np.ndarray | None = None  # (L, 2)
    landmark_covariances: list[np.ndarray] | None = None  # (L, 2, 2)
    current_robot_pose: np.ndarray | None = None  # (3,)
    predicted_robot_pose: np.ndarray | None = None  # (3,)


@dataclass(slots=True)
class MeasurementPrediction:
    """Observed measurements and the landmark predictions they were matched against."""

    observed_measurements: np.ndarray | None = None  # (M, 2)
    predicted_measurements: np.ndarray | None = None  # (L', 2)
    predicted_landmark_ids: np.ndarray | None = None  # (L',)


@dataclass(slots=True)
class DataAssociationResult:
    """Association outputs for one SLAM update."""

    landmark_ids_by_measurement: np.ndarray | None = None  # (M,)
    prediction_indices_by_measurement: np.ndarray | None = None  # (M,)


@dataclass(slots=True)
class StepDiagnostics:
    """Optional diagnostics produced by one SLAM update."""

    innovation_covariance: np.ndarray | None = None  # (2L', 2L')
    current_pose_covariance: np.ndarray | None = None  # (3, 3)


@dataclass(slots=True)
class SLAMStepOutput:
    """Algorithm-facing output returned by a single SLAM update."""

    estimate: StateEstimate
    measurement_prediction: MeasurementPrediction
    associations: DataAssociationResult
    diagnostics: StepDiagnostics


@dataclass(slots=True)
class SLAMHistoryEntry:
    """Stored per-step record used for analysis and visualization."""

    step_index: int
    step_input: SLAMStepInput
    step_output: SLAMStepOutput
    reference: StepReference | None = None
    dead_reckoning_poses: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
