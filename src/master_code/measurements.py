from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np


@dataclass(slots=True)
class RelativePoseMeasurement:
    """Relative pose measurement between consecutive SLAM poses."""

    pose: gtsam.Pose2
    covariance: np.ndarray


@dataclass(slots=True)
class SlamStepInput:
    """Processed input for one factor-graph SLAM update."""

    odometry: RelativePoseMeasurement
    measurements: np.ndarray
    scan_step: int
    scan_time: float = np.nan
