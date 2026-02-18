from dataclasses import dataclass
from typing import Iterable, Optional

import gtsam
import numpy as np

from utils.utils_types import PredictedMeasurement


@dataclass
class StepRecord:
    """Record of SLAM state and computations at a single timestep."""
    
    step: int

    # state snapshots
    poses: Optional[list[gtsam.Pose2]] = None  # (step+1, 3) 
    landmarks: Optional[gtsam.Point2] = None   # (L, 2)

    # measurements + prediction
    measurements: Optional[list[tuple[float, gtsam.Rot2]]] = None  # (M, 2) [range, bearing]
    predicted_measurements: Optional[list[PredictedMeasurement]] = None  # (L',2) [range, bearing]
    predicted_pose: Optional[np.ndarray] = None # (3,)

    # associations
    associations_ids: Optional[np.ndarray] = None  # (M,) -1 for unassociated, >=0 for index of global landmarks (L)
    associations_idx: Optional[np.ndarray] = None  # (M,) -1 for unassociated, >=0 for index of predicted measurements (L')
    
    # Optional future fields (nice to have for JCBB analysis)
    cov_innovation: Optional[np.ndarray] = None  # (L', L') full innovation covariance
    cov_current_pose: Optional[np.ndarray] = None  # (3,3) covariance of last pose (pose k)


class SLAMHistory:
    """In-memory per-step history: one StepRecord per step."""

    def __init__(self):
        self._records: dict[int, StepRecord] = {}

    def add(self, record: StepRecord, *, overwrite: bool = False) -> None:
        k = int(record.step)
        if (not overwrite) and (k in self._records):
            raise KeyError(f"Step {k} already exists.")
        self._records[k] = record

    def get(self, step: int) -> Optional[StepRecord]:
        if step == -1:
            step = max(self._records.keys())  # get last step
        return self._records.get(step)

    def get_or_raise(self, step: int) -> StepRecord:
        rec = self.get(step)
        if rec is None:
            raise KeyError(f"No record for step={step}")
        return rec

    @property
    def steps(self) -> list[int]:
        return sorted(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    def all_records(self) -> list[StepRecord]:
        return [self._records[k] for k in self.steps]

    def iter_records(self) -> Iterable[StepRecord]:
        for k in self.steps:
            yield self._records[k]
