from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from master_code.config import SlamConfig


@dataclass
class SupportingObservation:
    step: int
    measurement: np.ndarray # [range, bearing]

@dataclass
class TentativeLandmark:
    position: np.ndarray
    supporting_observations: list[SupportingObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(2)

    @property
    def hit_count(self) -> int:
        return len(self.supporting_observations)

    @property
    def last_seen_step(self) -> int:
        return self.supporting_observations[-1].step

    def update(
        self,
        new_position: np.ndarray,
        step: int,
        measurement: np.ndarray,
    ) -> None:
        """
        Update tentative landmark with a new supporting observation.

        Uses a simple running average for position. This is intentionally simple;
        it can later be replaced with a covariance-weighted update if desired.

        At most one observation is recorded per timestep: the manager never matches
        a landmark twice in the same step (see ``_find_best_match``), so every entry
        in ``supporting_observations`` corresponds to a distinct timestep.
        """
        new_position = np.asarray(new_position, dtype=float).reshape(2)
        measurement = np.asarray(measurement, dtype=float).reshape(2)

        self.supporting_observations.append(
            SupportingObservation(step=step, measurement=measurement)
        )

        # Simple running average update
        alpha = 1.0 / self.hit_count
        self.position = (1.0 - alpha) * self.position + alpha * new_position

    def steps_since_seen(self, current_step: int) -> int:
        return current_step - self.last_seen_step

    def is_confirmed(self, current_step: int, M: int, N: int) -> bool:
        """
        Sliding-window M/N logic:
        Confirm if landmark has been observed in at least M distinct timesteps
        during the last N timesteps (inclusive of current_step).
        """
        window_start = current_step - N + 1
        hits_in_window = sum(
            obs.step >= window_start for obs in self.supporting_observations
        )
        return hits_in_window >= M


def get_tentative_landmark_manager(cfg: SlamConfig) -> TentativeLandmarkManager:
    """Factory function to create a TentativeLandmarkManager based on the config."""
    return TentativeLandmarkManager(
        M=cfg.tentative.M,
        N=cfg.tentative.N,
        gate=cfg.tentative.gate,
    )

@dataclass
class TentativeLandmarkManager:
    """
    Manager for tentative landmarks that have been observed but not yet confirmed
    as global landmarks.

    Workflow:
      1. Unassociated measurements are transformed to world positions.
      2. Each such measurement is either matched to an existing tentative landmark
         or used to spawn a new one.
      3. Tentative landmarks are promoted once they satisfy M/N confirmation logic.
      4. Stale tentative landmarks are pruned.
    """

    def __init__(self, M, N, gate) -> None:
        
        assert M > 0, "M must be > 0"
        assert N > 0, "N must be > 0"
        assert M <= N, "M must be <= N"
        assert gate > 0, "association_gate must be > 0"

        self.M = M
        self.N = N
        self.association_gate = gate
        self.tentative_landmarks: list[TentativeLandmark] = []

    def add_tentative_landmarks(
        self,
        current_step: int,
        unassociated_measurements: np.ndarray,
        new_tentative_landmarks: np.ndarray,
    ) -> list[TentativeLandmark]:
        """
        Main entry point for one timestep.

        Parameters
        ----------
        current_step : int
            Current timestep.
        unassociated_measurements : np.ndarray, shape (M, 2)
            Raw unassociated measurements in (range, bearing) form.
        new_tentativ_landmarks : np.ndarray, shape (M, 2)
            Unassociated measurements transformed to world-frame 2D positions.

        Returns
        -------
        confirmed_landmarks : list[TentativeLandmark]
            Tentative landmarks that are ready to be promoted to graph landmarks.
        """
        M = unassociated_measurements.shape[0]
        if new_tentative_landmarks.shape[0] != M:
            raise ValueError(f"Expected new_tentative_landmarks to have shape ({M}, 2), got {new_tentative_landmarks.shape}")
    
        for i in range(M):
            W_l = new_tentative_landmarks[i]
            z = unassociated_measurements[i]

            match_idx = self._find_best_match(W_l, current_step)

            # Landmark not matched to existing tentative, spawn new one. Otherwise, update matched tentative.
            if match_idx is None:
                self._spawn_tentative(
                    step=current_step,
                    position=W_l,
                    measurement=z,
                )
            else:
                self.tentative_landmarks[match_idx].update(
                    step=current_step,
                    new_position=W_l,
                    measurement=z,
                )

        confirmed = self._extract_confirmed(current_step)
        self.prune_unconfirmable(current_step)

        return confirmed

    def _find_best_match(self, measurement_position: np.ndarray, current_step: int) -> int | None:
        """
        Find nearest tentative landmark within association gate.

        Returns index into self.tentative_landmarks, or None if no valid match found.
        """
        best_idx = None
        best_dist = np.inf

        for i, landmark in enumerate(self.tentative_landmarks):
            # skip landmarks already updated this step
            if landmark.last_seen_step == current_step:
                continue

            dist = np.linalg.norm(measurement_position - landmark.position)

            if dist < self.association_gate and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _spawn_tentative(
        self,
        step: int,
        position: np.ndarray,
        measurement: np.ndarray,
    ) -> None:
        landmark = TentativeLandmark(
            position=position,
            supporting_observations=[SupportingObservation(step, measurement)],
        )
        self.tentative_landmarks.append(landmark)

    def _extract_confirmed(self, current_step: int) -> list[TentativeLandmark]:
        """
        Return tentative landmarks that satisfy confirmation logic and remove 
        them from tentative landmarks.
        """
        confirmed: list[TentativeLandmark] = []
        remaining: list[TentativeLandmark] = []

        for landmark in self.tentative_landmarks:
            if landmark.is_confirmed(current_step, self.M, self.N):
                confirmed.append(landmark)
            else:
                remaining.append(landmark)

        self.tentative_landmarks = remaining
        return confirmed

    def prune_unconfirmable(self, current_step: int) -> None:
        """
        Remove tentative landmarks that can no longer reach M observations within
        their N-step confirmation window.
        """
        self.tentative_landmarks = [
            lm for lm in self.tentative_landmarks
            if lm.steps_since_seen(current_step) <= self.N
        ]

    def reset(self) -> None:
        """Remove all tentative landmarks."""
        self.tentative_landmarks.clear()

    def __len__(self) -> int:
        return len(self.tentative_landmarks)
