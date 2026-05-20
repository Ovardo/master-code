from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import SlamConfig


@dataclass
class SupportingObservation:
    step: int
    measurement: np.ndarray # [range, bearing]

@dataclass
class TentativeLandmark:
    position: np.ndarray
    birth_step: int
    last_seen_step: int
    hit_count: int = 1
    observed_steps: set[int] = field(default_factory=set)
    supporting_observations: list[SupportingObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(2)
        self.observed_steps.add(self.birth_step)

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
        """
        new_position = np.asarray(new_position, dtype=float).reshape(2)
        measurement = np.asarray(measurement, dtype=float).reshape(2)

        # Only count as a new hit if observed at a new timestep
        if step not in self.observed_steps:
            self.hit_count += 1
            self.observed_steps.add(step)

        # Simple running average update
        alpha = 1.0 / self.hit_count
        self.position = (1.0 - alpha) * self.position + alpha * new_position

        self.last_seen_step = step
        self.supporting_observations.append(
            SupportingObservation(
                step=step, 
                measurement=measurement,
            )
        )

    def age(self, current_step: int) -> int:
        return current_step - self.birth_step

    def steps_since_seen(self, current_step: int) -> int:
        return current_step - self.last_seen_step

    def is_confirmed(self, M: int) -> bool:
        """
        M/N logic:
        Confirm if landmark has been observed in at least M distinct timesteps
        during the last N timesteps (inclusive of current_step).
        """
        # # Fixed lifetime interpretation:
        return self.hit_count >= M # ()

        # # Sliding window intepreation: ]
        # window_start = current_step - N + 1
        # hits_in_window = sum(step >= window_start for step in self.observed_steps)
        # return hits_in_window >= M
    
    def can_still_be_confirmed(self, current_step: int, M: int, N: int) -> bool:
        """
        Return True if it is still possible for this tentative landmark to reach
        M observations within its N-step tentative lifetime.
        """
        steps_used = self.age(current_step) + 1
        future_steps_left = N - steps_used
        max_possible_hits = self.hit_count + max(0, future_steps_left)
        return max_possible_hits >= M
    

def get_tentative_landmark_manager(cfg: SlamConfig) -> TentativeLandmarkManager:
    """Factory function to create a TentativeLandmarkManager based on the config."""
    return TentativeLandmarkManager(
        M=cfg.tentative.M,
        N=cfg.tentative.N,
        gate=cfg.tentative.gate,
    )

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

    def process_unassociated_measurements(
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
        if new_tentative_landmarks.shape != (M, 2):
            raise ValueError(f"Expected new_tentative_landmarks to have shape ({M}, 2), got {new_tentative_landmarks.shape}")
    
        for i in range(M):
            W_l = new_tentative_landmarks[i]
            z = unassociated_measurements[i]

            match_idx = self._find_best_match(W_l, current_step)

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
            # Optional: skip landmarks already updated this step
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
            birth_step=step,
            last_seen_step=step,
            hit_count=1,
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
            if landmark.is_confirmed(self.M):
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
        self.tentative_landmarks = [lm for lm in self.tentative_landmarks
            if lm.can_still_be_confirmed(current_step, self.M, self.N)
        ]

        # # for sliding window interpretation:
        # self.tentative_landmarks = [
        #     lm for lm in self.tentative_landmarks
        #     if lm.steps_since_seen(current_step) <= self.max_missed_steps
        # ]

    def reset(self) -> None:
        """Remove all tentative landmarks."""
        self.tentative_landmarks.clear()

    def __len__(self) -> int:
        return len(self.tentative_landmarks)
