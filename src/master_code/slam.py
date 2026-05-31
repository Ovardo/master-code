from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
from tqdm import tqdm

from master_code.config import SlamConfig
from master_code.data_association import JCBB_association
from master_code.landmark_manager import get_tentative_landmark_manager
from master_code.logger import AssociationDiagnostics, SlamLogger, StepDiagnostics
from master_code.utils import pose2_to_array, reorder_covariance_naive


@dataclass(slots=True)
class SlamStepInput:
    relative_pose: gtsam.Pose2
    relative_pose_cov: np.ndarray
    measurements: np.ndarray
    scan_time: float


class SlamDataset(Protocol):
    name: str

    @property
    def initial_pose(self) -> np.ndarray:
        ...

    @property
    def max_steps(self) -> int:
        ...

    def iterate_slam(
        self,
        config: SlamConfig,
        num_steps: int | None = None,
    ) -> Iterator[SlamStepInput]:
        ...


@dataclass
class SlamState:
    isam2: gtsam.ISAM2
    new_factors: gtsam.NonlinearFactorGraph
    new_values: gtsam.Values
    pose_keys: list[int]
    landmark_keys: list[int]

    def update_and_clear(self) -> None:
        self.isam2.update(self.new_factors, self.new_values)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

    def get_poses(self) -> np.ndarray:
        return np.array([pose2_to_array(self.isam2.calculateEstimatePose2(key)) for key in self.pose_keys])

    def get_poses_covariance(self) -> np.ndarray:
        return np.stack([self.isam2.marginalCovariance(k) for k in self.pose_keys], axis=0)

    def get_landmarks(self) -> np.ndarray:
        return np.array([self.isam2.calculateEstimatePoint2(key) for key in self.landmark_keys])

    def get_landmarks_covariance(self) -> np.ndarray:
        covariances = [self.isam2.marginalCovariance(key) for key in self.landmark_keys]
        return np.stack(covariances, axis=0) if covariances else np.array([])

    def get_snapshot(self) -> dict:
        return {
            "poses": self.get_poses(),
            "poses_covariance": self.get_poses_covariance(),
            "landmarks": self.get_landmarks(),
            "landmarks_covariance": self.get_landmarks_covariance(),
        }


def run_slam(
    config: SlamConfig,
    dataset: SlamDataset,
    output_dir: Path,
    num_steps: int | None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    logger = SlamLogger(output_dir, config.logging)

    if num_steps is None:
        num_steps = dataset.max_steps
    else:
        num_steps = min(dataset.max_steps, num_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    slam = SlamState(
        isam2=gtsam.ISAM2(),
        new_factors=gtsam.NonlinearFactorGraph(),
        new_values=gtsam.Values(),
        pose_keys=[],
        landmark_keys=[],
    )

    manager = get_tentative_landmark_manager(config)

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([
        config.noise.sigma_init_pose_x,
        config.noise.sigma_init_pose_y,
        config.noise.sigma_init_pose_yaw_rad,
    ])

    # GTSAM's factor constructor expects bearing noise before range noise.
    bearing_range_noise = gtsam.noiseModel.Diagonal.Sigmas([
        config.noise.sigma_bearing_rad,
        config.noise.sigma_range,
    ])

    diagnostics_steps = []

    prior_mean = gtsam.Pose2(*dataset.initial_pose)

    slam.new_values.insert(X(0), prior_mean)
    slam.pose_keys.append(X(0))

    slam.new_factors.add(
        gtsam.PriorFactorPose2(
            key=X(0),
            prior=prior_mean,
            noiseModel=prior_noise,
        )
    )

    slam.update_and_clear()

    t_run_start = time.perf_counter()

    for k, meas in tqdm(enumerate(dataset.iterate_slam(config, num_steps)), total=num_steps, desc="SLAM"):
        diagnostics = StepDiagnostics()
        t_step = time.perf_counter()

        delta_T = meas.relative_pose
        delta_cov = meas.relative_pose_cov

        key_k = X(k)
        key_kp1 = X(k + 1)
        slam.pose_keys.append(key_kp1)

        T_k = slam.isam2.calculateEstimatePose2(key_k)
        T_kp1 = T_k.compose(delta_T)
        slam.new_values.insert(key_kp1, T_kp1)

        delta_cov_model = gtsam.noiseModel.Gaussian.Covariance(delta_cov)
        slam.new_factors.add(
            gtsam.BetweenFactorPose2(
                key1=key_k,
                key2=key_kp1,
                relativePose=delta_T,
                noiseModel=delta_cov_model,
            )
        )

        t0 = time.perf_counter()
        slam.update_and_clear()
        diagnostics.add_time("duration_optimization", time.perf_counter() - t0)

        measurements = meas.measurements

        local_landmarks = []
        local_landmarks_keys = []
        local_predicted_measurements = []
        local_jacobians_pose = []
        local_jacobians_landmarks = []

        for lm_key in slam.landmark_keys:
            lm = slam.isam2.calculateEstimatePoint2(lm_key)

            J_r_T = np.zeros((1, 3), order="F")
            J_r_lm = np.zeros((1, 2), order="F")
            J_b_T = np.zeros((1, 3), order="F")
            J_b_lm = np.zeros((1, 2), order="F")

            r = T_kp1.range(lm, J_r_T, J_r_lm)
            b = T_kp1.bearing(lm, J_b_T, J_b_lm).theta()

            if r < config.sensor.max_range and abs(b) < np.deg2rad(config.sensor.fov_deg / 2):
                local_landmarks.append(lm)
                local_landmarks_keys.append(lm_key)
                local_predicted_measurements.append(np.array([r, b]))
                local_jacobians_pose.append(np.vstack((J_r_T, J_b_T)))
                local_jacobians_landmarks.append(np.vstack((J_r_lm, J_b_lm)))

        local_landmarks = np.asarray(local_landmarks, dtype=float).reshape(-1, 2)
        local_predicted_measurements = np.asarray(local_predicted_measurements, dtype=float).reshape(-1, 2)

        n = len(local_landmarks_keys)

        R = np.kron(np.eye(n), config.noise.range_bearing_cov_matrix)
        H = np.zeros((2 * n, 3 + 2 * n), order="F")
        for i in range(n):
            H[2 * i:2 * i + 2, 0:3] = local_jacobians_pose[i]
            H[2 * i:2 * i + 2, 3 + 2 * i:3 + 2 * i + 2] = local_jacobians_landmarks[i]

        t0 = time.perf_counter()
        cov_query = [key_kp1] + local_landmarks_keys
        support_size = slam.isam2.jointMarginalSupportCliqueCount(cov_query)
        P = slam.isam2.jointMarginalCovariance(cov_query).fullMatrix()
        P = reorder_covariance_naive(P)
        diagnostics.duration_covariance_extraction = time.perf_counter() - t0

        S = H @ P @ H.T + R

        t0 = time.perf_counter()
        association = JCBB_association(
            measurements,
            local_predicted_measurements,
            S,
            config.association.alpha_individual,
            config.association.alpha_joint,
        )
        diagnostics.duration_association = time.perf_counter() - t0

        is_associated = association >= 0
        associated_measurements = measurements[is_associated]
        associated_landmarks_keys = [local_landmarks_keys[i] for i in association[is_associated]]

        for (r, b), lm_key in zip(associated_measurements, associated_landmarks_keys):
            slam.new_factors.add(
                gtsam.BearingRangeFactor2D(
                    poseKey=key_kp1,
                    pointKey=lm_key,
                    measuredBearing=gtsam.Rot2(b),
                    measuredRange=r,
                    noiseModel=bearing_range_noise,
                )
            )

        unassociated_measurements = measurements[~is_associated]

        tentative_landmarks = []
        for r, b in unassociated_measurements:
            B_lm = gtsam.Rot2(b).rotate(gtsam.Point2(r, 0.0))
            W_lm = T_kp1.transformFrom(B_lm)
            tentative_landmarks.append(W_lm)

        confirmed_landmarks = manager.add_tentative_landmarks(
            current_step=k + 1,
            unassociated_measurements=unassociated_measurements,
            new_tentative_landmarks=np.asarray(tentative_landmarks, dtype=float).reshape(-1, 2),
        )

        for lm in confirmed_landmarks:
            new_lm_key = L(len(slam.landmark_keys))
            slam.landmark_keys.append(new_lm_key)

            slam.new_values.insert(new_lm_key, lm.position)

            for obs in lm.supporting_observations:
                r, b = obs.measurement
                slam.new_factors.add(
                    gtsam.BearingRangeFactor2D(
                        poseKey=X(obs.step),
                        pointKey=new_lm_key,
                        measuredBearing=gtsam.Rot2(b),
                        measuredRange=r,
                        noiseModel=bearing_range_noise,
                    )
                )

        t0 = time.perf_counter()
        slam.update_and_clear()
        diagnostics.add_time("duration_optimization", time.perf_counter() - t0)

        diagnostics.duration_step = time.perf_counter() - t_step
        diagnostics.scan_step = k + 1
        diagnostics.scan_time = meas.scan_time
        diagnostics.num_landmarks = len(slam.landmark_keys)
        diagnostics.num_local_landmarks = len(local_landmarks_keys)
        diagnostics.num_associated_measurement = int(np.sum(is_associated))
        diagnostics.num_unassociated_measurement = int(np.sum(~is_associated))
        diagnostics.num_support_cliques = support_size

        diagnostics_steps.append(diagnostics)

        if logger.should_save_association_diagnostics(k + 1):
            logger.save_association_diagnostics(
                AssociationDiagnostics(
                    scan_step=k + 1,
                    scan_time=meas.scan_time,
                    pose_index=k + 1,
                    pose=pose2_to_array(T_kp1),
                    measurements=measurements,
                    predicted_measurements=local_predicted_measurements,
                    association=association,
                    local_landmarks=local_landmarks,
                    innovation_covariance=S,
                )
            )

    total_time = time.perf_counter() - t_run_start

    if diagnostics_steps:
        final_step = diagnostics_steps[-1].scan_step
        final_diagnostics = diagnostics_steps[-1]
    else:
        final_step = 0
        final_diagnostics = StepDiagnostics(scan_step=0, num_landmarks=len(slam.landmark_keys))

    logger.save_snapshot(final_step, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(diagnostics_steps)
    logger.save_metadata(final_diagnostics, total_time, dataset=dataset.name)

    if save_plots or show_plots:
        from master_code.plotter import SlamRunPlotter

        plotter = SlamRunPlotter.from_run(output_dir)
        plotter.plot_all(save=save_plots, show=show_plots)
