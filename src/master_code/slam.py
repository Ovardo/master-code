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
    """Data class representing a unified input for one SLAM step, regardless of the data source."""
    relative_pose: gtsam.Pose2 | None
    relative_pose_cov: np.ndarray | None
    measurements: np.ndarray
    scan_time: float


class SlamDataset(Protocol):
    """Protocol representing a SLAM dataset that can be iterated step-by-step with unified SLAM inputs."""
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


def _save_final_joint_covariance(
    output_dir: Path,
    covariance: np.ndarray,
    keys: list[int],
) -> None:
    """TEMP: persist the recovered joint marginal covariance of the final step.

    Saved to compare against the old covariance method. The block ordering of
    `covariance` is [pose(3), lm_0(2), lm_1(2), ...] matching `keys`. The keys are
    stored as raw GTSAM symbols (int64). Remove once the comparison is done.
    """
    np.savez(
        output_dir / "final_joint_covariance.npz",
        covariance=covariance,
        keys=np.asarray(keys, dtype=np.int64),
    )


def run_slam(
    config: SlamConfig,
    dataset: SlamDataset,
    output_dir: Path,
    num_steps: int | None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    # ======= Setup ========
    logger = SlamLogger(output_dir, config.logging)

    if num_steps is None:
        num_steps = dataset.max_steps
    else:
        num_steps = min(dataset.max_steps, num_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy config to output_dir for reproducibility.
    config.save(output_dir / "config.yaml")

    # ======= Initialize SLAM system ========
    # iSAM2 stores the factor graph estimate; new_factors/new_values are the next update batch.
    slam = SlamState(
        isam2=gtsam.ISAM2(),
        new_factors=gtsam.NonlinearFactorGraph(),
        new_values=gtsam.Values(),
        pose_keys=[],
        landmark_keys=[],
    )

    # Responsible for initializing landmarks once enough evidence has accumulated.
    manager = get_tentative_landmark_manager(config)

    # Noise models.
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

    # Logging.
    diagnostics_steps = []

    # TEMP: holds the final step's recovered joint covariance and its key ordering.
    final_joint_covariance: np.ndarray | None = None
    final_joint_covariance_keys: list[int] | None = None

    # ======= Add Prior Factor ========
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

    # Update iSAM2 With Anchoring Factor 
    slam.update_and_clear()

    # ======= Main Loop ========
    t_run_start = time.perf_counter()

    for k, meas in tqdm(enumerate(dataset.iterate_slam(config, num_steps)), total=num_steps, desc="SLAM"):
        diagnostics = StepDiagnostics()
        t_step = time.perf_counter()

        # ======= Current Pose ========
        pose_key = X(k)

        # ======= Predict and Add Odometry Factor ========
        if meas.relative_pose is not None:
            delta_T = meas.relative_pose
            delta_cov = meas.relative_pose_cov

            key_prev = X(k - 1)
            slam.pose_keys.append(pose_key)

            # Add an initial guess for the new pose.
            T_prev = slam.isam2.calculateEstimatePose2(key_prev)
            T_pose = T_prev.compose(delta_T)
            slam.new_values.insert(pose_key, T_pose)

            # Convert integrated noise to a GTSAM noise model.
            delta_cov_model = gtsam.noiseModel.Gaussian.Covariance(delta_cov)
            slam.new_factors.add(
                gtsam.BetweenFactorPose2(
                    key1=key_prev,
                    key2=pose_key,
                    relativePose=delta_T,
                    noiseModel=delta_cov_model,
                )
            )

            # ======= Update iSAM2 With Predicted Pose ========
            t0 = time.perf_counter()
            slam.update_and_clear()
            diagnostics.add_time("duration_optimization", time.perf_counter() - t0)
        else:
            # No motion into X(0); process measurements at the prior estimate.
            T_pose = slam.isam2.calculateEstimatePose2(pose_key)

        # ======= Process Measurements ========
        measurements = meas.measurements

        # ======= Extract Local Predicted Measurements and Jacobians ========
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

            r = T_pose.range(lm, J_r_T, J_r_lm)
            b = T_pose.bearing(lm, J_b_T, J_b_lm).theta()

            if r < config.sensor.range_local and abs(b) < config.sensor.bearing_local_rad:
                local_landmarks.append(lm)
                local_landmarks_keys.append(lm_key)
                local_predicted_measurements.append(np.array([r, b]))
                local_jacobians_pose.append(np.vstack((J_r_T, J_b_T)))
                local_jacobians_landmarks.append(np.vstack((J_r_lm, J_b_lm)))

        # Convert to arrays for downstream processing.
        local_landmarks = np.asarray(local_landmarks, dtype=float).reshape(-1, 2)
        local_predicted_measurements = np.asarray(local_predicted_measurements, dtype=float).reshape(-1, 2)

        # ======= Calculate Innovation Covariance ========
        n = len(local_landmarks_keys)

        # Form joint measurement noise covariance and Jacobian.
        R = np.kron(np.eye(n), config.noise.range_bearing_cov_matrix)
        H = np.zeros((2 * n, 3 + 2 * n), order="F")
        for i in range(n):
            H[2 * i:2 * i + 2, 0:3] = local_jacobians_pose[i]
            H[2 * i:2 * i + 2, 3 + 2 * i:3 + 2 * i + 2] = local_jacobians_landmarks[i]

        # Recover joint marginal covariance from the Bayes tree.
        t0 = time.perf_counter()
        cov_query = [pose_key] + local_landmarks_keys
        P = slam.isam2.jointMarginalCovariance(cov_query).fullMatrix()
        # marginals = gtsam.Marginals(slam.isam2.getFactorsUnsafe(), slam.isam2.calculateEstimate())
        # P = marginals.jointMarginalCovariance(cov_query).fullMatrix()
        P = reorder_covariance_naive(P)

        diagnostics.duration_covariance_extraction = time.perf_counter() - t0
        support_size = slam.isam2.jointMarginalSupportCliqueCount(cov_query) # TODO: comment out


        # Innovation covariance for local predicted measurements.
        S = H @ P @ H.T + R

        # ======= Perform Data Association ========
        t0 = time.perf_counter()
        association = JCBB_association(
            measurements,
            local_predicted_measurements,
            S,
            config.association.alpha_individual,
            config.association.alpha_joint,
        )
        diagnostics.duration_association = time.perf_counter() - t0

        # ======= Handle Associated Measurements ========
        is_associated = association >= 0
        associated_measurements = measurements[is_associated]
        associated_landmarks_keys = [local_landmarks_keys[i] for i in association[is_associated]]

        # Add range-bearing factors for associated measurements.
        for (r, b), lm_key in zip(associated_measurements, associated_landmarks_keys):
            slam.new_factors.add(
                gtsam.BearingRangeFactor2D(
                    poseKey=pose_key,
                    pointKey=lm_key,
                    measuredBearing=gtsam.Rot2(b),
                    measuredRange=r,
                    noiseModel=bearing_range_noise,
                )
            )

        # ======= Handle Unassociated Measurements ========
        unassociated_measurements = measurements[~is_associated]

        # Project unassociated measurements from the body frame to the world frame.
        tentative_landmarks = []
        for r, b in unassociated_measurements:
            B_lm = gtsam.Rot2(b).rotate(gtsam.Point2(r, 0.0))
            W_lm = T_pose.transformFrom(B_lm)
            tentative_landmarks.append(W_lm)

        # Add tentative landmarks to the manager and get confirmed landmarks ready for promotion.
        confirmed_landmarks = manager.add_tentative_landmarks(
            current_step=k,
            unassociated_measurements=unassociated_measurements,
            new_tentative_landmarks=np.asarray(tentative_landmarks, dtype=float).reshape(-1, 2),
        )

        # Add new landmark factors for confirmed landmarks.
        for lm in confirmed_landmarks:
            new_lm_key = L(len(slam.landmark_keys))
            slam.landmark_keys.append(new_lm_key)

            slam.new_values.insert(new_lm_key, lm.position)

            # Add all supporting observations retroactively as factors.
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

        # ======= Update iSAM2 With Measurement Factors and Values ========
        t0 = time.perf_counter()
        slam.update_and_clear()
        diagnostics.add_time("duration_optimization", time.perf_counter() - t0)

        # ======= Logging ========
        diagnostics.duration_step = time.perf_counter() - t_step
        diagnostics.scan_step = k
        diagnostics.scan_time = meas.scan_time
        diagnostics.num_landmarks = len(slam.landmark_keys)
        diagnostics.num_local_landmarks = len(local_landmarks_keys)
        diagnostics.num_associated_measurement = int(np.sum(is_associated))
        diagnostics.num_unassociated_measurement = int(np.sum(~is_associated))
        diagnostics.num_support_cliques = support_size # TODO: comment out

        diagnostics_steps.append(diagnostics)

        if logger.should_save_association_diagnostics(k):
            logger.save_association_diagnostics(
                AssociationDiagnostics(
                    scan_step=k,
                    scan_time=meas.scan_time,
                    pose_index=k,
                    pose=pose2_to_array(T_pose),
                    measurements=measurements,
                    predicted_measurements=local_predicted_measurements,
                    association=association,
                    local_landmarks=local_landmarks,
                    local_landmark_keys=local_landmarks_keys,
                    prior_joint_covariance=P,
                    innovation_covariance=S,
                )
            )

        if logger.should_save_snapshot(k):
            logger.save_snapshot(k, slam.get_snapshot())

    # ======= Save Result to Output Directory ========
    total_time = time.perf_counter() - t_run_start

    
    if diagnostics_steps:
        final_step = diagnostics_steps[-1].scan_step
        final_diagnostics = diagnostics_steps[-1]
    else:
        final_step = 0
        final_diagnostics = StepDiagnostics(scan_step=0, num_landmarks=len(slam.landmark_keys))

    # ======= Analysis ========
    # from master_code.analysis import compute_algebraic_connectivity
    # algebraic_connectivity = compute_algebraic_connectivity(slam.isam2, slam.isam2.calculateEstimate())

    logger.save_snapshot(final_step, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(diagnostics_steps)

    logger.save_metadata(
        final_diagnostics,
        total_time,
        dataset=dataset.name,
    )

    if save_plots or show_plots:
        from master_code.plotter import SlamRunPlotter

        plotter = SlamRunPlotter.from_run(output_dir)
        plotter.plot_all(save=save_plots, show=show_plots)
