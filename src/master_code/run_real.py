# Adapted from Odin Aleksander Severinsen Graded Assignment 2 code in TTK4250.
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
from tqdm import tqdm
from pathlib import Path

from master_code.data_association import JCBB_association
from master_code.config import SlamConfig
from master_code.loaders.victoria_park import VictoriaParkLoader
from master_code.logger import AssociationDiagnostics, SlamLogger, StepDiagnostics
from master_code.paths import RUNS_ROOT
from master_code.plotter import SlamRunPlotter
from master_code.preprocessing import detect_trees, preintegrate, relative_pose
from master_code.landmark_manager import get_tentative_landmark_manager
from master_code.utils import make_psd, pose2_to_array, reorder_covariance_naive

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
            'poses':                self.get_poses(),
            'poses_covariance':     self.get_poses_covariance(),
            'landmarks':            self.get_landmarks(),
            'landmarks_covariance': self.get_landmarks_covariance(),
        }


def run_real(
    config: SlamConfig,
    output_dir: Path,
    num_steps: int,
    show_plots: bool = False,
    save_plots: bool = True,
):
    
    # ======= Setup ========
    logger = SlamLogger(output_dir, config.logging)
    loader = VictoriaParkLoader()

    if num_steps is None:
        num_steps = loader.max_steps
    else:
        num_steps = min(loader.max_steps, num_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml") # Copy config to output_dir for reproducibility 


    # ======= Initialize SLAM system ========
    
    # iSAM2 specifics 
    slam = SlamState(
        isam2=gtsam.ISAM2(),
        new_factors=gtsam.NonlinearFactorGraph(),
        new_values=gtsam.Values(),
        pose_keys=[],
        landmark_keys=[],
    )

    # Responsible for initilizing landmarks if enough evidence
    manager = get_tentative_landmark_manager(config)
    
    # Noise models 
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([
        config.noise.sigma_init_pose_x, 
        config.noise.sigma_init_pose_y, 
        config.noise.sigma_init_pose_yaw_rad
    ])
    
    # Note that GTSAM expects bearing first!
    bearing_range_noise = gtsam.noiseModel.Diagonal.Sigmas([
        config.noise.sigma_bearing_rad, 
        config.noise.sigma_range
    ])
    
    # Logging
    diagnostics_steps = []

    # ======= Add prior factor ========
    prior_mean = gtsam.Pose2(*loader.initial_pose)
   
    slam.new_values.insert(X(0), prior_mean)
    slam.pose_keys.append(X(0))

    slam.new_factors.add(
        gtsam.PriorFactorPose2(
            key=X(0), 
            prior=prior_mean, 
            noiseModel=prior_noise
        )
    )

    # ======= Update ISAM2 with anchoring factor =======
    slam.update_and_clear()
    
    # ======= Main loop =======
    t_run_start = time.perf_counter()
    
    for k, meas in tqdm(enumerate(loader.iterate(num_steps)), total=num_steps, desc="SLAM"):
        
        diagnostics = StepDiagnostics()
        t_step = time.perf_counter()
        
        # ======= Raw measurements ========
        wheel_odometry = meas.odometry
        scan = meas.scan 
        
        # ======= Preprocessing and preintegratin of wheel odometry ========
        delta_odo = []
        delta_odo_covs = []
        for odo in wheel_odometry:
            delta = relative_pose(odo.velocity, odo.steering, odo.dt)
            delta_odo.append(delta)
            delta_cov = config.noise.odom_cov_matrix * odo.dt
            delta_odo_covs.append(delta_cov)

        delta_T, delta_cov = preintegrate(delta_odo, delta_odo_covs)
        
        # ======= Add odometry factor ========
        key_k   = X(k)
        key_kp1 = X(k + 1)
        slam.pose_keys.append(key_kp1)

        # Add initial guess for new pose
        T_k = slam.isam2.calculateEstimatePose2(key_k)
        T_kp1 = T_k.compose(delta_T)
        slam.new_values.insert(key_kp1, T_kp1)

        # Convert integrated noise to noise model 
        delta_cov_model = gtsam.noiseModel.Gaussian.Covariance(delta_cov)

        slam.new_factors.add(
            gtsam.BetweenFactorPose2(
                key1=key_k, 
                key2=key_kp1, 
                relativePose=delta_T,
                noiseModel=delta_cov_model
            )
        )
        
        # ======= Update iSAM2 with predicted pose =======
        t0 = time.perf_counter()
        slam.update_and_clear()
        diagnostics.add_time("duration_optimization", time.perf_counter() - t0)
        
        # ======= Process lidar scan ========
        measurements = detect_trees(scan)
        measurements = measurements[measurements[:, 0] < config.sensor.max_range] # Filter by max range

        # ======= Extract local predicted measurements and jacobians =========
        local_landmarks = []
        local_landmarks_keys = []
        local_predicted_measurements = []
        local_jacobians_pose = []
        local_jacobians_landmarks = []
        
        for lm_key in slam.landmark_keys:
            lm = slam.isam2.calculateEstimatePoint2(lm_key)
            
            J_r_T  = np.zeros((1, 3), order="F")
            J_r_lm = np.zeros((1, 2), order="F")
            J_b_T  = np.zeros((1, 3), order="F")
            J_b_lm = np.zeros((1, 2), order="F")
            
            r = T_kp1.range(lm,   J_r_T, J_r_lm)
            b = T_kp1.bearing(lm, J_b_T, J_b_lm).theta()

            if r < config.sensor.max_range and abs(b) < np.deg2rad(config.sensor.fov_deg/2):
                local_landmarks.append(lm)
                local_landmarks_keys.append(lm_key)
                local_predicted_measurements.append(np.array([r, b]))
                local_jacobians_pose.append(np.vstack((J_r_T, J_b_T)))
                local_jacobians_landmarks.append(np.vstack((J_r_lm, J_b_lm)))

        # Convert to arrays for downstream processing
        local_landmarks = np.array(local_landmarks)
        local_predicted_measurements = np.array(local_predicted_measurements)
        
        # ======= Calculate innovation covariance ========
        n = len(local_landmarks_keys)

        # Form joint measurement noise covariance and jacobian 
        R = np.kron(np.eye(n), config.noise.range_bearing_cov_matrix)
        H = np.zeros((2*n, 3 + 2*n), order="F")
        for i in range(n):
            H[2*i:2*i+2, 0:3] = local_jacobians_pose[i]
            H[2*i:2*i+2, 3+2*i:3+2*i+2] = local_jacobians_landmarks[i]

        # Recover joint marginal covariance from Bayes tree
        t0 = time.perf_counter()
        cov_query = [key_kp1] + local_landmarks_keys
        support_size = slam.isam2.jointMarginalSupportCliqueCount(cov_query)
        P = slam.isam2.jointMarginalCovariance(cov_query).fullMatrix()
        P = reorder_covariance_naive(P)
        diagnostics.duration_covariance_extraction = time.perf_counter() - t0
        

        # Innovation covariance for local predicted measurements
        S = H @ P @ H.T + R
        # S = make_psd(S)
        
        # ======= Perform data association ========
        t0 = time.perf_counter()
        association = JCBB_association(measurements, local_predicted_measurements, S, config.association.alpha_individual,  config.association.alpha_joint)
        diagnostics.duration_association = time.perf_counter() - t0

        # ======= Handle asociated measurements ======= 
        is_associated = association >= 0
        associated_measurements = measurements[is_associated]
        associated_landmarks_keys = [local_landmarks_keys[i] for i in association[is_associated]]
        
        # Add range bearing factors for associated measurements
        for (r, b), lm_key in zip(associated_measurements, associated_landmarks_keys):
            slam.new_factors.add(
                gtsam.BearingRangeFactor2D(
                    poseKey=key_kp1, 
                    pointKey=lm_key, 
                    measuredBearing=gtsam.Rot2(b), 
                    measuredRange=r, 
                    noiseModel=bearing_range_noise
                )
            )
        
        # ======= Handle unnasociated measurements ======= 
        unassociated_measurements = measurements[~is_associated]
        
        # Measurement projection to world frame
        tentative_landmarks = []
        for r, b in unassociated_measurements:
            B_lm = gtsam.Rot2(b).rotate(gtsam.Point2(r, 0.0)) # body 
            W_lm = T_kp1.transformFrom(B_lm) # world 
            tentative_landmarks.append(W_lm)

        # Add tenative landmarks to manager, get back confirmed landmarks ready for promotion .
        confirmed_landmarks = manager.add_tentative_landmarks(
            current_step=k+1,
            unassociated_measurements=unassociated_measurements,
            new_tentative_landmarks=np.array(tentative_landmarks),
        )

        # Add new landmark factors for confirmed landmarks
        for lm in confirmed_landmarks:
            
            new_lm_key = L(len(slam.landmark_keys))
            slam.landmark_keys.append(new_lm_key)

            slam.new_values.insert(new_lm_key, lm.position)

            # Add retroactively all supporting observations as factors
            for obs in lm.supporting_observations:
                r, b = obs.measurement
                k_seen = obs.step
                slam.new_factors.add(
                    gtsam.BearingRangeFactor2D(
                        poseKey=X(k_seen),
                        pointKey=new_lm_key,
                        measuredBearing=gtsam.Rot2(b),
                        measuredRange=r,
                        noiseModel=bearing_range_noise
                    )
                )
     
        
        # ======= Update ISAM2 with measurement factors and values ========
        t0 = time.perf_counter()
        slam.update_and_clear()
        diagnostics.add_time("duration_optimization", time.perf_counter() - t0)

        # ======= Logging ========
        diagnostics.duration_step = time.perf_counter() - t_step
        diagnostics.scan_step = k+1
        diagnostics.scan_time = meas.scan_time
        diagnostics.num_landmarks = len(slam.landmark_keys)
        diagnostics.num_local_landmarks = len(local_landmarks_keys)
        diagnostics.num_associated_measurement = np.sum(is_associated)
        diagnostics.num_unassociated_measurement = np.sum(~is_associated)
        diagnostics.num_support_cliques = support_size
        
        diagnostics_steps.append(diagnostics)

        if logger.should_save_association_diagnostics(k+1):
            logger.save_association_diagnostics(
                AssociationDiagnostics(
                    scan_step=k+1,
                    scan_time=meas.scan_time,
                    pose_index=k+1,
                    pose=pose2_to_array(T_kp1),
                    measurements=measurements,
                    predicted_measurements=local_predicted_measurements,
                    association=association,
                    local_landmarks=local_landmarks,
                    innovation_covariance=S
                )
            )
        
    total_time = time.perf_counter() - t_run_start
    
    # ======= Save result to output dir ========
    logger.save_snapshot(k, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(diagnostics_steps)
    logger.save_metadata(diagnostics_steps[-1], total_time, dataset="victoria_park")

    if save_plots or show_plots:
        plotter = SlamRunPlotter.from_run(output_dir)
        plotter.plot_all(save=save_plots, show=show_plots)

    
def main() -> None:
    NUM_STEPS   = 500
    OUTPUT_DIR  = RUNS_ROOT / "real" / datetime.now().strftime("%Y%m%d_%H%M%S") 
    CONFIG_NAME = "real.yaml"

    config = SlamConfig.load(CONFIG_NAME)
    
    run_real(
        config=config,
        output_dir=OUTPUT_DIR,
        num_steps=NUM_STEPS,
        show_plots=True,
        save_plots=True,
    )


if __name__ == "__main__":
    main()


