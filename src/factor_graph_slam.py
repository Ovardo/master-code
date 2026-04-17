from __future__ import annotations

import time
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from config import SLAMConfig
from association import get_associatior
from tentative import get_tentative_landmark_manager, TentativeLandmark
from models import get_sensor_model 
from data_loader import LidarStepInput

from utils.utils_gtsam import pose2_to_array, reorder_covariance_naive
from utils.utils_math import make_psd
from utils.utils_victoria_park import detectTrees, odom_from_u


class FactorGraphSLAM:
    """Main SLAM estimator using factor graph."""

    def __init__(self, cfg: SLAMConfig, pose0: np.ndarray):  
        self.cfg = cfg
        
        self.tentative = get_tentative_landmark_manager(cfg)
        self.associator = get_associatior(cfg)
        self.sensor = get_sensor_model(cfg)
        # self.profiler = get_profiler(cfg)
        
        
        # ISAM2 stuff 
        self.isam = gtsam.ISAM2() # could add tsam.ISAM2Params() to config...
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
      
        # Noise models
        self.Sigma_meas = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.gtsam_landmark_sigmas)
        self.Sigma_prior = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.gtsam_prior_pose_sigmas)
        
        # Noise covariance matrices
        self.Q_input = cfg.noise.odom_input_cov
        self.Q_output = cfg.noise.odom_output_cov
        self.R = cfg.noise.landmark_cov

        # State tracking
        self._num_poses = 0 
        self._num_landmarks = 0 

        # Initialize graph with prior on initial pose
        self._add_prior_factor(pose0)  # NOTE: could considering makig this public

        # Odometry integration
        self.T_delta = gtsam.Pose2()
        self.Sigma_delta = np.zeros((3,3))
        
        self.poses_dr = [gtsam.Pose2(*pose0)]  # for dead reckoning trajectory


        # Logging and viualization data
        self._counts_local_landmark: list = []
        self._counts_landmark: list = []
        self._times_update: list = []
        self._times_covariance_extraction: list = []
        self._times_association: list = []
        self._times_optimization: list = []
        self._times_optimization2: list = []

    
    def update(self, step: LidarStepInput):
        start = time.perf_counter()
        
        for z_odo in step.odometry:
            self._register_odometry(z_odo)
        self._register_scan(step.z_lsr)
        
        self._times_update.append(time.perf_counter() - start)

    def _register_odometry(self, z_odo):
        T_odo, J_odo_u = odom_from_u(z_odo.ve, z_odo.alpha, z_odo.dt)

        Q_input = J_odo_u @ self.Q_input @ J_odo_u.T * 0
        Q_odo = Q_input  + z_odo.dt * self.Q_output 
        # Q_odo = z_odo.dt * self.Q_output 

        H1 = np.zeros((3,3), order='F')
        H2 = np.zeros((3,3), order='F')
    
        self.T_delta = self.T_delta.compose(T_odo, H1, H2)
        self.Sigma_delta = H1 @ self.Sigma_delta @ H1.T + H2 @ Q_odo @ H2.T

    def _register_scan(self, z_lsr):
        
        # Incorprate preintegrated odometry and reset it
        T_pred = self._incorporate_odometry()

        # Ensure preintegrated values are up to date before extracting covariance 
        start = time.perf_counter()
        self._optimize() 
        self._times_optimization2.append(time.perf_counter() - start)

        z = detectTrees(z_lsr)

        # filter away measurements long rang measurements as often inprecise
        z = z[z[:, 0] < self.cfg.sensor.max_range] # TODO: this should perhanps be sepperate gate

        # Data assocation
        z_hat, z_hat_ids = self._get_predicted_measurements(T_pred)
        z_hat_gated, z_hat_gated_ids = self._gate_predicted_measurements(z_hat, z_hat_ids)
        
        start = time.perf_counter()
        joint_covariance = self._extract_covariance(z_hat_gated_ids)
        self._times_covariance_extraction.append(time.perf_counter() - start)
        
        innovation_covariance = self._compute_innovation_covariance(
            z_hat_gated_ids,
            T_pred,
            joint_covariance,
        )
    
        start = time.perf_counter()
        asssociation = self._compute_association(
            z,
            z_hat_gated,
            z_hat_gated_ids,
            innovation_covariance,
        )
        self._times_association.append(time.perf_counter() - start)
        
        self._handle_association(z, asssociation)
        
        start = time.perf_counter()
        self._optimize()
        self._times_optimization.append(time.perf_counter() - start)

        
    def _add_prior_factor(self, pose0: np.ndarray):
        """Add prior factor for initial pose."""
        pose0 = gtsam.Pose2(*pose0)
        pose0_factor = gtsam.PriorFactorPose2(X(0), pose0, self.Sigma_prior)
        self._new_factors.add(pose0_factor)
        self._new_values.insert(X(0), pose0)
        self._num_poses += 1 
        self._optimize()  # optimize immediately to initialize ISAM2 with the prior


    def _optimize(self) -> None:
        self.isam.update(self._new_factors, self._new_values)
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()

    
    def _incorporate_odometry(self) -> gtsam.Pose2:
        X_curr = X(self._num_poses - 1)
        X_next = X(self._num_poses)

        # Add odometry factor
        odom_factor = gtsam.BetweenFactorPose2(
            X_curr, X_next, self.T_delta, 
            gtsam.noiseModel.Gaussian.Covariance(self.Sigma_delta)
        )

        self._new_factors.add(odom_factor)

        # Predict next pose for initialization
        T_curr = self.isam.calculateEstimatePose2(X_curr)
        T_next = T_curr.compose(self.T_delta )
        self._new_values.insert(X_next, T_next)
        self._num_poses += 1 

        # Update dead reckoning trajectory
        self.poses_dr.append(self.poses_dr[-1].compose(self.T_delta))

        # Reset accumulated odom
        self.T_delta = gtsam.Pose2()
        self.Sigma_delta = np.zeros((3,3))

        return T_next
    

    def _get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted measurements for all landmarks based on priori pose estimate and landmark estimates."""
        
        M = self._num_landmarks

        zbar = np.zeros((M,2), dtype=float)  
        zbar_ids = np.zeros(M, dtype=int) 
        
        for j in range(self._num_landmarks):
            lm = self.isam.calculateEstimatePoint2(L(j))
            zbar[j,0] = pose_pred.range(lm)
            zbar[j,1] = pose_pred.bearing(lm).theta() 
            zbar_ids[j] = j 
        return zbar, zbar_ids
    
    def _gate_predicted_measurements(self, zbar, zbar_ids) -> tuple[np.ndarray, np.ndarray]:
        """Gate predicted measurements based on range and bearing thresholds."""
        zbar_gated = []
        zbar_gated_ids = []
        for z, id in zip(zbar, zbar_ids):
            r = z[0] # range
            b = z[1] # bearing
            if r < self.cfg.sensor.max_range and np.abs(b) < np.deg2rad(self.cfg.sensor.fov_deg/2): # TODO: should maybe use a map config and not sensor.max_range
                zbar_gated.append(z)
                zbar_gated_ids.append(id) 

        self._counts_local_landmark.append(len(zbar_gated)) # for logging

        return np.array(zbar_gated, dtype=float).reshape(-1, 2), \
               np.array(zbar_gated_ids, dtype=int)
    
    
    def _extract_covariance(self, zbar_ids: np.ndarray) -> np.ndarray:
        """
        Extract joint covariance for last pose and predicted measurements 
        coresponding to the ids in zbar_ids.
        """
        if len(zbar_ids) == 0:
            # No predicted measurements after gating, only at initalization
            return np.zeros([3,3]) 
        
        # NOTE: The order in which the keys are added can be important
        query = [X(self._num_poses-1)] + [L(id) for id in zbar_ids]   
        
        covariance = self.isam.jointMarginalCovariance(query)
        
        # Reorder covariance to match state ordering
        covariance = reorder_covariance_naive(covariance) # TODO: maybe make more secure

        return covariance


    def _compute_innovation_covariance(
        self,
        zbar_ids: np.ndarray,
        pose_pred: gtsam.Pose2,
        cov_body: np.ndarray,
    ):
        """Compute measurement/innovation covariance for predicted measurements."""
        m = np.array([self.isam.calculateEstimatePoint2(L(id)) for id in zbar_ids]) # (M', 2)
        x = pose2_to_array(pose_pred) # (3,)
        
        S = self.sensor.predicted_measurement_covariance(x, m, cov_body)
        S = make_psd(S) 

        return S

    def _compute_association(
        self,
        z: np.ndarray,
        zbar: np.ndarray,
        zbar_ids: np.ndarray,
        S: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute association between measurements and predicted measurements.

        Returns
        -------
        association_global
            Landmark IDs for associated measurements, or -1 for unassociated ones.
        """
        association_local = self.associator.associate(z, zbar, S)
        
        # Convert local indices to global landmark indices
        association_global = association_local.copy()  
        for i, a in enumerate(association_local):
            if a >= 0:
                association_global[i] = zbar_ids[a]

        return association_global
    

    def _handle_association(self, z: np.ndarray, association: np.ndarray):
        associated_mask = association >= 0
        unassociated_mask = association == -1

        associated_measurements = z[associated_mask]
        associated_landmark_ids = association[associated_mask]
        unassociated_measurements = z[unassociated_mask]

        self._add_associated_landmark_measurements(
            associated_measurements,
            associated_landmark_ids,
        )

        confirmed_tentatives = self._process_unassociated_measurements(
            unassociated_measurements,
        )

        self._promote_tentative_landmarks(confirmed_tentatives)
        

    def _add_associated_landmark_measurements(
        self, 
        measurements: np.ndarray, 
        associations: np.ndarray
    ):
        """Add factors only for measurements associated with confirmed landmarks."""
        pose_key = X(self._num_poses-1)

        for (r, b), a_j in zip(measurements, associations):
            meas_factor = gtsam.BearingRangeFactor2D(
                pose_key, L(a_j), gtsam.Rot2(b), r, self.Sigma_meas
            )
            self._new_factors.add(meas_factor)
            
    def _process_unassociated_measurements(
        self,
        measurements: np.ndarray,   # (M, 2), columns = [range, bearing]
    ) -> list:
        """
        Send unassociated measurements to tentative landmark manager.

        Returns a list of tentative landmarks that are now confirmed and ready
        to be promoted into the factor graph.
        """
        pose_key = X(self._num_poses - 1)
        current_pose = self.isam.calculateEstimatePose2(pose_key)

        world_measurements = []
        raw_measurements = []

        for r, b in measurements:
            lm_x_local = r * np.cos(b)
            lm_y_local = r * np.sin(b)
            lm_local = gtsam.Point2(lm_x_local, lm_y_local)
            lm_global = current_pose.transformFrom(lm_local)

            world_measurements.append(np.array([lm_global[0], lm_global[1]]))
            raw_measurements.append(np.array([r, b]))

        confirmed_tentatives = self.tentative.process_unassociated_measurements(
            current_step=self._num_poses - 1,
            world_measurements=world_measurements,
            raw_measurement=raw_measurements,
        )

        return confirmed_tentatives
    
    def _promote_tentative_landmarks(self, confirmed_tentatives: list[TentativeLandmark]) -> None:
        """
        Promote confirmed tentative landmarks into the factor graph.

        Simple version:
        - insert landmark variable
        - initialize its position
        - add factor to the pose of the most recent supporting observation
        """
        for tlm in confirmed_tentatives:
            lm_id = self._num_landmarks
            lm_key = L(lm_id)
            self._num_landmarks += 1

            lm_global = gtsam.Point2(float(tlm.position[0]), float(tlm.position[1]))
            self._new_values.insert(lm_key, lm_global)

            # Add only one factor from the most recent supporting observation
            # obs = tlm.supporting_observations[-1]
            # r, b = obs.measurement

            # meas_factor = gtsam.BearingRangeFactor2D(
            #     pose_key,
            #     lm_key,
            #     gtsam.Rot2(float(b)),
            #     float(r),
            #     self.Sigma_meas,
            # )
            # self._new_factors.add(meas_factor)

            # Add retroactively all supporting observations as factors
            for obs in tlm.supporting_observations:
                r, b = obs.measurement

                meas_factor = gtsam.BearingRangeFactor2D(
                    X(obs.step),
                    lm_key,
                    gtsam.Rot2(float(b)),
                    float(r),
                    self.Sigma_meas,
                )
                self._new_factors.add(meas_factor)

            # Optional:
            # store mapping if you want to remember which tentative became which landmark
            # self.confirmed_landmark_metadata[lm_id] = ...
    





    def get_current_pose(self) -> gtsam.Pose2:
        """Get current robot pose estimate"""
        return self.isam.calculateEstimatePose2(X(self._num_poses-1))
    
    def get_estimated_poses(self) -> np.ndarray:
        """Get all pose estimates"""
        return np.array([pose2_to_array(self.isam.calculateEstimatePose2(X(k))) for k in range(self._num_poses)])

    def get_estimated_pose_covariances(self) -> list[np.ndarray]:
        """Get covariances for all pose estimates"""
        return [self.isam.marginalCovariance(X(k)) for k in range(self._num_poses)]

    def get_estimated_landmarks(self) -> np.ndarray:
        """Get all landmark estimates"""
        return np.array([self.isam.calculateEstimatePoint2(L(lm)) for lm in range(self._num_landmarks)])

    def get_estimated_landmark_covariances(self) -> list[np.ndarray]:
        """Get covariances for all landmark estimates"""
        return [self.isam.marginalCovariance(L(lm)) for lm in range(self._num_landmarks)]

    def get_dead_reckoning_poses(self) -> np.ndarray:
        """Get dead reckoning trajectory"""
        return np.array([pose2_to_array(pose) for pose in self.poses_dr])
    
    def get_num_poses(self) -> int:
        return self._num_poses

    def get_num_landmarks(self) -> int:
        return self._num_landmarks
    

