from __future__ import annotations

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from association import get_associatior
from config import SlamConfig
from data_loader import LidarStepInput
from logger import SlamLogger
from models import get_sensor_model
from tentative import TentativeLandmark, get_tentative_landmark_manager
from utils.utils_gtsam import pose2_to_array, reorder_covariance_naive
from utils.utils_math import make_psd
from utils.utils_victoria_park import detectTrees, odom_from_u


class FactorGraphSLAM:
    """Main SLAM estimator using factor graph."""

    def __init__(
            self, 
            cfg: SlamConfig, 
            logger: SlamLogger | None = None
        ):  

        self.cfg = cfg
        self.logger = logger
        
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
        
        # Noise covariance matrices
        self.Q_input = cfg.noise.odom_input_cov
        self.Q_output = cfg.noise.odom_output_cov
        self.R = cfg.noise.landmark_cov

        # State tracking
        self._n_poses = 0 
        self._n_landmarks = 0 

        # Odometry integration
        self.T_delta = gtsam.Pose2()
        self.Sigma_delta = np.zeros((3,3))
        

    def add_prior_factor(self, prior_pose: np.ndarray):
        """Add prior factor for initial pose."""
        prior_pose = gtsam.Pose2(*prior_pose)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self.cfg.noise.gtsam_prior_pose_sigmas)
        prior_pose_factor = gtsam.PriorFactorPose2(X(0), prior_pose, prior_noise)
        self._new_factors.add(prior_pose_factor)
        self._new_values.insert(X(0), prior_pose)
        self._n_poses += 1 
        self._optimize()  # optimize immediately to initialize ISAM2 with the prior
    
    def update(self, step: LidarStepInput):
        t0 = time.perf_counter()
        
        for z_odo in step.odometry:
            self._register_odometry(z_odo)
        self._register_scan(step.z_lsr)
        
        t_total = time.perf_counter() - t0
        curr_step = self._n_poses - 1

        if self.logger is not None:
            self.logger.log_step(
                step=curr_step,
                times={
                    "total":                 t_total,
                    "covariance_extraction": self._t_cov,
                    "association":           self._t_assoc,
                    "optimization":          self._t_opt,
                },
                counts={
                    "local_landmarks": self._n_local_landmarks,
                    "total_landmarks": self._n_landmarks,
                },
            )
 
            if (self.logger.snapshot_every > 0
                    and curr_step % self.logger.snapshot_every == 0):
                self.logger.log_snapshot(curr_step, self.get_snapshot())
        


    def _register_odometry(self, z_odo):
        T_odo, J_odo_u = odom_from_u(z_odo.ve, z_odo.alpha, z_odo.dt)

        Q_input = J_odo_u @ self.Q_input @ J_odo_u.T * 0 # TODO
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
        t0 = time.perf_counter()
        self._optimize() 
        self._t_opt = time.perf_counter() - t0 

        z = detectTrees(z_lsr)

        # filter away measurements long rang measurements as often inprecise
        z = z[z[:, 0] < self.cfg.sensor.max_range] 

        # Data assocation
        z_hat, z_hat_ids = self._get_predicted_measurements(T_pred)
        z_hat_gated, z_hat_gated_ids = self._gate_predicted_measurements(z_hat, z_hat_ids)
        
        t0 = time.perf_counter()
        joint_covariance = self._extract_covariance(z_hat_gated_ids)
        self._t_cov = time.perf_counter() - t0 
        
        innovation_covariance = self._compute_innovation_covariance(
            z_hat_gated_ids, T_pred, joint_covariance,
        )
    
        t0 = time.perf_counter()
        asssociation = self._compute_association(
            z, z_hat_gated, z_hat_gated_ids, innovation_covariance,
        )
        self._t_assoc = time.perf_counter() - t0
        
        self._handle_association(z, asssociation)
        
        t0 = time.perf_counter()
        self._optimize()
        self._t_opt += time.perf_counter() - t0

        
    def _optimize(self) -> None:
        self.isam.update(self._new_factors, self._new_values)
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()

    
    def _incorporate_odometry(self) -> gtsam.Pose2:
        X_curr = X(self._n_poses - 1)
        X_next = X(self._n_poses)

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
        self._n_poses += 1 

        # Reset accumulated odom
        self.T_delta = gtsam.Pose2()
        self.Sigma_delta = np.zeros((3,3))

        return T_next
    

    def _get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted measurements for all landmarks based on priori pose estimate and landmark estimates."""
        
        M        = self._n_landmarks
        zbar     = np.zeros((M,2), dtype=float)  
        zbar_ids = np.zeros(M, dtype=int) 
        
        for j in range(self._n_landmarks):
            lm = self.isam.calculateEstimatePoint2(L(j))
            zbar[j,0] = pose_pred.range(lm)
            zbar[j,1] = pose_pred.bearing(lm).theta() 
            zbar_ids[j] = j 

        return zbar, zbar_ids
    
    def _gate_predicted_measurements(self, zbar, zbar_ids) -> tuple[np.ndarray, np.ndarray]:
        """Gate predicted measurements based on range and bearing thresholds."""
        fov_rad      = np.deg2rad(self.cfg.sensor.fov_deg)
        max_range    = self.cfg.sensor.max_range
        mask         = (zbar[:, 0] < max_range) & (np.abs(zbar[:, 1]) < fov_rad/2)
 
        gated        = zbar[mask]
        gated_ids    = zbar_ids[mask]
 
        # Store for logger — overwritten each scan
        self._n_local_landmarks = len(gated)
 
        return gated.reshape(-1, 2), gated_ids
    
    
    def _extract_covariance(self, zbar_ids: np.ndarray) -> np.ndarray:
        """
        Extract joint covariance for last pose and predicted measurements 
        coresponding to the ids in zbar_ids.
        """
        if len(zbar_ids) == 0:
            # No predicted measurements after gating, only at initalization
            return np.zeros([3,3]) 
        
        # NOTE: The order in which the keys are added can be important
        query = [X(self._n_poses-1)] + [L(id) for id in zbar_ids]   
        
        # marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), self.isam.calculateEstimate())
        # covariance = marginals.jointMarginalCovariance(query).fullMatrix()
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
    ) -> np.ndarray:
        """Compute association between measurements and predicted measurements.

        Returns
        -------
        association_global
            Landmark IDs for associated measurements, or -1 for unassociated ones.
        """
        association_local  = self.associator.associate(z, zbar, S)
        association_global = association_local.copy()  
        for i, a in enumerate(association_local):
            if a >= 0:
                association_global[i] = zbar_ids[a]
        return association_global
    

    def _handle_association(self, z: np.ndarray, association: np.ndarray):
        associated_mask   = association >= 0
        unassociated_mask = association == -1

        self._add_associated_landmark_measurements(
            z[associated_mask], association[associated_mask]
        )
        confirmed_tentatives = self._process_unassociated_measurements(
            z[unassociated_mask],
        )
        self._promote_tentative_landmarks(confirmed_tentatives)
        

    def _add_associated_landmark_measurements(
        self, 
        measurements: np.ndarray, 
        associations: np.ndarray
    ):
        """Add factors only for measurements associated with confirmed landmarks."""
        pose_key = X(self._n_poses-1)

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
        pose_key = X(self._n_poses - 1)
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
            current_step=self._n_poses - 1,
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
            lm_id = self._n_landmarks
            lm_key = L(lm_id)
            self._n_landmarks += 1

            lm_global = gtsam.Point2(float(tlm.position[0]), float(tlm.position[1]))
            self._new_values.insert(lm_key, lm_global)


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
            # store mapping if wanting to remember which tentative became which landmark
            # self.confirmed_landmark_metadata[lm_id] = ...
    
    
    def get_poses(self) -> np.ndarray:
        """Get all pose estimates"""
        return np.array([pose2_to_array(self.isam.calculateEstimatePose2(X(k))) for k in range(self._n_poses)])

    def get_poses_covariance(self) -> np.ndarray:
        """Get marginal covariances for all pose estimates (#poses,3,3)"""
        return np.stack([self.isam.marginalCovariance(X(k)) for k in range(self._n_poses)], axis=0)

    def get_landmarks(self) -> np.ndarray:
        """Get all landmark estimates"""
        return np.array([self.isam.calculateEstimatePoint2(L(lm)) for lm in range(self._n_landmarks)])

    def get_landmarks_covariance(self) -> np.ndarray:
        """Get marginal covariances for all landmark estimates (#landmarks,2,2)"""
        return np.stack([self.isam.marginalCovariance(L(lm)) for lm in range(self._n_landmarks)], axis=0)
    
    def get_snapshot(self) -> dict[str, np.ndarray]:
        """Get snapshot of current state for logging."""
        return dict(
            poses=self.get_poses(),
            poses_covariance=self.get_poses_covariance(),
            landmarks=self.get_landmarks(),
            landmarks_covariance=self.get_landmarks_covariance(),
        )
    
    def get_n_poses(self) -> int:
        return self._n_poses

    def get_n_landmarks(self) -> int:
        return self._n_landmarks
    


