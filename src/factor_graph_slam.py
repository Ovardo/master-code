from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from config import SLAMConfig
from association import get_associatior
from tentative import get_tentative_landmark_manager, TentativeLandmark
from models import get_sensor_model 

from timing_profiler import TimingProfiler

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
        
        # Graph and values
        self.graph = gtsam.NonlinearFactorGraph() # TODO: remove this(?)
        self.values = gtsam.Values()
        
        # ISAM2 stuff 
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.isam = gtsam.ISAM2() # could add tsam.ISAM2Params() to config...
      
        # Noise models
        self.Sigma_meas = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.gtsam_landmark_sigmas)
        self.Sigma_prior = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.gtsam_prior_pose_sigmas)
        
        # Noise covariance matrices
        self.Q_input = cfg.noise.odom_input_cov
        self.Q_output = cfg.noise.odom_output_cov
        self.R = cfg.noise.landmark_cov

        # State tracking
        self.num_poses = 0 
        self.num_landmarks = 0 

        # Initialize graph with prior on initial pose
        self._add_prior_factor(pose0)  # NOTE: could considering makig this public

        # Odometry integration
        self.T_delta = gtsam.Pose2()
        self.Sigma_delta = np.zeros((3,3))
        
        self.poses_dr = [gtsam.Pose2(*pose0)]  # for dead reckoning trajectory
    
    def register_odometry(self, z_odo):
        
        T_odo, J_odo_u = odom_from_u(z_odo[0], z_odo[1], z_odo[2])
        
        Q_odo = J_odo_u @ self.Q_input @ J_odo_u.T + self.Q_output 

        H1 = np.zeros((3,3), order='F')
        H2 = np.zeros((3,3), order='F')
    
        self.T_delta = self.T_delta.compose(T_odo, H1, H2)
        self.Sigma_delta = H1 @ self.Sigma_delta @ H1.T + H2 @ Q_odo @ H2.T
        

    def register_scan(self, z_lsr):
        
        # Incorprate preintegrated odometry and reset it
        T_pred = self._incorporate_odometry()

        z = detectTrees(z_lsr)

        # filter away measurements with range > gate as often inprecise
        z = z[z[:, 0] < self.cfg.sensor.max_range] # TODO: this should perhanps be sepperate gate

        # Data assocation
        z_hat, z_hat_ids = self._get_predicted_measurements(T_pred)
        z_hat_gated, z_hat_gated_ids = self._gate_predicted_measurements(z_hat, z_hat_ids)
        
        joint_covariance = self._extract_covariance(z_hat_gated_ids)
        
        innovation_covariance = self._compute_innovation_covariance(
            z_hat_gated_ids,
            T_pred,
            joint_covariance,
        )
    
        asssociation = self._compute_association(
            z,
            z_hat_gated,
            z_hat_gated_ids,
            innovation_covariance,
        )

        self._handle_association(z, asssociation)
        
        self._optimize()

        
    def _add_prior_factor(self, pose0: np.ndarray):
        """Add prior factor for initial pose."""
        pose0 = gtsam.Pose2(*pose0)
        pose0_factor = gtsam.PriorFactorPose2(X(0), pose0, self.Sigma_prior)
        self.graph.add(pose0_factor)
        self.values.insert(X(0), pose0)
        self.new_factors.add(pose0_factor)
        self.new_values.insert(X(0), pose0)
        self.num_poses += 1 
    
    def _incorporate_odometry(self) -> gtsam.Pose2:
        X_curr = X(self.num_poses - 1)
        X_next = X(self.num_poses)

        # Add odometry factor
        odom_factor = gtsam.BetweenFactorPose2(
            X_curr, X_next, self.T_delta, 
            gtsam.noiseModel.Gaussian.Covariance(self.Sigma_delta)
        )

        self.graph.add(odom_factor)
        self.new_factors.add(odom_factor)

        # Predict next pose for initialization
        T_curr = self.values.atPose2(X_curr)
        T_next = T_curr.compose(self.T_delta )
        self.values.insert(X_next, T_next)
        self.new_values.insert(X_next, T_next)
        self.num_poses += 1 

        # Update dead reckoning trajectory
        self.poses_dr.append(self.poses_dr[-1].compose(self.T_delta))

        # Reset accumulated odom
        self.T_delta = gtsam.Pose2()
        self.Sigma_delta = np.zeros((3,3))

        return T_next
    
    # def _profile(self, name: str):
    #     return self.profiler.section(name, iteration=self._active_iteration)

    def _get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted measurements for all landmarks based on priori pose estimate and landmark estimates."""
        
        M = self.num_landmarks

        zbar = np.zeros((M,2), dtype=float)  
        zbar_ids = np.zeros(M, dtype=int) 
        
        for j in range(self.num_landmarks):
            lm = self.values.atPoint2(L(j))
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
        query = [X(self.num_poses-1)] + [L(id) for id in zbar_ids]   
        
        # Ensure values are up to date before extracting covariance 
        self._optimize() #TODO: find more efficient way
        
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
        m = np.array([self.values.atPoint2(L(id)) for id in zbar_ids]) # (M', 2)
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
        pose_key = X(self.num_poses-1)

        for (r, b), a_j in zip(measurements, associations):
            meas_factor = gtsam.BearingRangeFactor2D(
                pose_key, L(a_j), gtsam.Rot2(b), r, self.Sigma_meas
            )
            self.graph.add(meas_factor)
            self.new_factors.add(meas_factor)
            
    def _process_unassociated_measurements(
        self,
        measurements: np.ndarray,   # (M, 2), columns = [range, bearing]
    ) -> list:
        """
        Send unassociated measurements to tentative landmark manager.

        Returns a list of tentative landmarks that are now confirmed and ready
        to be promoted into the factor graph.
        """
        pose_key = X(self.num_poses - 1)
        current_pose = self.values.atPose2(pose_key)

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
            current_step=self.num_poses - 1,
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
            lm_id = self.num_landmarks
            lm_key = L(lm_id)
            self.num_landmarks += 1

            lm_global = gtsam.Point2(float(tlm.position[0]), float(tlm.position[1]))
            self.values.insert(lm_key, lm_global)
            self.new_values.insert(lm_key, lm_global)

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
            # self.graph.add(meas_factor)
            # self.new_factors.add(meas_factor)

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
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)

            # Optional:
            # store mapping if you want to remember which tentative became which landmark
            # self.confirmed_landmark_metadata[lm_id] = ...
    
    def _optimize(self) -> gtsam.Values:
        self.isam.update(self.new_factors, self.new_values)
        self.values = self.isam.calculateEstimate()
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

    def get_marginals(self) -> gtsam.Marginals:
        """Compute marginals for current estimate"""
        return gtsam.Marginals(self.graph, self.values)

    def get_current_pose(self) -> gtsam.Pose2:
        """Get current robot pose estimate"""
        return self.values.atPose2(X(self.num_poses-1))
    
    def get_estimated_poses(self) -> np.ndarray:
        """Get all pose estimates up to current step"""
        return np.array([pose2_to_array(self.values.atPose2(X(k))) for k in range(self.num_poses)])

    def get_estimated_pose_covariances(self) -> list[np.ndarray]:
        """Get covariances for all pose estimates"""
        marginals = self.get_marginals()
        return [marginals.marginalCovariance(X(k)) for k in range(self.num_poses)]

    def get_estimated_landmarks(self) -> np.ndarray:
        """Get all landmark estimates up to current step"""
        return np.array([self.values.atPoint2(L(lm)) for lm in range(self.num_landmarks)])

    def get_estimated_landmark_covariances(self) -> list[np.ndarray]:
        """Get covariances for all landmark estimates"""
        marginals = self.get_marginals()
        return [marginals.marginalCovariance(L(lm)) for lm in range(self.num_landmarks)]

    def get_poses_dr(self) -> np.ndarray:
        """Get dead reckoning trajectory"""
        return np.array([pose2_to_array(pose) for pose in self.poses_dr])
