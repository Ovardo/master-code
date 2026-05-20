from __future__ import annotations

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from association import get_associator
from config import SlamConfig
from data_loader import LidarStepInput, WheelOdometry
from div.utils_gtsam import reorder_covariance_naive
from logger import SlamLogger
from preprocessing import detect_trees, relative_pose
from sensor import get_sensor_model
from tentative import TentativeLandmark, get_tentative_landmark_manager
from utils import make_psd, pose2_to_array

# TODO: visualize display(graphviz.Source(isam.dot()))

class FactorGraphSLAM:
    """Main SLAM estimator using factor graph."""

    def __init__(
            self, 
            config: SlamConfig, 
            logger: SlamLogger,
            initial_pose: np.ndarray = np.zeros(3),
        ):  

        self.cfg = config
        self.logger = logger
        
        self.tentative = get_tentative_landmark_manager(config)
        self.associator = get_associator(config)
        self.sensor = get_sensor_model(config)
        # self.profiler = get_profiler(config)
        
        # ISAM2 stuff 
        params = gtsam.ISAM2Params()
        # params.evaluateNonlinearError=True
        self.isam2 = gtsam.ISAM2(params) 
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
      
        # Noise models
        self.bearing_range_noise = gtsam.noiseModel.Diagonal.Sigmas(
            [config.noise.sigma_bearing_rad, 
             config.noise.sigma_range]
        )
    
        self.Q_input  = config.noise.control_input_cov_matrix
        self.Q_output = config.noise.odom_cov_matrix
        self.R        = config.noise.range_bearing_cov_matrix

        # State tracking
        self._n_poses = 0 
        self._n_landmarks = 0 
        
        self._poses_keys = list()  # list of pose keys in the factor graph
        self._map_keys = list()  # list of landmark keys in the factor graph
        
        self.step_metrics = dict()  # for logging and analysis

        # Initialize with prior factor for initial pose
        self._add_prior_factor(initial_pose)

    def update(self, data: LidarStepInput):
        _t0 = time.perf_counter()

        self.step_metrics.clear() # clear diagnostic info from previous step
        
        T_k_kp1, cov_k_kp1 = self._preintegrate_odometry(data.odometry)
        T_kp1 = self._add_relative_pose_factor(T_k_kp1, cov_k_kp1)
        self._optimize()
        
        # Convert from raw scan to tree range-bearing measurements
        z = detect_trees(data.scan)
        z = z[z[:, 0] < self.cfg.sensor.max_range] # filter away measurements long rang measurements as often inprecise

        local_map, local_map_keys = self._get_local_map(T_kp1)
    
        query          = [self._poses_keys[-1]] + local_map_keys
        query_cov      = self._extract_joint_covariance(query)
        innovation_cov = self.sensor.innovation_covariance(T_kp1, local_map, query_cov)   
        local_z_hat    = self.sensor.h(T_kp1, local_map)
       
        association = self.associator.associate(z, local_z_hat, innovation_cov)
        is_associated   = association >= 0
        z_associated    = z[is_associated]
        z_unassociated  = z[~is_associated]
        associated_keys = [local_map_keys[i] for i in association[is_associated]]

        self._add_bearing_range_factors(z_associated, associated_keys)
        
        landmarks_unassociated = self.sensor.h_inverse(T_kp1, z_unassociated) 
        
        confirmed_tentatives = self.tentative.process_unassociated_measurements(
            current_step=self._n_poses,
            unassociated_measurements=z_unassociated,
            new_tentative_landmarks=landmarks_unassociated,
        )

        self._promote_tentative_landmarks(confirmed_tentatives)

        result = self._optimize() # TODO: can use data in result for analysis


        self.step_metrics["scan_time"] = data.scan_time
        self.step_metrics["scan_step"] = data.scan_step
        self.step_metrics["n_landmarks"] = self._n_landmarks
        self.step_metrics["n_local_landmarks"] = len(local_map_keys)
        self.step_metrics["t_update"] = time.perf_counter() - _t0
        
        return self.step_metrics.copy()
    

    def _add_prior_factor(self, prior_pose: np.ndarray) -> None:

        prior_pose = gtsam.Pose2(*prior_pose)
        self._new_values.insert(X(1), prior_pose)
        self._poses_keys.append(X(1))
        self._n_poses = 1

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            [self.cfg.noise.sigma_init_pose_x, 
             self.cfg.noise.sigma_init_pose_y, 
             self.cfg.noise.sigma_init_pose_yaw_rad]
        )

        self._new_factors.add(
            gtsam.PriorFactorPose2(
                key=X(1), 
                prior=prior_pose, 
                noiseModel=prior_noise
            )
        )

        self._optimize()  
    
    def _preintegrate_odometry(self, wheel_odometry: list[WheelOdometry]) -> tuple[gtsam.Pose2, np.ndarray]:
        
        Delta = gtsam.Pose2.Identity()
        Sigma = np.zeros((3,3))
        
        for odo in wheel_odometry:
            Delta_inc, J_delta_u = relative_pose(odo.velocity, odo.steering, odo.dt)
            # J_odo_u @ self.Q_input @ J_odo_u.T
            Sigma_inc = odo.dt * self.Q_output # TODO: consider J_odo_u

            H1 = np.zeros((3,3), order='F')
            H2 = np.zeros((3,3), order='F')
        
            Delta = Delta.compose(Delta_inc, H1, H2)
            Sigma = H1 @ Sigma @ H1.T + H2 @ Sigma_inc @ H2.T
        
        return Delta, Sigma
    
    def _add_relative_pose_factor(self, T_k_kp1: gtsam.Pose2, cov_k_kp1: np.ndarray):
        k   = X(len(self._poses_keys))
        kp1 = X(len(self._poses_keys) + 1)
        self._poses_keys.append(kp1)
        self._n_poses += 1

        # Add initial guess for new pose
        T_k = self.isam2.calculateEstimatePose2(k)
        T_kp1 = T_k.compose(T_k_kp1)
        self._new_values.insert(kp1, T_kp1)

        # Odometry factor: measurement from k to kp1 
        cov = gtsam.noiseModel.Gaussian.Covariance(cov_k_kp1)

        self._new_factors.add(
            gtsam.BetweenFactorPose2(
                key1=k, 
                key2=kp1, 
                relativePose=T_k_kp1,
                noiseModel=cov
            )
        )

        return T_kp1

    def _optimize(self) -> gtsam.ISAM2Result:
        _t0 = time.perf_counter()
    
        result = self.isam2.update(self._new_factors, self._new_values)
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()

        _t1 = time.perf_counter()
        self.step_metrics["t_optimize"] = self.step_metrics.get("t_optimize", 0.0) + (_t1 - _t0)

        return result

    def _get_local_map(self, T_k: gtsam.Pose2) -> tuple[np.ndarray, list[int]]:
        local_map = []
        local_map_keys = []

        for j in self._map_keys:
            l_j = self.isam2.calculateEstimatePoint2(j)
            
            r = T_k.range(l_j)
            b = T_k.bearing(l_j).theta()  
            
            if self._is_inside_gate(r, b):
                local_map.append(l_j)
                local_map_keys.append(j)

        return np.array(local_map), local_map_keys

    def _is_inside_gate(self, range, bearing):
        inside_range = range < self.cfg.sensor.max_range
        inside_fov = np.abs(bearing) < np.deg2rad(self.cfg.sensor.fov_deg)/2
        return inside_range and inside_fov
    

    def _extract_joint_covariance(self, query: list[int]) -> np.ndarray:
        """
        Extract joint covariance for last pose and predicted measurements 
        coresponding to the ids in z_hat_ids.
        """
        _t0 = time.perf_counter()

        # Check for no predicted measurements after gating, happens only at initalization
        if len(query) <= 1:
            return np.zeros([3,3]) 
    
        covariance = self.isam2.jointMarginalCovariance(query) 
        covariance = reorder_covariance_naive(covariance) 

        self.step_metrics["t_covariance_extraction"] = time.perf_counter() - _t0
        return covariance
    

    def _add_bearing_range_factors(
        self, 
        measurements: np.ndarray, 
        associated_landmark_keys: list[int]
    ):
        """Add factors for associated measurements."""
        for (r, b), j in zip(measurements, associated_landmark_keys):
            self._new_factors.add(
                gtsam.BearingRangeFactor2D(
                    poseKey=X(self._n_poses), 
                    pointKey=j, 
                    measuredBearing=gtsam.Rot2(b), 
                    measuredRange=r, 
                    noiseModel=self.bearing_range_noise
                )
            )

    def _promote_tentative_landmarks(self, confirmed_tentatives: list[TentativeLandmark]) -> None:
        """
        Promote confirmed tentative landmarks into the factor graph.
        """
        for tentative_lm in confirmed_tentatives:
            
            new_lm_key = L(self._n_landmarks)
            self._map_keys.append(new_lm_key)
            self._n_landmarks += 1

            self._new_values.insert(new_lm_key, tentative_lm.position)

            # Add retroactively all supporting observations as factors
            for obs in tentative_lm.supporting_observations:
                r, b = obs.measurement
                self._new_factors.add(
                    gtsam.BearingRangeFactor2D(
                        poseKey=X(obs.step),
                        pointKey=new_lm_key,
                        measuredBearing=gtsam.Rot2(b),
                        measuredRange=r,
                        noiseModel=self.bearing_range_noise
                    )
                )
    
    # Getters for logging and visualization
    def get_error(self) -> float:
        factors = self.isam2.getFactorsUnsafe()
        estimate = self.isam2.calculateEstimate()
        return factors.error(estimate)
    
    def get_n_factors(self) -> int:
        return self.isam2.getFactorsUnsafe().size()

    def get_n_poses(self) -> int:
        return self._n_poses

    def get_n_landmarks(self) -> int:
        return self._n_landmarks

    def get_poses(self) -> np.ndarray:
        """Get all pose estimates"""
        return np.array([pose2_to_array(self.isam2.calculateEstimatePose2(X(k))) for k in range(2, self._n_poses+1)])

    def get_poses_covariance(self) -> np.ndarray:
        """Get marginal covariances for all pose estimates (#poses,3,3)"""
        return np.stack([self.isam2.marginalCovariance(X(k)) for k in range(2, self._n_poses+1)], axis=0)

    def get_landmarks(self) -> np.ndarray:
        """Get all landmark estimates"""
        return np.array([self.isam2.calculateEstimatePoint2(L(lm)) for lm in range(self._n_landmarks)])

    def get_landmarks_covariance(self) -> np.ndarray:
        """Get marginal covariances for all landmark estimates (#landmarks,2,2)"""
        return np.stack([self.isam2.marginalCovariance(L(lm)) for lm in range(self._n_landmarks)], axis=0)
    
    def get_snapshot(self) -> dict[str, np.ndarray]:
        """Get snapshot of current state for logging."""
        return dict(
            poses=self.get_poses(),
            poses_covariance=self.get_poses_covariance(),
            landmarks=self.get_landmarks(),
            landmarks_covariance=self.get_landmarks_covariance(),
        )
    
