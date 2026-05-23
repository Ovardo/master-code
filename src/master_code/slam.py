from __future__ import annotations

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from master_code.association import get_associator
from master_code.config import SlamConfig
from master_code.data_loader import LidarStepInput, WheelOdometry
from master_code.div.utils_gtsam import reorder_covariance_naive
from master_code.logger import SlamLogger, AssociationDiagnostics, StepDiagnostics
from master_code.preprocessing import detect_trees, relative_pose
from master_code.sensor import get_sensor_model
from master_code.tentative import TentativeLandmark, get_tentative_landmark_manager
from master_code.utils import pose2_to_array

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

        # State tracking
        self._poses_keys = list()  # list of pose keys in the factor graph
        self._landmark_keys = list()  # list of landmark keys in the factor graph
        
        self.step_diagnostics = StepDiagnostics()  # for logging and analysis

        # Initialize with prior factor for initial pose
        self._add_prior_factor(initial_pose)
    

    @property
    def num_poses(self) -> int:
        return len(self._poses_keys)
    
    @property
    def num_landmarks(self) -> int:
        return len(self._landmark_keys)

    def _add_prior_factor(self, prior_pose: np.ndarray) -> None:

        prior_pose = gtsam.Pose2(*prior_pose)
        self._new_values.insert(X(1), prior_pose)
        self._poses_keys.append(X(1))

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

    def update(self, data: LidarStepInput):
        # Clear diagnostic info from previous step
        self.step_diagnostics.clear()

        _t0 = time.perf_counter()
        
        # Main SLAM update steps
        T_pred = self._register_odometry(data.odometry)
        self._register_scan(data.scan, T_pred, data.scan_step, data.scan_time)

        # Diagnostics 
        self.step_diagnostics.duration_update = time.perf_counter() - _t0
        self.step_diagnostics.scan_time = data.scan_time
        self.step_diagnostics.scan_step = data.scan_step
        self.step_diagnostics.num_landmarks = self.num_landmarks
        # self.step_diagnostics.factor_graph_error = self.get_error() # consider commenting out this for faster runtime
        # self.step_diagnostics.num_factors = self.get_num_factors() # consider commenting out this for faster runtime
        
        if self.logger.should_save_snapshot(data.scan_step):
            self.logger.save_snapshot(
                step=data.scan_step, 
                snapshot=self.get_snapshot(), 
            )
        
        return self.step_diagnostics.copy()
    
    def _optimize(self) -> gtsam.ISAM2Result:
        _t0 = time.perf_counter()
    
        result = self.isam2.update(self._new_factors, self._new_values)
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()

        _t1 = time.perf_counter()
        self.step_diagnostics.add_time("duration_optimization", _t1 - _t0)

        return result

    def _register_odometry(self, odometry: list[WheelOdometry]) -> gtsam.Pose2:
        T_odom, T_odom_cov = self._preintegrate_odometry(odometry)
        T_kp1 = self._add_relative_pose_factor(T_odom, T_odom_cov)
        self._optimize()
        return T_kp1
    
    
    def _register_scan(
        self,
        scan: np.ndarray,
        T_k: gtsam.Pose2,
        scan_step: int,
        scan_time: float,
    ) -> None:
        t0 = time.perf_counter()
        z = self._detect_measurements(scan)
        self.step_diagnostics.duration_scan_processing = time.perf_counter() - t0

        t0 = time.perf_counter()
        local_lm, local_lm_keys = self._get_local_landmarks(T_k)
        self.step_diagnostics.duration_local_landmark_extraction = time.perf_counter() - t0

        association = self._associate_measurements(
            z=z,
            T_k=T_k,
            local_lm=local_lm,
            local_lm_keys=local_lm_keys,
            scan_step=scan_step,
            scan_time=scan_time,
        )

        self._handle_association(
            z=z,
            association=association,
            T_k=T_k,
            local_lm_keys=local_lm_keys,
        )

        self._optimize()

        self.step_diagnostics.num_local_landmarks = len(local_lm_keys)

    def _preintegrate_odometry(self, wheel_odometry: list[WheelOdometry]) -> tuple[gtsam.Pose2, np.ndarray]:
        T_odom     = gtsam.Pose2.Identity()
        T_odom_cov = np.zeros((3, 3))
        
        for odo in wheel_odometry:
            T_delta, J_delta_u = relative_pose(odo.velocity, odo.steering, odo.dt)
            T_delta_cov = odo.dt * self.Q_output

            H1 = np.zeros((3, 3), order='F')
            H2 = np.zeros((3, 3), order='F')
        
            T_odom     = T_odom.compose(T_delta, H1, H2)
            T_odom_cov = H1 @ T_odom_cov @ H1.T + H2 @ T_delta_cov @ H2.T
        
        return T_odom, T_odom_cov
    

    def _add_relative_pose_factor(self, T_odom: gtsam.Pose2, T_odom_cov: np.ndarray):
        key_k   = X(self.num_poses)
        key_kp1 = X(self.num_poses + 1)
        self._poses_keys.append(key_kp1)

        # Add initial guess for new pose
        T_k = self.isam2.calculateEstimatePose2(key_k)
        T_kp1 = T_k.compose(T_odom)
        self._new_values.insert(key_kp1, T_kp1)

        # Odometry factor: measurement from k to kp1 
        cov = gtsam.noiseModel.Gaussian.Covariance(T_odom_cov)

        self._new_factors.add(
            gtsam.BetweenFactorPose2(
                key1=key_k, 
                key2=key_kp1, 
                relativePose=T_odom,
                noiseModel=cov
            )
        )

        return T_kp1
    
    def _detect_measurements(self, scan: np.ndarray) -> np.ndarray:
        z = detect_trees(scan)
        z = z[z[:, 0] < self.cfg.sensor.max_range]
        return z

    def _associate_measurements(
        self,
        z: np.ndarray,
        T_k: gtsam.Pose2,
        local_lm: np.ndarray,
        local_lm_keys: list[int],
        scan_step: int,
        scan_time: float,
    ) -> np.ndarray:

        query = [self._poses_keys[-1]] + local_lm_keys
        query_cov = self._extract_joint_covariance(query)
        
        t0 = time.perf_counter()
        innovation_cov = self.sensor.innovation_covariance(T_k, local_lm, query_cov)
        self.step_diagnostics.duration_innovation_covariance = time.perf_counter() - t0

        z_pred = self.sensor.h(T_k, local_lm)

        t0 = time.perf_counter()
        association = self.associator.associate(
            z,
            z_pred,
            innovation_cov,
        )
        self.step_diagnostics.duration_association = time.perf_counter() - t0

        if self.logger.should_save_association_diagnostics(scan_step):
            self.logger.save_association_diagnostics(
                AssociationDiagnostics(
                    scan_step=scan_step,
                    scan_time=scan_time,
                    pose_index=self.num_poses,
                    pose=pose2_to_array(T_k),
                    measurements=z,
                    predicted_measurements=z_pred,
                    association=association,
                    local_landmarks=local_lm,
                    local_landmark_keys=local_lm_keys,
                    innovation_covariance=innovation_cov,
                )
            )

        return association
    
    def _handle_association(
        self,
        z: np.ndarray,
        association: np.ndarray,
        T_k: gtsam.Pose2,
        local_lm_keys: list[int],
    ) -> None:

        is_associated  = association >= 0
        z_associated   = z[is_associated]
        z_unassociated = z[~is_associated]
        lm_associated_keys = [local_lm_keys[i] for i in association[is_associated]]

        self._add_bearing_range_factors(z_associated, lm_associated_keys)

        lm_unassociated = self.sensor.h_inverse(T_k, z_unassociated)

        self.t0 = time.perf_counter()
        confirmed_tentatives = self.tentative.process_unassociated_measurements(
            current_step=self.num_poses,
            unassociated_measurements=z_unassociated,
            new_tentative_landmarks=lm_unassociated,
        )

        self._promote_tentative_landmarks(confirmed_tentatives)
        self.step_diagnostics.duration_tentative_processing = time.perf_counter() - self.t0

        self.step_diagnostics.num_associated_measurement = int(np.sum(is_associated))
        self.step_diagnostics.num_unassociated_measurement = int(np.sum(~is_associated))


    def _get_local_landmarks(self, T_k: gtsam.Pose2) -> tuple[np.ndarray, list[int]]:
        local_lm = []
        local_lm_keys = []

        for j in self._landmark_keys:
            lm_j = self.isam2.calculateEstimatePoint2(j) # world frame
            
            r = T_k.range(lm_j)
            b = T_k.bearing(lm_j).theta()  
            
            if self._is_inside_gate(r, b):
                local_lm.append(lm_j)
                local_lm_keys.append(j)

        return np.array(local_lm), local_lm_keys


    def _is_inside_gate(self, r: float, b: float) -> bool:
        inside_range = r < self.cfg.sensor.max_range
        inside_fov = np.abs(b) < np.deg2rad(self.cfg.sensor.fov_deg) / 2
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
        covariance = reorder_covariance_naive(covariance.fullMatrix())

        self.step_diagnostics.duration_covariance_extraction = time.perf_counter() - _t0
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
                    poseKey=X(self.num_poses), 
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
            
            new_lm_key = L(self.num_landmarks)
            self._landmark_keys.append(new_lm_key)

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
    
    def get_num_factors(self) -> int:
        return self.isam2.getFactorsUnsafe().size()

    def get_poses(self) -> np.ndarray:
        """Get all pose estimates"""
        return np.array([pose2_to_array(self.isam2.calculateEstimatePose2(X(k))) for k in range(2, self.num_poses+1)])

    def get_poses_covariance(self) -> np.ndarray:
        """Get marginal covariances for all pose estimates (#poses,3,3)"""
        return np.stack([self.isam2.marginalCovariance(X(k)) for k in range(2, self.num_poses+1)], axis=0)

    def get_landmarks(self) -> np.ndarray:
        """Get all landmark estimates"""
        return np.array([self.isam2.calculateEstimatePoint2(L(lm)) for lm in range(self.num_landmarks)])

    def get_landmarks_covariance(self) -> np.ndarray:
        """Get marginal covariances for all landmark estimates (#landmarks,2,2)"""
        covariances = [self.isam2.marginalCovariance(L(lm)) for lm in range(self.num_landmarks)]
        if covariances:
            return np.stack(covariances, axis=0)
        else:
            return np.array([])
    
    def get_snapshot(self) -> dict[str, np.ndarray]:
        """Get snapshot of current state for logging."""
        return dict(
            step=self.num_poses, # TODO: double check if one of error
            poses=self.get_poses(),
            poses_covariance=self.get_poses_covariance(),
            landmarks=self.get_landmarks(),
            landmarks_covariance=self.get_landmarks_covariance(),
        )
    
