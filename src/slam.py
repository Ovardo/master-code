from __future__ import annotations

import time

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from association import get_associator
from config import SlamConfig
from data_loader import LidarStepInput, WheelOdometry
from logger import SlamLogger
from sensor import get_sensor_model
from tentative import TentativeLandmark, get_tentative_landmark_manager
from div.utils_gtsam import reorder_covariance_naive
from utils import make_psd, pose2_to_array
from preprocessing import detect_trees, relative_pose

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
        
        self.step_metrics = dict()  # for logging and analysis

        # Initialize with prior factor for initial pose
        self._add_prior_factor(initial_pose)

    def _add_prior_factor(self, prior_pose: np.ndarray) -> None:

        prior_pose = gtsam.Pose2(*prior_pose)
        self._new_values.insert(X(1), prior_pose)
        self._n_poses += 1 
        
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
        _t0 = time.perf_counter()

        self.step_metrics.clear() # clear diagnostic info from previous step
        self.step_metrics["scan_time"] = data.scan_time
        self.step_metrics["scan_step"] = data.scan_step
        
        self._register_odometry(data.odometry)
        self._register_scan(data.scan)

        self.step_metrics["n_landmarks"] = self._n_landmarks
        self.step_metrics["t_update"] = time.perf_counter() - _t0
        
        return self.step_metrics.copy()


    def _preintegrate_odometry(self, odometry: list[WheelOdometry]) -> tuple[gtsam.Pose2, np.ndarray]:
        
        Delta = gtsam.Pose2.Identity()
        Sigma = np.zeros((3,3))
        
        for odo in odometry:
            T_odo, J_odo_u = relative_pose(odo.velocity, odo.steering, odo.dt)
            # J_odo_u @ self.Q_input @ J_odo_u.T
            Q_odo =  odo.dt * self.Q_output # TODO: consider J_odo_u

            H1 = np.zeros((3,3), order='F')
            H2 = np.zeros((3,3), order='F')
        
            Delta = Delta.compose(T_odo, H1, H2)
            Sigma = H1 @ Sigma @ H1.T + H2 @ Q_odo @ H2.T
        
        return Delta, Sigma
    
    def _add_odometry_factor(self, T_Bk_Bkp1: gtsam.Pose2, Sigma_Bk_Bkp1: np.ndarray):
        
        Bk = X(self._n_poses)
        Bkp1 = X(self._n_poses + 1)
        self._n_poses += 1 

        # Add initial guess for new pose
        T_W_Bk = self.isam2.calculateEstimatePose2(Bk)
        T_W_Bkp1 = T_W_Bk.compose(T_Bk_Bkp1)
        self._new_values.insert(Bkp1, T_W_Bkp1)

        # Odometry factor: measurement from Bk to Bkp1 
        noise_model = gtsam.noiseModel.Gaussian.Covariance(Sigma_Bk_Bkp1)

        self._new_factors.add(
            gtsam.BetweenFactorPose2(
                key1=Bk, 
                key2=Bkp1, 
                relativePose=T_Bk_Bkp1,
                noiseModel=noise_model
            )
        )
    
    def _register_odometry(self, odometry: list[WheelOdometry]):
        T_Bk_Bkp1, Sigma_Bk_Bkp1 = self._preintegrate_odometry(odometry)
        self._add_odometry_factor(T_Bk_Bkp1, Sigma_Bk_Bkp1)
        self._optimize()


    def _register_scan(self, scan):
        
        # Convert from raw scan to tree range-bearing measurements
        z = detect_trees(scan)

        # filter away measurements long rang measurements as often inprecise
        z = z[z[:, 0] < self.cfg.sensor.max_range] 

        # Data association
        z_hat, z_hat_ids = self._get_predicted_measurements()
        
        cov_joint = self._extract_joint_covariance(z_hat_ids)

        cov_innovation = self._compute_innovation_covariance(z_hat_ids, cov_joint)   
    
        asssociation = self._compute_association(z, z_hat, z_hat_ids, cov_innovation)
        
        self._handle_association(z, asssociation)

        result = self._optimize() # TODO: can use data in result for analysis
  

    def _optimize(self) -> gtsam.ISAM2Result:
        _t0 = time.perf_counter()
    
        result = self.isam2.update(self._new_factors, self._new_values)
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()

        _t1 = time.perf_counter()
        self.step_metrics["t_optimize"] = self.step_metrics.get("t_optimize", 0.0) + (_t1 - _t0)

        return result

    def _get_predicted_measurements(self) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted measurements for all landmarks based on priori pose estimate and landmark estimates."""
        
        z_hat     = list() 
        z_hat_ids = list()

        T = self.isam2.calculateEstimatePose2(X(self._n_poses))
        
        for j in range(self._n_landmarks):
            lm = self.isam2.calculateEstimatePoint2(L(j))
            
            r = T.range(lm)
            b = T.bearing(lm).theta()  
            
            if self._is_inside_gate(r, b):
                z_hat.append([r, b])
                z_hat_ids.append(j)
        
        self.step_metrics["n_local_landmarks"] = len(z_hat)

        return np.array(z_hat), np.array(z_hat_ids)


    def _is_inside_gate(self, range, bearing):
        inside_range = range < self.cfg.sensor.max_range
        inside_fov = np.abs(bearing) < np.deg2rad(self.cfg.sensor.fov_deg)/2
        return inside_range and inside_fov
    

    def _extract_joint_covariance(self, z_hat_ids: np.ndarray) -> np.ndarray:
        """
        Extract joint covariance for last pose and predicted measurements 
        coresponding to the ids in z_hat_ids.
        """
        _t0 = time.perf_counter()

        # Check for no predicted measurements after gating, happens only at initalization
        if len(z_hat_ids) == 0:
            return np.zeros([3,3]) 
        
        # The order in which the keys are added is the order in which the covariance is returned
        query = [X(self._n_poses)] + [L(id) for id in z_hat_ids]   
        
        covariance = self.isam2.jointMarginalCovariance(query) # TODO: for new version, assume you can use .at(qurey) to get correct ordering
        
        # jointMarginalCov = self.isam2.jointMarginalCovariance(query)
        # covariance = jointMarginalCov.at(query) # covariance = jointMarginalCov.fullMatrix() does not give correct ordering, i think. Have to check source


        # Reorder covariance to match state ordering
        covariance = reorder_covariance_naive(covariance) # TODO: maybe make more secure, see comment above

        self.step_metrics["t_covariance_extraction"] = time.perf_counter() - _t0
        return covariance


    def _compute_innovation_covariance(
        self,
        zbar_ids: np.ndarray,
        Sigma_joint: np.ndarray,
    ):
        """Compute measurement/innovation covariance for predicted measurements."""
        m = np.array([self.isam2.calculateEstimatePoint2(L(id)) for id in zbar_ids]) # (M', 2)
        p = pose2_to_array(self.isam2.calculateEstimatePose2(X(self._n_poses))) # (3,)
        
        S = self.sensor.predicted_measurement_covariance(p, m, Sigma_joint)
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
        t0 = time.perf_counter()

        association_local  = self.associator.associate(z, zbar, S)
        association_global = association_local.copy()  
        for i, a in enumerate(association_local):
            if a >= 0:
                association_global[i] = zbar_ids[a]

        self.step_metrics["t_association"] = time.perf_counter() - t0
        return association_global
    

    def _handle_association(self, measurements: np.ndarray, association: np.ndarray):
        associated_mask   = association >= 0
        unassociated_mask = association == -1

        self._add_associated_landmark_measurements(
            measurements[associated_mask], association[associated_mask]
        )
        confirmed_tentatives = self._process_unassociated_measurements(
            measurements[unassociated_mask],
        )
        self._promote_tentative_landmarks(confirmed_tentatives)

        
        self.step_metrics["n_associated"] = np.sum(associated_mask)
        self.step_metrics["n_unassociated"] = np.sum(unassociated_mask)
        

    def _add_associated_landmark_measurements(
        self, 
        measurements: np.ndarray, 
        associations: np.ndarray
    ):
        """Add factors for associated measurements."""
        for (r, b), a_j in zip(measurements, associations):
            self._new_factors.add(
                gtsam.BearingRangeFactor2D(
                    poseKey=X(self._n_poses), 
                    pointKey=L(a_j), 
                    measuredBearing=gtsam.Rot2(b), 
                    measuredRange=r, 
                    noiseModel=self.bearing_range_noise
                )
            )
            
    def _process_unassociated_measurements(
        self,
        measurements: np.ndarray,   # (M, 2), columns = [range, bearing]
    ) -> list:
        """
        Send unassociated measurements to tentative landmark manager.

        Returns a list of tentative landmarks that are now confirmed and ready
        to be promoted into the factor graph.
        """
        pose_key = X(self._n_poses)
        T_W_B = self.isam2.calculateEstimatePose2(pose_key)

        world_positions = np.empty_like(measurements) # (M, 2)

        for i, (r, b) in enumerate(measurements):
            B_lm_x = r * np.cos(b)
            B_lm_y = r * np.sin(b)
            B_lm = gtsam.Point2(B_lm_x, B_lm_y)
            W_lm = T_W_B.transformFrom(B_lm)

            world_positions[i] = W_lm

        confirmed_tentatives = self.tentative.process_unassociated_measurements(
            current_step=self._n_poses,
            measurements=measurements,
            world_positions=world_positions,
        )

        return confirmed_tentatives
    
    def _promote_tentative_landmarks(self, confirmed_tentatives: list[TentativeLandmark]) -> None:
        """
        Promote confirmed tentative landmarks into the factor graph.
        """
        for tentative_lm in confirmed_tentatives:
            
            new_lm_key = L(self._n_landmarks)
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
    
