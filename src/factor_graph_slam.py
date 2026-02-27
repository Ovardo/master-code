from __future__ import annotations

from abc import ABC, abstractmethod

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from association import Associator
from config import InferenceConfig
from models.dynamicmodels import OdometrySE2
from models.measurementmodels_offset import RangeBearing
from result import StepRecord
from utils.utils_gtsam import (
    pose2_to_array,
    reorder_covariance_auto,
    reorder_covariance_naive,
)
from utils.utils_math import make_psd, rotmat2
from utils.utils_victoria_park import Car, odometry_func


class Estimator(ABC):
    @abstractmethod
    def process_step(self, odometry, measurements):
        """Process one SLAM step given odometry and measurements. Returns current estimate."""
        pass


class FactorGraphSLAM:
    """Main SLAM estimator using factor graph."""

    def __init__(
        self,
        cfg: InferenceConfig,
        initial_pose: np.ndarray,
    ):  
        self.cfg = cfg
        self.associator = Associator(cfg)

        # Graph and values
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        
        # ISAM2 stuff 
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)

        # Models
        self.motion_model = OdometrySE2( # TODO: remove
            sigma_x=cfg.noise.x_std,
            sigma_y=cfg.noise.y_std,
            sigma_theta=cfg.noise.theta_std_rad,
        )
        self.sensor_model = RangeBearing(
            sigma_range=cfg.noise.range_std, 
            sigma_bearing=cfg.noise.bearing_std_rad,
            sensor_offset=np.array(cfg.sensor_offset),
        )

        # Noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.odometry_std)
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.measurement_std)

        # State tracking
        self.num_poses = 0 
        self.num_landmarks = 0 

        # Initialize graph with prior on initial pose
        initial_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.prior_std)
        self._add_prior_factor(gtsam.Pose2(*initial_pose), initial_pose_noise)

        # Odometry integration
        self.Delta = gtsam.Pose2()
        self.Sigma = np.zeros((3,3))
        self.poses_dr = [gtsam.Pose2(*initial_pose)]  # for dead reckoning trajectory


    def _add_prior_factor(self, prior_pose: gtsam.Pose2, prior_pose_noise: np.ndarray):
        """Add prior factor for initial pose."""
 
        prior_factor = gtsam.PriorFactorPose2(X(0), prior_pose, prior_pose_noise)
        
        self.graph.add(prior_factor)
        self.values.insert(X(0), prior_pose)

        self.new_factors.add(prior_factor)
        self.new_values.insert(X(0), prior_pose)
        self.num_poses += 1 


    def get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> tuple[np.ndarray, np.ndarray]:
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
    
    def gate_predicted_measurements(self, zbar, zbar_ids) -> tuple[np.ndarray, np.ndarray]:
        """Gate predicted measurements based on range and bearing thresholds."""
        zbar_gated = []
        zbar_gated_ids = []
        for z, id in zip(zbar, zbar_ids):
            r = z[0] # range
            b = z[1] # bearing
            if r < self.cfg.range_gate and np.abs(b) < np.deg2rad(self.cfg.fov_gate_deg)/2:
                zbar_gated.append(z)
                zbar_gated_ids.append(id)  

        return np.array(zbar_gated, dtype=float).reshape(-1, 2), np.array(zbar_gated_ids, dtype=int)


    def extract_covariance(self, zbar_ids: np.ndarray) -> np.ndarray:
        """Extract joint covariance for last pose and predicted measurements coresponding to the ids in zbar_ids."""
        
        pose_pred_key = X(self.num_poses-1)
        
        keys = [pose_pred_key]  # NOTE: order in which the keys are added is important
        keys += [L(id) for id in zbar_ids]   

        # This ensures we are computing covariances based on the latest linearization
        # linear_graph = self.isam.getFactorsUnsafe()
        # linear_values = self.isam.calculateBestEstimate()

        # # 2. Compute the marginals
        # marginals = gtsam.Marginals(linear_graph, linear_values)
          
        marginals = gtsam.Marginals(self.graph, self.values)

        # Joint covariance for previous pose and local landmarks (in local frame)
        covariance  = marginals.jointMarginalCovariance(keys).fullMatrix()

        # Reorder covariance to match state ordering
        covariance = reorder_covariance_naive(covariance) # TODO: maybe make more secure
        # covariance  = reorder_covariance_auto(
        #     covariance ,
        #     source_keys=sorted(keys),
        #     target_keys=keys,
        #     values=self.values,
        # )

        return covariance


    def compute_innovation_covariance(
        self,
        zbar_ids: np.ndarray,
        pose_pred: gtsam.Pose2,
        cov_body: np.ndarray,
    ):
        """Compute innovation covariance for predicted measurements."""
        
        n = len(zbar_ids) # num predicted measurements

        H = np.zeros((2 * n, 3 + 2 * n))
        R = np.zeros((2 * n, 2 * n))
        
        R_i = np.diag((self.cfg.noise.range_std**2, self.cfg.noise.bearing_std_rad**2))
 
        for i, id in enumerate(zbar_ids):
            m_i = self.values.atPoint2(L(id))
            H_x = self.sensor_model.H_x(pose2_to_array(pose_pred), m_i)
            H_mi = self.sensor_model.H_m(pose2_to_array(pose_pred), m_i)
            H[2 * i : 2 * i + 2, 0:3] = H_x
            H[2 * i : 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2] = H_mi
            R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = R_i

        S = H @ cov_body @ H.T + R
        S = make_psd(S) 

        return S

    def compute_association(self, z: np.ndarray, zbar: np.ndarray, zbar_ids, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """Compute association between measurements and predicted measurements using self.associator."""
        
        # Do association
        association_indices = self.associator.associate(z, zbar, S)
    
        # As Assoicator.associate returns indices into zbar for each measurement, we need to convert to landmark IDs
        association_ids = association_indices.copy()  
        association_ids[association_indices >= 0] = [zbar_ids[idx] for idx in association_indices if idx >= 0]

        return association_ids, association_indices
    

    
    # def process_step(self, z_odometry: tuple[float, float, float], z_range_bearing: list[tuple[float, float]]) -> StepRecord:
    def process_step(self, odometry: gtsam.Pose2, measurements: np.ndarray) -> StepRecord:
        """Main SLAM step processing:

        Parameters
        -------
        z_odometry : TODO np.ndarray (3,)
            (u, v, psi) representing odometry measurement (relative motion)
        z_range_bearing: np.ndarray (M,2)
            (range, bearing) measurements to landmarks

        Returns
        -------
        StepRecord 
            current estimates, measurements, associations etc. for this step

        """
        
        
        # odo = gtsam.Pose2(*odometry)  

        H1 = np.zeros((3,3), order='F')
        H2 = np.zeros((3,3), order='F')
    
        self.Delta = self.Delta.compose(odometry, H1, H2)

        # Ru = np.diag(np.array([0.02, 2*np.pi/180])**2)
        # Qu = odom_jacobien @ Ru @ odom_jacobien.T + Qf
        # Qu += np.diag([1e-6, 1e-6, 1e-8])  # tune (units: m^2, m^2, rad^2)

        Qu = np.diag(self.cfg.noise.odometry_std**2)

        self.Sigma = H1 @ self.Sigma @ H1.T + H2 @ Qu @ H2.T

        # pose_pred = self._add_odometry(odo)

        
        if len(measurements):
            from_idx = self.num_poses - 1
            to_idx = self.num_poses

            odom_factor = gtsam.BetweenFactorPose2(
                X(from_idx), X(to_idx), self.Delta, gtsam.noiseModel.Gaussian.Covariance(self.Sigma)
            )
            self.graph.add(odom_factor)
            self.new_factors.add(odom_factor)

            # Update dead reckoning trajectory
            self.poses_dr.append(self.poses_dr[-1].compose(self.Delta))

            # Predict next pose for initialization
            pose_prev = self.values.atPose2(X(from_idx))
            pose_pred = pose_prev.compose(self.Delta)
            self.values.insert(X(to_idx), pose_pred)
            self.new_values.insert(X(to_idx), pose_pred)
            self.num_poses += 1 # pose added

            # Reset accumulated odom
            self.Delta = gtsam.Pose2()
            self.Sigma = np.zeros((3,3))

            # self.isam.update(self.new_factors, self.new_values)
            # self.new_factors = gtsam.NonlinearFactorGraph()
            # self.new_values = gtsam.Values()

            # Data assocation
            zbar, zbar_ids = self.get_predicted_measurements(pose_pred)
            zbar_gated, zbar_gated_ids = self.gate_predicted_measurements(zbar, zbar_ids)
            
            cov = self.extract_covariance(zbar_gated_ids)
            cov_innovation = self.compute_innovation_covariance(zbar_gated_ids, pose_pred, cov)
            asssoc_ids, assoc_idx = self.compute_association(measurements, zbar_gated, zbar_gated_ids, cov_innovation)
                
            self._add_landmark_measurements(measurements, asssoc_ids)
            
            self.optimize_graph()

            # ---- store history ----
            record = StepRecord(
                step=self.num_poses-1,
                poses=self.get_estimated_poses(), 
                #poses_cov=self.get_estimated_pose_covariances(),
                poses_dr=self.get_poses_dr(),
                landmarks=self.get_estimated_landmarks(), 
                #landmarks_cov=self.get_estimated_landmark_covariances(),
                measurements=measurements,
                predicted_measurements=zbar_gated,
                predicted_measurements_ids=zbar_gated_ids,
                associations_ids=asssoc_ids,
                associations_idx=assoc_idx,
                cov_innovation=cov_innovation,
            )

            return record
        return None


    def optimize_graph(self) -> gtsam.Values:
        """Run optimization on the current factor graph"""
        if self.cfg.algorithm == "isam2":
            self.isam.update(self.new_factors, self.new_values)
            self.new_factors = gtsam.NonlinearFactorGraph()
            self.new_values = gtsam.Values()
            self.values = self.isam.calculateEstimate()
        elif self.cfg.algorithm == "batch":  # full batch optimization
            optParams = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, optParams)
            self.values = optimizer.optimize()
        else:
            raise ValueError(f"Unknown algorithm: {self.cfg.algorithm}")

    def _add_odometry(self, odometry: gtsam.Pose2) -> gtsam.Pose2:
        """Add odometry factor between consecutive poses"""
        from_idx = self.num_poses - 1
        to_idx = self.num_poses

        odom_factor = gtsam.BetweenFactorPose2(
            X(from_idx), X(to_idx), odometry, self.odometry_noise
        )
        self.graph.add(odom_factor)
        self.new_factors.add(odom_factor)

        # Predict next pose for initialization
        pose_prev = self.values.atPose2(X(from_idx))
        pose_pred = pose_prev.compose(odometry)
        self.values.insert(X(to_idx), pose_pred)
        self.new_values.insert(X(to_idx), pose_pred)
        
        self.num_poses += 1 # pose added
        
        return pose_pred

    def _add_landmark_measurements(
        self, 
        measurements: np.ndarray, # (M,2)
        associations: np.ndarray, # (M,) 
    ):
        """Add landmark measurement factors"""
        pose_key = X(self.num_poses-1)

        # j is measurement index, a_j is associated landmark index
        for (r, b), a_j in zip(measurements, associations):
            if a_j >= 0:  # measurement j associated with previously observed landmark a_j
                meas_factor = gtsam.BearingRangeFactor2D(pose_key, L(a_j), gtsam.Rot2(b), r, self.measurement_noise)
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)
            elif a_j == -1:  # new landmark, initialize and add factor
                lm_key = L(self.num_landmarks)
                self.num_landmarks += 1
                meas_factor = gtsam.BearingRangeFactor2D(pose_key, lm_key, gtsam.Rot2(b), r, self.measurement_noise)
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)
                self._initialize_landmark(lm_key, pose_key, r, b)
            elif a_j == -2:  # ambiguous association, could be outlier or valid match. For now, treat as outlier (no factor)
                continue
            else: 
                raise ValueError(f"Invalid association index: {a_j}")
        
    def _initialize_landmark(self, lm_key: int, pose_key: int, range: float, bearing: float) -> None:
        """Initialize a newly observed landmark."""
        current_pose = self.values.atPose2(pose_key)

        # Convert from polar to Cartesian in robot frame
        lm_x_local = range * np.cos(bearing)
        lm_y_local = range * np.sin(bearing)
        
        # Transform from local to global frame
        lm_local = gtsam.Point2(lm_x_local, lm_y_local)
        lm_global = current_pose.transformFrom(lm_local)
        
        # Insert into values
        self.values.insert(lm_key, lm_global)
        self.new_values.insert(lm_key, lm_global)


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