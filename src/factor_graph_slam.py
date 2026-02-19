from __future__ import annotations

from abc import ABC, abstractmethod

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from association import Associator
from config import InferenceConfig
from models.dynamicmodels import OdometrySE2
from models.measurementmodels import RangeBearing
from result import StepRecord
from utils.utils_gtsam import pose2_to_array, reorder_covariance_auto
from utils.utils_math import make_psd, rotmat2
from utils.utils_types import PredictedMeasurement


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
        self.motion_model = OdometrySE2(
            sigma_x=cfg.noise.x_std,
            sigma_y=cfg.noise.y_std,
            sigma_theta=cfg.noise.theta_std_rad,
        )
        self.sensor_model = RangeBearing(
            sigma_range=cfg.noise.range_std, sigma_bearing=cfg.noise.bearing_std_rad,
        )

        # Noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.odometry_vec)
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.measurement_vec)
        self.measurement_noise_birth = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.measurement_vec * 1.0)  # higher uncertainty for new landmarks

        # State tracking
        self.num_steps = 0 
        self.num_landmarks = 0 

        # Initialize graph with prior on initial pose
        initial_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.prior_vec)
        self._add_prior_factor(gtsam.Pose2(*initial_pose), initial_pose_noise)


    def _add_prior_factor(self, prior_pose: gtsam.Pose2, prior_pose_noise: np.ndarray):
        """Add prior factor for initial pose."""
 
        prior_factor = gtsam.PriorFactorPose2(X(0), prior_pose, prior_pose_noise)
        
        self.graph.add(prior_factor)
        self.values.insert(X(0), prior_pose)

        self.new_factors.add(prior_factor)
        self.new_values.insert(X(0), prior_pose)

        self.optimize_graph()
        # self.num_steps += 1  # count prior as first step


    def get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted measurements for all landmarks based on predicted pose estimate and current landmark estimates."""
        
        M = self.num_landmarks

        zbar = np.zeros((M,2), dtype=float)  
        zbar_ids = np.zeros(M, dtype=int) 
        
        for j in range(self.num_landmarks):
            lm = self.values.atPoint2(L(j))
            zbar[j,0] = pose_pred.range(lm)
            zbar[j,1] = pose_pred.bearing(lm).theta() 
            zbar_ids[j] = j 
        return zbar, zbar_ids
    
    def gate_predicted_measurements(self, predicted_measurements, predicted_measurements_ids) -> tuple[np.ndarray, np.ndarray]:
        """Gate predicted measurements based on range and bearing thresholds."""
        zbar_gated = []
        zbar_gated_ids = []
        for zbar, zbar_id in zip(predicted_measurements, predicted_measurements_ids):
            range = zbar[0] 
            bearing = zbar[1] 
            if range < self.cfg.range_gate and np.abs(bearing) < np.deg2rad(self.cfg.fov_gate_deg)/2:
                zbar_gated.append(zbar)
                zbar_gated_ids.append(zbar_id)  
    
        return np.array(zbar_gated), np.array(zbar_gated_ids)


    def extract_covariance(self, zbar_ids: np.ndarray) -> np.ndarray:
        """Extract joint covariance for predicted measurements from marginals."""
        
        pose_prev_key = X(self.num_steps - 1)
        pose = self.values.atPose2(pose_prev_key)
        
        keys = [pose_prev_key]  # NOTE: order in which the keys are added is important
        keys += [L(id) for id in zbar_ids]   
          
        marginals = gtsam.Marginals(self.graph, self.values)

        # Joint covariance for previous pose and local landmarks (in local frame)
        cov_local  = marginals.jointMarginalCovariance(keys).fullMatrix()

        # Reorder covariance to match state ordering
        cov_local  = reorder_covariance_auto(
            cov_local ,
            source_keys=sorted(keys),
            target_keys=keys,
            values=self.values,
        )

        # Rotate to world frame
        R_WL = rotmat2(pose.theta())
        E_mat = np.eye(cov_local .shape[0])
        E_mat[:2, :2] = R_WL
        cov_global = E_mat @ cov_local  @ E_mat.T  # TODO: optimize construction

        return cov_global

    def propagate_covariance(
        self, Sigma_prev_W: np.ndarray, pose_prev: gtsam.Pose2, odometry: gtsam.Pose2,
    ):
        """Propagate covariance to predicted pose frame."""
        F = np.eye(Sigma_prev_W.shape[0])
        F[0:3, 0:3] = self.motion_model.F_x(
        pose2_to_array(pose_prev), pose2_to_array(odometry)
        )

        Q = np.zeros((Sigma_prev_W.shape[0], Sigma_prev_W.shape[0]))
        Q[0:3, 0:3] = self.motion_model.Q(
            pose2_to_array(pose_prev), pose2_to_array(odometry)
        )  # does already multiply with jacobian internally

        Sigma_pred_W = F @ Sigma_prev_W @ F.T + Q
        return Sigma_pred_W

    def compute_innovation_covariance(
        self,
        zbar_ids: np.ndarray,
        pose_pred: gtsam.Pose2,
        cov_world: np.ndarray,
    ):
        """Compute innovation covariance for predicted measurements."""
        
        n = len(zbar_ids) # num predicted measurements

        H = np.zeros((2 * n, 3 + 2 * n))
        R = np.zeros((2 * n, 2 * n))
 
        for i, id in enumerate(zbar_ids):
            m_i = self.values.atPoint2(L(id))
            H_x = self.sensor_model.H_x(pose2_to_array(pose_pred), m_i)
            H_mi = self.sensor_model.H_m(pose2_to_array(pose_pred), m_i)
            H[2 * i : 2 * i + 2, 0:3] = H_x
            H[2 * i : 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2] = H_mi
            R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.diag(self.cfg.noise.measurement_vec**2)

        S = H @ cov_world @ H.T + R
        S = make_psd(S)  # TODO: ensure this isnt messing up results

        return S

    def compute_association(self, z: np.ndarray, zbar: np.ndarray, zbar_ids, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """Compute association between measurements and predicted measurements using self.associator."""
        
        # Do association
        association_indices = self.associator.associate(z, zbar, S)
    
        # As Assoicator.associat returns indices into zbar for each measurement, we need to convert to landmark IDs
        association_ids = association_indices.copy()  
        association_ids[association_indices >= 0] = [zbar_ids[idx] for idx in association_indices if idx >= 0]

        return association_ids, association_indices
    
    # def process_step(self, z_odometry: tuple[float, float, float], z_range_bearing: list[tuple[float, float]]) -> StepRecord:
    def process_step(self, odometry: gtsam.Pose2, measurements: np.ndarray) -> StepRecord:
        """Main SLAM step processing:

        Parameters
        -------
        z_odometry : gtsam.Pose2
            (u, v, psi) representing odometry measurement (relative motion)
        z_range_bearing: np.ndarray
            (range, bearing) measurements to landmarks

        Returns
        -------
        StepRecord 
            current estimates, measurements, associations etc. for this step

        """

        if self.num_steps == 0:
            asssoc_ids = np.full(measurements.shape[0], -1, dtype=int)
            assoc_idx  = np.full(measurements.shape[0], -1, dtype=int)

            zbar_gated = np.empty((0, 2), dtype=float)
            zbar_gated_ids = np.empty((0,), dtype=int)
        else:
            pose_prev = self.values.atPose2(X(self.num_steps - 1))
            pose_pred = pose_prev.compose(odometry)
            
            zbar, zbar_ids = self.get_predicted_measurements(pose_pred)
            zbar_gated, zbar_gated_ids = self.gate_predicted_measurements(zbar, zbar_ids)
          
            cov_prev_W = self.extract_covariance(zbar_gated_ids)
            cov_pred_W = self.propagate_covariance(cov_prev_W, pose_prev, odometry)
            cov_innovation = self.compute_innovation_covariance(zbar_gated_ids, pose_pred, cov_pred_W)
            asssoc_ids, assoc_idx = self.compute_association(measurements, zbar_gated, zbar_gated_ids, cov_innovation)
            
        # print(asssocations)

        self.update_graph(odometry, measurements, asssoc_ids)
        self.optimize_graph()

        # ---- store history ----
        record = StepRecord(
            step=self.num_steps,
            poses=self.get_estimated_poses(), 
            landmarks=self.get_estimated_landmarks(), 
            measurements=measurements,
            predicted_measurements=zbar_gated,
            predicted_measurements_ids=zbar_gated_ids,
            associations_ids=asssoc_ids,
            associations_idx=assoc_idx,
        )

        # Important: update current step
        self.num_steps += 1

        return record

    def update_graph(
        self,
        odometry: gtsam.Pose2,
        landmark_measurements: np.ndarray,
        associations: np.ndarray,
    ) -> None:
        """
        Process one SLAM step

        Args:
            odometry: Relative motion from previous pose (None for first step)
            landmark_measurements: 
            associations: Corresponding landmark IDs for each measurement

        Returns:
            Optimized values after this step
        """

        # Add odometry factor
        if odometry:
            self._add_odometry(odometry)

        # Add range-bearing factors and initialize landmarks
        if len(landmark_measurements):
            self._add_landmark_measurements(landmark_measurements, associations)


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

    def _add_odometry(self, odometry: gtsam.Pose2):
        """Add odometry factor between consecutive poses"""
        from_idx = self.num_steps
        to_idx = self.num_steps + 1

        odom_factor = gtsam.BetweenFactorPose2(
            X(from_idx), X(to_idx), odometry, self.odometry_noise
        )
        self.graph.add(odom_factor)
        self.new_factors.add(odom_factor)

        # Predict next pose for initialization
        prev_pose = self.values.atPose2(X(from_idx))
        predicted_pose = prev_pose.compose(odometry)
        self.values.insert(X(to_idx), predicted_pose)
        self.new_values.insert(X(to_idx), predicted_pose)

    def _add_landmark_measurements(
        self, 
        measurements: np.ndarray, # (M,2)
        associations: np.ndarray, # (M,) 
    ):
        """Add landmark measurement factors"""
        pose_key = X(self.num_steps)

        # j is measurement index, a_j is associated landmark index
        for (r, b), a_j in zip(measurements, associations):
            if a_j >= 0:  # measurement j associated with previously observed landmark a_j
                meas_factor = gtsam.BearingRangeFactor2D(pose_key, L(a_j), gtsam.Rot2(b), r, self.measurement_noise)
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)
            elif a_j == -1:  # new landmark, initialize and add factor
                lm_key = L(self.num_landmarks)
                self.num_landmarks += 1
                meas_factor = gtsam.BearingRangeFactor2D(pose_key, lm_key, gtsam.Rot2(b), r, self.measurement_noise_birth)
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
        return self.values.atPose2(X(self.num_steps))
    
    def get_estimated_poses(self) -> np.ndarray:
        """Get all pose estimates up to current step"""
        return np.array([pose2_to_array(self.values.atPose2(X(k))) for k in range(self.num_steps)])

    def get_estimated_landmarks(self) -> np.ndarray:
        """Get all landmark estimates up to current step"""
        return  np.array([self.values.atPoint2(L(lm)) for lm in range(self.num_landmarks)])
