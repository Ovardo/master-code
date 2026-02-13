from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
        self._add_prior_factor(gtsam.Pose2(initial_pose), initial_pose_noise)


    def _add_prior_factor(self, prior_pose: gtsam.Pose2, prior_pose_noise: np.ndarray):
        """Add prior factor for initial pose."""
 
        prior_factor = gtsam.PriorFactorPose2(X(0), prior_pose, prior_pose_noise)
        
        self.graph.add(prior_factor)
        self.values.insert(X(0), prior_pose)

        self.new_factors.add(prior_factor)
        self.new_values.insert(X(0), prior_pose)

        self.optimize_graph()
        self.num_steps += 1  # count prior as first step


    def get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> list[PredictedMeasurement]:
        """Get predicted measurements for all landmarks based on predicted pose estimate and current landmark estimates."""
        zbar_list = []
        for lm_id in range(self.num_landmarks):
            m = self.values.atPoint2(L(lm_id))
            r = pose_pred.range(m)
            b = pose_pred.bearing(m).theta() 
            zbar = np.array([r, b])
            zbar_list.append(PredictedMeasurement(lm_id=lm_id, zbar=zbar))
        return zbar_list
    
    def gate_predicted_measurements(self, zbar_list: list[PredictedMeasurement]) -> list[PredictedMeasurement]:
        """Gate predicted measurements based on range and bearing thresholds."""
        zbar_gated_list = []
        for zbar in zbar_list:
            range = zbar.zbar[0]
            bearing = zbar.zbar[1]
            if range < self.cfg.range_gate and bearing < np.deg2rad(self.cfg.fov_gate_deg):
                zbar_gated_list.append(zbar)
        return zbar_gated_list

    def extract_covariance(self, zbar_list: list[PredictedMeasurement]) -> np.ndarray:
        """Extract joint covariance for predicted measurements from marginals."""
        
        pose_prev_key = X(self.num_steps - 1)
        pose = self.values.atPose2(pose_prev_key)
        
        keys = [pose_prev_key]  # NOTE: order in which the keys are added is important
        keys += [L(zbar.lm_id) for zbar in zbar_list]   
          
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
        zbar_list: list[PredictedMeasurement],
        pose_pred: gtsam.Pose2,
        cov_world: np.ndarray,
    ):
        """Compute innovation covariance for predicted measurements."""
        n = len(zbar_list) # num predicted measurements

        H = np.zeros((2 * n, 3 + 2 * n))
        R = np.zeros((2 * n, 2 * n))

        for i, zbar in enumerate(zbar_list):
            m_key = L(zbar.lm_id)
            m_i = self.values.atPoint2(m_key)
            H_x = self.sensor_model.H_x(pose2_to_array(pose_pred), m_i)
            H_mi = self.sensor_model.H_m(pose2_to_array(pose_pred), m_i)
            H[2 * i : 2 * i + 2, 0:3] = H_x
            H[2 * i : 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2] = H_mi
            R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.diag(self.cfg.noise.measurement_vec**2)

        S = H @ cov_world @ H.T + R
        S = make_psd(S)  # TODO: ensure this isnt messing up results

        return S

    def compute_association(self, z_list: list[tuple[float, gtsam.Rot2]], zbar_list: list[PredictedMeasurement], S: np.ndarray) -> list[int]: 
        """Compute association between measurements and predicted measurements using self.associator."""
        
        # Convert to numpy arrays for associator
        measurements = np.array([(z[0], z[1].theta()) for z in z_list])
        predicted_measurements = np.array([(zbar.zbar) for zbar in zbar_list])
        
        # Do association
        association = self.associator.associate(measurements, predicted_measurements, S)
    
        # As Assoicator.associat returns indices into zbar for each measurement, we need to convert to landmark IDs
        association_lm_ids = []
        for a in association:
            if a >= 0:
                lm_id = zbar_list[a].lm_id
                association_lm_ids.append(lm_id)
            else:
                association_lm_ids.append(a)  # keep -1 and -2 as is

        return association
    
    # def process_step(self, z_odometry: tuple[float, float, float], z_range_bearing: list[tuple[float, float]]) -> StepRecord:
    def process_step(self, z_odometry: gtsam.Pose2, z_range_bearing: list[tuple[float, gtsam.Rot2]]) -> StepRecord:
        """Main SLAM step processing:

        Parameters
        -------
        z_odometry : gtsam.Pose2
            (u, v, psi) representing odometry measurement (relative motion)
        z_range_bearing: list[tuple[float, gtsam.Rot2]]
            (range, bearing) measurements to landmarks

        Returns
        -------
        StepRecord 
            current estimates, measurements, associations etc. for this step

        """

        if self.num_steps == 0:
            asssocations = [-1] * len(z_range_bearing)
        else:
            pose_prev = self.values.atPose2(X(self.num_steps - 1))
            pose_pred = pose_prev.compose(z_odometry)
            
            zbar_list = self.get_predicted_measurements(pose_pred)
            zbar_gated_list = self.gate_predicted_measurements(zbar_list)
          
            cov_prev_W = self.extract_covariance(zbar_gated_list)
            cov_pred_W = self.propagate_covariance(cov_prev_W, pose_prev, z_odometry)
            cov_innovation = self.compute_innovation_covariance(zbar_gated_list, pose_pred, cov_pred_W)
            asssocations = self.compute_association(z_range_bearing, zbar_gated_list, cov_innovation)
            
        print(asssocations)

        self.update_graph(z_odometry, z_range_bearing, asssocations)
        self.optimize_graph()

        # ---- store history ----
        record = StepRecord(
            step=self.num_steps,
            poses=None,  # TODO
            landmarks=None,  # TODO
            measurements=z_range_bearing,
            predicted_measurements=zbar_gated_list,
            predicted_pose=pose_pred,
            associations=asssocations,
            cov_innovation=cov_innovation,
        )

        # Important: update current step
        self.num_steps += 1

        return record

    def update_graph(
        self,
        odometry: Optional[gtsam.Pose2],
        landmark_measurements: list[tuple[float, gtsam.Rot2]],
        associations: list[int],
    ) -> None:
        """
        Process one SLAM step

        Args:
            odometry: Relative motion from previous pose (None for first step)
            landmark_measurements: list of (range, bearing) measurements
            associations: Corresponding landmark IDs for each measurement

        Returns:
            Optimized values after this step
        """

        # Add odometry factor
        if odometry:
            self._add_odometry(odometry)

        # Add range-bearing factors and initialize landmarks
        if landmark_measurements:
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
        from_idx = self.num_steps - 1
        to_idx = self.num_steps

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
        self, measurements: list[tuple[float, gtsam.Rot2]], associations: list[int]
    ):
        """Add landmark measurement factors"""
        pose_key = X(self.num_steps)

        # j is measurement index, a_j is associated landmark index
        for j, ((z_range, z_bearing), a_j) in enumerate(zip(measurements, associations)):
            if a_j == -2:  # ambiguous association, could be outlier or valid match. For now, treat as outlier (no factor)
                continue
            elif a_j > -1:  # measurement j associated with previously observed landmark a_j
                meas_factor = gtsam.BearingRangeFactor2D(
                    pose_key, L(a_j), z_bearing, z_range, self.measurement_noise
                )
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)

            else:  # a_j = -1, i.e measurement j not associated with any landmark
                lm_key = L(self.num_landmarks)
                self.num_landmarks += 1
                measure_factor = gtsam.BearingRangeFactor2D(
                    pose_key,
                    lm_key,
                    z_bearing,
                    z_range,
                    self.measurement_noise_birth,
                )
                self.graph.add(measure_factor)
                self.new_factors.add(measure_factor)

                self._initialize_landmark(lm_key, pose_key, z_range, z_bearing)

    def _initialize_landmark(self, lm_key: int, pose_key: int, range: float, bearing: gtsam.Rot2) -> None:
        """Initialize a newly observed landmark."""
        current_pose = self.values.atPose2(pose_key)

        # Convert from polar to Cartesian in robot frame
        lm_x_local = range * np.cos(bearing.theta())
        lm_y_local = range * np.sin(bearing.theta())
        
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
    
    def get_estimated_poses(self) -> list[gtsam.Pose2]:
        """Get all pose estimates up to current step"""
        return [self.values.atPose2(X(k)) for k in range(self.num_steps)]

    def get_estimated_landmarks(self) -> list[gtsam.Point2]:
        """Get all landmark estimates up to current step"""
        return  [self.values.atPoint2(L(l)) for l in range(self.num_landmarks)]
