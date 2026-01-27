from __future__ import annotations

from dataclasses import dataclass, field
import copy
from scipy.stats import chi2

import gtsam
import numpy as np

from typing import Dict, List, Optional, Tuple, Any
from association import JCBB, make_psd
from models.dynamicmodels import OdometrySE2
from models.measurementmodels import RangeBearing
from utilities.cov_reorder import reorder_covariance_auto
from utilities.utils import pose2_to_array, rotmat2, wrapToPi
from gtsam.symbol_shorthand import X, L
from tuning import NonlinearFactorGraphParams
from association import NIS

from utilities.plot_utils import plot_ellipse, plot_se2_covariance_on_manifold_gtsam, plot_pose2_trajectory, plot_pose2_on_axes, MultivariateNormalParameters


# @dataclass
# class SLAMConfig: # TODO: just use NonlinearFactorGRaphParams instead 
#     """Configuration parameters for SLAM"""
#     prior_noise_sigmas: np.ndarray
#     odometry_noise_sigmas: np.ndarray
#     measurement_noise_sigmas: np.ndarray
#     motion_sigma_x: float
#     motion_sigma_y: float
#     motion_sigma_theta: float
#     sensor_sigma_range: float
#     sensor_sigma_bearing: float
#     optimizer_params: gtsam.LevenbergMarquardtParams = field(default_factory=gtsam.LevenbergMarquardtParams)


class FactorGraphSLAM:
    """Main SLAM estimator using factor graphs"""
    
    def __init__(self, config: NonlinearFactorGraphParams, prior_pose: gtsam.Pose2, gt_associations: Optional[Dict[int, List[int]]] = None):
        self.config = config
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
       
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
       
        self.current_step = 0
        self.gt_associations = gt_associations
        
        # Models
        self.motion_model = OdometrySE2(
            sigma_x=config.sigma_x,
            sigma_y=config.sigma_y,
            sigma_theta=config.sigma_theta
        )
        self.sensor_model = RangeBearing(
            sigma_range=config.sigma_range,
            sigma_bearing=config.sigma_bearing
        )
        
        # Noise models
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(config.P_x0_vec)
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(config.Q_vec)
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(config.R_vec)
        self.measurement_noise_birth = gtsam.noiseModel.Diagonal.Sigmas(config.R_vec * 1.0)  # higher uncertainty for new landmarks
        
        # State tracking
        self.landmark_keys = set()  # Track observed landmarks
        self.history = SLAMHistory()  # For visualization/analysis
        
        # Initialize with prior
        self._add_prior(prior_pose)
    
    @property
    def num_landmarks(self) -> int:
        return len(self.landmark_keys)
        
    def _add_prior(self, prior_pose: gtsam.Pose2):
        """Add prior factor for initial pose"""
        prior_factor = gtsam.PriorFactorPose2(X(0), prior_pose, self.prior_noise)
        self.graph.add(prior_factor)
        self.values.insert(X(0), prior_pose)

        self.new_factors.add(prior_factor)
        self.new_values.insert(X(0), prior_pose)
        
        self.optimize_graph()
        
        
        self.history.add_estimate(self.current_step, self.values)
    
    def local_feature_filtering(self, odometry: gtsam.Pose2):
        pose_prev = self.values.atPose2(X(self.current_step - 1))
        pose_pred = pose_prev.compose(odometry)
        local_landmarks_keys = []
        for i, key_i in enumerate(self.landmark_keys):
            m_i = self.values.atPoint2(key_i)
            range_pred = pose_pred.range(m_i)
            if range_pred < self.config.r_local:
                local_landmarks_keys.append(key_i)  
        return local_landmarks_keys, pose_pred

    def covariance_extraction(self, local_landmarks_keys: set):
        pose_prev_key = X(self.current_step - 1)
        pose_prev = self.values.atPose2(pose_prev_key)
        local_keys = [pose_prev_key] + local_landmarks_keys
        marginals = gtsam.Marginals(self.graph, self.values)
        
        # Joint covariance for previous pose and local landmarks (in local frame)
        Sigma_prev_L = marginals.jointMarginalCovariance(local_keys).fullMatrix()
        
        # Reorder covariance to match state ordering
        Sigma_prev_L = reorder_covariance_auto(Sigma_prev_L, source_keys=sorted(local_keys),   
                                               target_keys=local_keys, values=self.values)
        
        # Rotate to world frame
        R_WL = rotmat2(pose_prev.theta())
        E_mat = np.eye(Sigma_prev_L.shape[0]) 
        E_mat[:2, :2] = R_WL
        Sigma_prev_W = E_mat @ Sigma_prev_L @ E_mat.T # TODO: optimize construction
        
        return Sigma_prev_W, Sigma_prev_L
    
    def covariance_propagation(self, Sigma_prev_W: np.ndarray, pose_prev: gtsam.Pose2, odometry: gtsam.Pose2):
        """Propagate covariance to predicted pose frame"""
        F = np.eye(Sigma_prev_W.shape[0])
        F[0:3, 0:3] = self.motion_model.F_x(pose2_to_array(pose_prev), pose2_to_array(odometry))
        
        Q = np.zeros((Sigma_prev_W.shape[0], Sigma_prev_W.shape[0]))
        Q[0:3, 0:3] = self.motion_model.Q(pose2_to_array(pose_prev), pose2_to_array(odometry)) # does already multiply with jacobian internally
   
        Sigma_pred_W = F @ Sigma_prev_W @ F.T + Q
        return Sigma_pred_W
        
    def innovation_covariance_computation(self, local_landmarks_keys: set, pose_pred: gtsam.Pose2, Sigma_pred_W: np.ndarray):
        """Compute innovation covariance S for local landmarks at predicted pose"""
        num_local_landmarks = len(local_landmarks_keys)

        local_predicted_measurements = np.zeros((num_local_landmarks, 2))
        for i, key_i in enumerate(local_landmarks_keys):
            m_i = self.values.atPoint2(key_i)
            range_pred = pose_pred.range(m_i) # float
            bearing_pred = pose_pred.bearing(m_i) # Rot2
            local_predicted_measurements[i] = (range_pred, bearing_pred.theta())
        
        
        H = np.zeros((2 * num_local_landmarks, 3 + 2 * num_local_landmarks))
        R = np.zeros((2 * num_local_landmarks, 2 * num_local_landmarks))
        for i, key_i in enumerate(local_landmarks_keys):
            m_i = self.values.atPoint2(key_i)
            H_x = self.sensor_model.H_x(pose2_to_array(pose_pred), m_i)
            H_mi = self.sensor_model.H_m(pose2_to_array(pose_pred), m_i)
            H[2*i:2*i+2, 0:3] = H_x
            H[2*i:2*i+2, 3+2*i:3+2*i+2] = H_mi
            R[2*i:2*i+2, 2*i:2*i+2] = np.diag([self.config.sigma_range**2, self.config.sigma_bearing**2])
        
        S = H @ Sigma_pred_W @ H.T + R
        S = make_psd(S) # TODO: ensure this isnt messing up results

        return S, local_predicted_measurements
       

    def compute_association(self, measurements, predicted_measurements, S, local_landmarks_ids) -> List[int]: # predicted_measurements, S, alpha_individ, alpha_joint
        """Placeholder for JCBB data association logic"""
        # For now, return dummy associations (all -1)

        alpha_ind = self.config.alpha_individual
        alpha_jnt = self.config.alpha_joint

        z = np.zeros((len(measurements), 2)) 
        for j, z_j in enumerate(measurements): # turn List[(float, Rot2)] into (Mx2) np array 
            z[j] = np.array([z_j[0], z_j[1].theta()])

        association_hyp_local = JCBB(z, predicted_measurements, S, alpha_ind, alpha_jnt)

        # Convert from local landmark indices to global landmark IDs
        i_to_id = {i: lm_id for i, lm_id in enumerate(local_landmarks_ids)}
        association_hyp_global = [i_to_id[a_j] if a_j > -1 else -1 for a_j in association_hyp_local]
   
        return association_hyp_global, association_hyp_local


    def process_step(self, odometry, z_range_bearing):
        
        local_predicted_measurements = np.empty((0,2))
        pose_pred = None
        local_landmarks_keys = []
        S_k = None

        if self.config.association_type == "jcbb":
            if self.current_step == 0:
                ass_global = [-1] * len(z_range_bearing)
                ass_local = [-1] * len(z_range_bearing)
            else:
                local_landmarks_keys, pose_pred = self.local_feature_filtering(odometry)
                local_landmark_ids = [gtsam.symbolIndex(key) for key in local_landmarks_keys]
                
                Sigma_prev_W, _ = self.covariance_extraction(local_landmarks_keys)
                Sigma_pred_W = self.covariance_propagation(Sigma_prev_W, pose_pred, odometry)
                S_k, local_predicted_measurements = self.innovation_covariance_computation(local_landmarks_keys, pose_pred, Sigma_pred_W)
                ass_global, ass_local = self.compute_association(z_range_bearing, local_predicted_measurements, S_k, local_landmark_ids)

        elif self.config.association_type == "known":
            ass_global = self.gt_associations[self.current_step]
            ass_local = ass_global

        self.update_graph(odometry, z_range_bearing, ass_global)
        self.optimize_graph()
        # self.new_factors = gtsam.NonlinearFactorGraph() # reset new factors
        # self.new_values = gtsam.Values() # reset new values

        # ---- store history ----
        rec = self.history.ensure_step(self.current_step)
        rec.estimate = copy.deepcopy(self.values)
        rec.measurements = list(z_range_bearing)
        rec.predicted_measurements = np.asarray(local_predicted_measurements, dtype=float).reshape(-1, 2)
        rec.predicted_pose = pose_pred
        rec.associations = list(ass_global) 
        rec.associations_local = list(ass_local) 
        rec.innovation_covariance = S_k
        rec.cov_last_pose = self.isam.marginalCovariance(X(self.current_step)) if self.config.use_isam else gtsam.Marginals(self.graph, self.values).marginalCovariance(X(self.current_step)) 

        # mapping predicted index i -> landmark id
        rec.local_landmark_ids = [gtsam.symbolIndex(key) for key in local_landmarks_keys]

        # optional extras for later:
        rec.S = None if S_k is None else np.asarray(S_k, dtype=float)

        self.current_step += 1
        return self.values

    
    def update_graph(self, 
                     odometry: Optional[gtsam.Pose2],
                     landmark_measurements: List[Tuple[float, gtsam.Rot2]],
                     associations: List[int]) -> gtsam.Values:
        """
        Process one SLAM step
        
        Args:
            odometry: Relative motion from previous pose (None for first step)
            landmark_measurements: List of (range, bearing) measurements
            associations: Corresponding landmark IDs for each measurement
            
        Returns:
            Optimized values after this step
        """
        
        # Add odometry factor
        if odometry:
            self._add_odometry(odometry)
        
        # Add range-bearing factors and initialize landmarks
        if landmark_measurements:
            if self.config.association_type == "jcbb":
                self._add_landmark_measurements_jcbb(landmark_measurements, associations)
            elif self.config.association_type == "known":
                self._add_landmark_measurements(landmark_measurements, associations)
            else:
                raise ValueError(f"Unknown association type: {self.config.association_type}")

    def optimize_graph(self) -> gtsam.Values:
        """Run optimization on the current factor graph"""
        if self.config.use_isam:
            self.isam.update(self.new_factors, self.new_values)
            self.new_factors = gtsam.NonlinearFactorGraph()
            self.new_values = gtsam.Values()
            self.values = self.isam.calculateEstimate()
            
        else: # full batch optimization
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.values, self.config.optimizer_params
            )
            self.values = optimizer.optimize()

    
    def _add_odometry(self, odometry: gtsam.Pose2):
        """Add odometry factor between consecutive poses"""
        from_idx = self.current_step - 1
        to_idx = self.current_step
        
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
    
    def _add_landmark_measurements(self, 
                                   measurements: List[Tuple[float, gtsam.Rot2]], 
                                   landmark_ids: List[int]):
        """Add landmark measurement factors"""
        current_pose_key = X(self.current_step)
        
        for (z_range, z_bearing), lm_id in zip(measurements, landmark_ids):
            lm_key = L(lm_id)
            
            # Add measurement factor
            meas_factor = gtsam.BearingRangeFactor2D(
                current_pose_key, lm_key, z_bearing, z_range, self.measurement_noise
            )
            self.graph.add(meas_factor)
            self.new_factors.add(meas_factor)
            
            # Initialize landmark if first observation
            if lm_key not in self.landmark_keys:
                self._initialize_landmark(lm_key, current_pose_key, z_range, z_bearing)
                self.landmark_keys.add(lm_key)
    
    def _add_landmark_measurements_jcbb(self, 
                                   measurements: List[Tuple[float, gtsam.Rot2]], 
                                   associations: List[int]):
        """Add landmark measurement factors"""
        current_pose_key = X(self.current_step)
        
        # j is measurement index, a_j is associated landmark index
        for j, ((z_range, z_bearing), a_j) in enumerate(zip(measurements, associations)):
            
            if a_j > -1: # measurement j associated with previously observed landmark a_j
                # Add measurement factor
                meas_factor = gtsam.BearingRangeFactor2D(
                    current_pose_key, L(a_j), z_bearing, z_range, self.measurement_noise
                )
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)
             
            else: # a_j = -1, i.e measurement j not associated with any landmark
                # TODO; add logic for false alarms if needed
                lm_key = L(self.num_landmarks)
                self.landmark_keys.add(lm_key) # this updates num_landmarks property btw
                measure_factor = gtsam.BearingRangeFactor2D(
                    current_pose_key, lm_key, z_bearing, z_range, self.measurement_noise_birth
                )
                self.graph.add(measure_factor)
                self.new_factors.add(measure_factor)

                self._initialize_landmark(lm_key, current_pose_key, z_range, z_bearing)
                
    
    def _initialize_landmark(self, lm_key: int, pose_key: int, 
                            range_val: float, bearing: gtsam.Rot2):
        """Initialize a newly observed landmark"""
        current_pose = self.values.atPose2(pose_key)
        
        # Convert from polar to Cartesian in robot frame
        delta_x = range_val * np.cos(bearing.theta())
        delta_y = range_val * np.sin(bearing.theta())
        
        # Transform to global frame
        landmark_global = current_pose.transformFrom(gtsam.Point2(delta_x, delta_y))
        self.values.insert(lm_key, landmark_global)
        self.new_values.insert(lm_key, landmark_global)


    def get_marginals(self) -> gtsam.Marginals:
        """Compute marginals for current estimate"""
        return gtsam.Marginals(self.graph, self.values)
    
    def get_current_pose(self) -> gtsam.Pose2:
        """Get current robot pose estimate"""
        return self.values.atPose2(X(self.current_step))
    
    def get_landmark_positions(self) -> Dict[int, gtsam.Point2]:
        """Get all landmark position estimates"""
        return {
            key: self.values.atPoint2(key)
            for key in self.landmark_keys
        }
    
    @property
    def num_poses(self) -> int:
        return self.current_step
    
    @property
    def num_landmarks(self) -> int:
        return len(self.landmark_keys)


####################################################################
####################################################################


@dataclass
class StepRecord:
    step: int
    estimate: Optional[gtsam.Values] = None

    measurements: Optional[List[Tuple[float, gtsam.Rot2]]] = None
    predicted_measurements: Optional[np.ndarray] = None  # (L,2) array [range, bearing]
    local_landmark_ids: Optional[List[int]] = None  # landmark ids for predicted_measurements
    predicted_pose: Optional[gtsam.Pose2] = None

    associations: Optional[List[int]] = None  # length M, each in {-1, 0..}
    associations_local: Optional[List[int]] = None  # length M' (for filtered landmarks)
    # Optional future fields (nice to have for JCBB analysis)
    innovation_covariance: Optional[np.ndarray] = None            # innovation covariance
    d2: Optional[np.ndarray] = None           # per-measurement Mahalanobis^2
    correct_mask: Optional[np.ndarray] = None # per-measurement correctness (if GT)
    cov_last_pose: Optional[np.ndarray] = None  # covariance of last pose


class SLAMHistory:
    """Stores per-step SLAM history robustly (one record per step)."""

    def __init__(self):
        self._records: Dict[int, StepRecord] = {}

    # --------- core API ---------
    def ensure_step(self, step: int) -> StepRecord:
        if step not in self._records:
            self._records[step] = StepRecord(step=step)
        return self._records[step]

    def add_estimate(self, step: int, values: gtsam.Values, deep_copy: bool = True) -> None:
        rec = self.ensure_step(step)
        rec.estimate = copy.deepcopy(values) if deep_copy else values

    def add_measurements(self, step: int, measurements: List[Tuple[float, gtsam.Rot2]]) -> None:
        rec = self.ensure_step(step)
        # shallow copy is enough; Rot2 is immutable-ish, but list might be mutated
        rec.measurements = list(measurements)

    def add_predicted_pose(self, step: int, pose_pred: Optional[gtsam.Pose2]) -> None:
        rec = self.ensure_step(step)
        rec.predicted_pose = pose_pred

    def add_predicted_measurements(self, step: int, z_pred: Any) -> None:
        """
        Accepts list-like or np.ndarray. Stored as np.ndarray with shape (L,2) when possible.
        """
        rec = self.ensure_step(step)
        if z_pred is None:
            rec.predicted_measurements = None
            return
        arr = np.asarray(z_pred, dtype=float)
        rec.predicted_measurements = arr

    def add_associations(self, step: int, associations: List[int]) -> None:
        rec = self.ensure_step(step)
        rec.associations = list(associations)

    # --------- optional JCBB analysis fields ---------
    def add_innovation_covariance(self, step: int, S: Optional[np.ndarray]) -> None:
        rec = self.ensure_step(step)
        rec.S = None if S is None else np.asarray(S, dtype=float)

    def add_mahalanobis2(self, step: int, d2: Optional[np.ndarray]) -> None:
        rec = self.ensure_step(step)
        rec.d2 = None if d2 is None else np.asarray(d2, dtype=float)

    def add_correct_mask(self, step: int, mask: Optional[np.ndarray]) -> None:
        rec = self.ensure_step(step)
        rec.correct_mask = None if mask is None else np.asarray(mask, dtype=bool)

    # --------- getters / properties ---------
    @property
    def steps(self) -> List[int]:
        return sorted(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    def get(self, step: int) -> Optional[StepRecord]:
        return self._records.get(step, None)

    def get_or_raise(self, step: int) -> StepRecord:
        rec = self.get(step)
        if rec is None:
            raise KeyError(f"No history recorded for step={step}")
        return rec

    def get_estimate(self, step: int) -> Optional[gtsam.Values]:
        rec = self.get(step)
        return None if rec is None else rec.estimate

    def get_measurements(self, step: int) -> Optional[List[Tuple[float, gtsam.Rot2]]]:
        rec = self.get(step)
        return None if rec is None else rec.measurements

    def get_predicted_measurements(self, step: int) -> Optional[np.ndarray]:
        rec = self.get(step)
        return None if rec is None else rec.predicted_measurements

    def get_associations(self, step: int) -> Optional[List[int]]:
        rec = self.get(step)
        return None if rec is None else rec.associations

    def all_records(self) -> List[StepRecord]:
        return [self._records[k] for k in self.steps]




class SLAMVisualizer:
    """Handle SLAM visualization"""

    @staticmethod
    def plot_measurement_space(
        slam,
        step: int,
        show_lines: bool = True,
        show_labels: bool = True,
        figsize=(7, 5),
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        rec = slam.history.get_or_raise(step)

        if rec.measurements is None:
            raise ValueError(f"No measurements stored for step={step}")

        z = np.array([[r, b.theta()] for (r, b) in rec.measurements], dtype=float)  # (M,2)

        zhat = rec.predicted_measurements
        if zhat is None:
            zhat = np.empty((0, 2), dtype=float)
        else:
            zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)

        assoc = rec.associations if rec.associations is not None else []
        local_ids = getattr(rec, "local_landmark_ids", None)

        fig, ax = plt.subplots(figsize=figsize)

        # predicted + measured
        if len(zhat) > 0:
            ax.scatter(zhat[:, 0], zhat[:, 1], marker="o", label="predicted")
        if len(z) > 0:
            ax.scatter(z[:, 0], z[:, 1], marker="x", label="measured")

        # association lines measured -> predicted
        if show_lines and local_ids is not None and len(local_ids) == len(zhat) and len(assoc) == len(z):
            id_to_i = {lm_id: i for i, lm_id in enumerate(local_ids)}
            i_to_id = {i: lm_id for i, lm_id in enumerate(local_ids)}

            assoc_arr = np.asarray(assoc, dtype=int)
            new_mask = assoc_arr == -1

            for j, a_j in enumerate(assoc_arr):
                if a_j == -1:
                    continue
                i = id_to_i.get(int(a_j), None)
                if i is None:
                    continue

                ax.plot([z[j, 0], zhat[i, 0]], [z[j, 1], zhat[i, 1]], linewidth=1, alpha=0.6)

                if show_labels:
                    ax.text(z[j, 0], z[j, 1], f"{a_j}", fontsize=8, alpha=0.8)

            # mark unassociated
            if np.any(new_mask):
                ax.scatter(z[new_mask, 0], z[new_mask, 1], marker="x", label="unassociated (-1)")

        ax.set_title(f"Measurement space (step {step})")
        ax.set_xlabel("range [m]")
        ax.set_ylabel("bearing [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_NIS(slam, figsize=(13, 3), ax=None, show_expected=True):
        import matplotlib.pyplot as plt
        
        steps = list(slam.history.steps)
        N = len(steps)

        nis_sequence  = np.full(N, np.nan, dtype=float)
        dof_sequence  = np.zeros(N, dtype=int)
        lower_bounds  = np.full(N, np.nan, dtype=float)
        upper_bounds  = np.full(N, np.nan, dtype=float)

        for k, step in enumerate(steps):
            if step == 0:
                continue  # skip first step (no measurements)
            rec = slam.history.get_or_raise(step)

            if rec.innovation_covariance is None:
                raise ValueError(f"No innovation covariance stored for step={step}")

            S = rec.innovation_covariance
            z = rec.measurements
            zhat = rec.predicted_measurements
            
            assoc = np.array(rec.associations_local)
            z = np.array([[r, b.theta()] for (r, b) in z], dtype=float)  # (M,2)

            # number of associated landmark measurements (each landmark gives 2D measurement)
            num_assoc = np.sum(assoc > -1)
            dof = 2 * num_assoc

            dof_sequence[k] = dof

            # If no associations, NIS is not meaningful (0 dof -> chi2 not defined nicely)
            if dof <= 0:
                continue

            nis_sequence[k] = NIS(z, zhat, S, assoc)

            lower, upper = chi2.interval(slam.config.alpha_joint, df=dof)
            lower_bounds[k] = lower
            upper_bounds[k] = upper

        # ---- plotting ----
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(N)

        ax.plot(x, nis_sequence, label="NIS", linewidth=1.8)
        ax.plot(x, lower_bounds, "--", label=r"$\chi^2_{{dof},1-\alpha_{joint}}$", linewidth=1.2)
        ax.plot(x, upper_bounds, "--", label=r"$\chi^2_{{dof},\alpha_{joint}}$", linewidth=1.2)

        if show_expected:
            # E[chi2(dof)] = dof
            expected = np.where(dof_sequence > 0, dof_sequence.astype(float), np.nan)
            ax.plot(x, expected, ":", label="E[NIS] = dof", linewidth=1.2)

        ax.set_title("NIS consistency over time")
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("NIS")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax
    
    def plot_NEES(slam, gt_poses):
        steps = list(slam.history.steps)
        N = len(steps)

        nees_sequence = np.full(N, np.nan, dtype=float)
        lower_bounds  = np.full(N, np.nan, dtype=float)
        upper_bounds  = np.full(N, np.nan, dtype=float)

        dof = 3  # Pose2 minimal dimension
        alpha = 0.95

        for k, step in enumerate(steps):
            rec = slam.history.get_or_raise(step)
            est = rec.estimate
            cov = rec.cov_last_pose

            if est is None:
                continue

            if step >= len(gt_poses):
                continue

            pose_est = est.atPose2(X(step))
            pose_gt = gt_poses[step]

            error = pose2_to_array(pose_est.between(pose_gt))  # in minimal coordinates

            nees_sequence[k] = error.T @ np.linalg.inv(cov) @ error

            lower, upper = chi2.interval(alpha, df=dof)
            lower_bounds[k] = lower
            upper_bounds[k] = upper
        
        # ---- plotting ----
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 3))
        x = np.arange(N)
        ax.plot(x, nees_sequence, label="NEES", linewidth=1.8)
        ax.plot(x, lower_bounds, "--", label=f"Lower bound (α={alpha:g})", linewidth=1.2)
        ax.plot(x, upper_bounds, "--", label=f"Upper bound (α={alpha:g})", linewidth=1.2)
        ax.set_title("NEES consistency over time")
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("NEES")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax
    

    @staticmethod
    def plot_error(slam, gt_poses):
        import matplotlib.pyplot as plt
        steps = list(slam.history.steps)
        N = len(steps)

        # errors (x, y, theta) and sigmas
        err = np.full((N, 3), np.nan, dtype=float)
        sig = np.full((N, 3), np.nan, dtype=float)

        for k, step in enumerate(steps):
            rec = slam.history.get_or_raise(step)
            est = rec.estimate
            cov = rec.cov_last_pose  # expected 3x3 in (x,y,theta) minimal coords

            if est is None or cov is None:
                continue
            if step >= len(gt_poses):
                continue

            pose_est = est.atPose2(X(step))
            pose_gt  = gt_poses[step]

            # Minimal error coordinates: Pose2 "between" -> (dx, dy, dtheta)
            e = pose2_to_array(pose_est.between(pose_gt))
            e[2] = wrapToPi(e[2])

            err[k, :] = e
            sig[k, :] = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        # ---- plotting ----
        labels = ["x error [m]", "y error [m]", "yaw error [rad]"]
        fig, axs = plt.subplots(3, 1, figsize=(13, 5.5), sharex=True)

        x = np.arange(N)
        for i, ax in enumerate(axs):
            ax.plot(x, err[:, i], linewidth=1.6, label="Error")

            # envelopes
            ax.fill_between(x, -2*sig[:, i],  2*sig[:, i], alpha=0.25, label="±2σ")
            ax.fill_between(x, -3*sig[:, i],  3*sig[:, i], alpha=0.15, label="±3σ")

            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axs[-1].set_xlabel("Timestep index")
        fig.suptitle("Pose estimation error with 2σ/3σ covariance envelopes", y=0.98)
        fig.tight_layout()
        return fig, axs

    

    @staticmethod
    def plot_result_step(
        slam,
        step: int,
        marginals: Optional[gtsam.Marginals] = None,
        poses_gt: Optional[List[gtsam.Pose2]] = None,
        landmarks_gt: Optional[List[gtsam.Point2]] = None,
        poses_dead_reckoning: Optional[List[gtsam.Pose2]] = None,
        show_covariances: bool = True,
        show_landmarks: bool = True,
        axis_length: float = 0.5,
        figsize=(22, 6),
        ax=None,
        title: Optional[str] = None,
        show_orientations: bool = True,


    ):
        """
        Plot estimate at a given step using history (StepRecord).

        Notes on covariances:
          - `marginals` must correspond to the same (graph, values) solution.
          - If you pass the current slam.get_marginals(), it usually corresponds to the final step.
        """
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot

        rec = slam.history.get_or_raise(step)
        if rec.estimate is None:
            raise ValueError(f"No estimate stored for step={step}")

        est = rec.estimate

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        ax.set_aspect("equal")
        if title is None:
            title = f"SLAM result at step {step}"
            if show_covariances and marginals is not None:
                title += " (with marginals)"
        ax.set_title(title)

        # ----- Plot estimated poses up to step -----
        x_coords = []
        y_coords = []
        for k in range(step + 1):
            pose_key = X(k)
            if not est.exists(pose_key):
                continue
            pose = est.atPose2(pose_key)
            x_coords.append(pose.x())
          
            y_coords.append(pose.y())
        ax.plot(x_coords, y_coords, '-r', label=r"$\hat{x}$")

        for k in range(step + 1):
            pose_key = X(k)
            if not est.exists(pose_key):
                continue

            pose = est.atPose2(pose_key)

            if show_covariances and (marginals is not None):
                try:
                    cov = marginals.marginalCovariance(pose_key)
                    plot_se2_covariance_on_manifold_gtsam(ax, dist=MultivariateNormalParameters(mean=pose, covariance=cov), fill_alpha=0.2, fill_color="red", linestyle="none")
                    #plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, show_axis=show_orientations)
                    #gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, covariance=cov)
                except Exception:
                    gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)
            else:
                gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)

        # ----- Plot estimated landmarks (that exist in this estimate) -----
        if show_landmarks:
            # Count how many landmarks exist in this estimate
            est_landmark_count = 0
            for lm_key in slam.landmark_keys:
                if not est.exists(lm_key):
                    continue

                lm_pos = est.atPoint2(lm_key)
                est_landmark_count += 1

                if show_covariances and (marginals is not None):
                    try:
                        cov = marginals.marginalCovariance(lm_key)
                        ax.plot(lm_pos[0], lm_pos[1], 'ob')
                        plot_ellipse(ax, MultivariateNormalParameters(mean=lm_pos, covariance=cov), fill_alpha=0.2, fill_color="blue", linestyle="", linewidth=0.8)
                        #gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b", P=cov)
                    except Exception:
                        gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b")
                else:
                    gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b")

            # Add a legend entry indicating the number of estimated landmarks
            try:
                ax.plot([], [], 'ob', label=f"$\\hat{{m}}$ (#{est_landmark_count})")
            except Exception:
                pass
 
        # ----- Optional: overlay GT on same axes -----
        if poses_gt is not None:
            for pose in poses_gt[: step + 1]:
                plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, marker='x', color='green')
                # gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)
            ax.plot([], [], 'gx', label="$x_{GT}$")

        if landmarks_gt is not None:
            for lm_pos in landmarks_gt:
                ax.plot(lm_pos[0], lm_pos[1], 'x', color='orange')
                # gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="gx")
            ax.plot([], [], 'x', color='orange', label=r"$m_{GT}$")


        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        return fig, ax

    @staticmethod
    def plot_final_result(
        slam,
        marginals: Optional[gtsam.Marginals] = None,
        poses_gt: Optional[List[gtsam.Pose2]] = None,
        landmarks_gt: Optional[List[gtsam.Point2]] = None,
        poses_dead_reckoning: Optional[List[gtsam.Pose2]] = None,
        **kwargs,
    ):
        if len(slam.history) == 0:
            raise ValueError("No history in slam.history")
        last_step = slam.history.steps[-1]
        return SLAMVisualizer.plot_result_step(
            slam,
            step=last_step,
            marginals=marginals,
            poses_dead_reckoning=poses_dead_reckoning,
            poses_gt=poses_gt,
            landmarks_gt=landmarks_gt,
            **kwargs,
        )

    # @staticmethod
    # def plot_final_result(slam: FactorGraphSLAM, 
    #                      marginals: Optional[gtsam.Marginals] = None,
    #                      figsize=(22, 6)):
    #     """Plot final SLAM result with covariances"""
    #     import matplotlib.pyplot as plt
    #     from gtsam.utils import plot as gtsam_plot
        
    #     if marginals is None:
    #         marginals = slam.get_marginals()
        
    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     ax.set_aspect('equal')
    #     ax.set_title("Nonlinear 2D SLAM with Marginals")
        
    #     # Plot poses
    #     for k in range(slam.num_poses):
    #         pose_key = X(k)
    #         pose = slam.values.atPose2(pose_key)
    #         cov = marginals.marginalCovariance(pose_key)
    #         gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5, covariance=cov)
        
    #     # Plot landmarks
    #     for lm_key in slam.landmark_keys:
    #         lm_pos = slam.values.atPoint2(lm_key)
    #         cov = marginals.marginalCovariance(lm_key)
    #         gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='b', P=cov)
        
    #     plt.tight_layout()
    #     return fig, ax

    # @staticmethod
    # def plot_ground_truth(poses_gt: List[gtsam.Pose2], 
    #                       landmarks_gt: List[gtsam.Point2],
    #                       figsize=(22, 6)):
    #     """Plot ground truth trajectory and landmarks"""
    #     import matplotlib.pyplot as plt
    #     from gtsam.utils import plot as gtsam_plot  
    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     ax.set_aspect('equal')
    #     ax.set_title("Ground Truth Trajectory and Landmarks")
    #     # Plot ground truth poses
    #     for k, pose in enumerate(poses_gt):
    #         gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5)
    #     # Plot ground truth landmarks
    #     for lm_pos in landmarks_gt:
    #         gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='go')
    #     plt.tight_layout()
    #     return fig, ax

    
    @staticmethod
    def plot_step_by_step(
        slam,
        subplot_size: float = 4.0,
        axis_length: float = 0.5,
        margin_fraction: float = 0.2,
        min_margin: float = 0.5,
    ):
        """
        Plot SLAM evolution step-by-step in a grid of subplots using StepRecords.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot

        plt.ioff()

        steps = slam.history.steps
        K = len(steps)
        if K == 0:
            print("No estimates to plot!")
            return None, None

        # Compute grid layout
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))

        # Compute global axis limits across all stored estimates
        xlim, ylim = SLAMVisualizer._compute_global_limits_from_history(
            slam, steps, margin_fraction, min_margin
        )

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # Plot each record
        for plot_idx, step in enumerate(steps):
            ax = axes_flat[plot_idx]
            rec = slam.history.get_or_raise(step)
            est = rec.estimate

            ax.set_aspect("equal")
            ax.set_title(f"Step {step} ({plot_idx}/{K-1})")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            if est is None:
                ax.text(0.5, 0.5, "No estimate", transform=ax.transAxes, ha="center", va="center")
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                continue

            # Plot poses up to current step
            for k in range(step + 1):
                pose_key = X(k)
                if est.exists(pose_key):
                    pose = est.atPose2(pose_key)
                    gtsam_plot.plot_pose2_on_axes(ax, pose, axis_length=axis_length)

            # Plot observed landmarks (that exist in this estimate)
            for lm_key in slam.landmark_keys:
                if est.exists(lm_key):
                    lm_pos = est.atPoint2(lm_key)
                    gtsam_plot.plot_point2_on_axes(ax, lm_pos, linespec="b")

            # Apply global limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(K, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        return fig, axes

    
    @staticmethod
    def _compute_global_limits_from_history(
        slam,
        steps,
        margin_fraction: float = 0.2,
        min_margin: float = 0.5,
    ):
        xs, ys = [], []

        for step in steps:
            rec = slam.history.get(step)
            if rec is None or rec.estimate is None:
                continue
            est = rec.estimate

            # poses up to this step
            for k in range(step + 1):
                pose_key = X(k)
                if est.exists(pose_key):
                    pose = est.atPose2(pose_key)
                    xs.append(pose.x())
                    ys.append(pose.y())

            # landmarks present in this estimate
            for lm_key in slam.landmark_keys:
                if est.exists(lm_key):
                    lm = est.atPoint2(lm_key)
                    xs.append(lm[0])
                    ys.append(lm[1])

        if len(xs) == 0:
            return (-1, 1), (-1, 1)

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        xspan = max(1e-3, xmax - xmin)
        yspan = max(1e-3, ymax - ymin)

        x_margin = max(min_margin, margin_fraction * xspan)
        y_margin = max(min_margin, margin_fraction * yspan)

        return (xmin - x_margin, xmax + x_margin), (ymin - y_margin, ymax + y_margin)

    
    @staticmethod
    def plot_trajectory_with_uncertainty(slam: FactorGraphSLAM,
                                        marginals: Optional[gtsam.Marginals] = None,
                                        show_landmarks: bool = True,
                                        figsize=(12, 8)):
        """
        Plot robot trajectory with uncertainty ellipses
        
        Args:
            slam: FactorGraphSLAM object
            marginals: Pre-computed marginals (computed if None)
            show_landmarks: Whether to show landmarks
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot
        
        if marginals is None:
            marginals = slam.get_marginals()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_title("Robot Trajectory with Uncertainty")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        
        # Plot trajectory line
        trajectory_x = []
        trajectory_y = []
        for k in range(slam.num_poses):
            pose = slam.values.atPose2(X(k))
            trajectory_x.append(pose.x())
            trajectory_y.append(pose.y())
        
        ax.plot(trajectory_x, trajectory_y, 'r--', alpha=0.5, linewidth=1, label='Trajectory')
        
        # Plot poses with covariance
        for k in range(slam.num_poses):
            pose_key = X(k)
            pose = slam.values.atPose2(pose_key)
            cov = marginals.marginalCovariance(pose_key)
            gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5, covariance=cov)
        
        # Plot landmarks if requested
        if show_landmarks:
            for lm_key in slam.landmark_keys:
                lm_pos = slam.values.atPoint2(lm_key)
                cov = marginals.marginalCovariance(lm_key)
                gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='b', P=cov)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax


    @staticmethod
    def plot_measurement_space_step_by_step(
        slam,
        subplot_size: float = 4.0,
        show_lines: bool = True,
        show_labels: bool = False,
        margin_fraction: float = 0.1,
        min_margin_r: float = 0.5,
        min_margin_b: float = 0.1,
    ):
        """
        Plot measurement-space evolution step-by-step (range vs bearing) in a grid of subplots.

        Uses StepRecords in slam.history:
          - rec.measurements: list of (range, bearing_obj) where bearing_obj.theta() is used
          - rec.predicted_measurements: (N,2) array-like of [range, bearing]
          - rec.associations: list length M, with landmark ids or -1 for new/unassociated
          - rec.local_landmark_ids: list length N matching predicted_measurements
        """

        plt.ioff()

        steps = slam.history.steps
        K = len(steps)
        if K == 0:
            print("No history to plot!")
            return None, None

        # ---- global limits across all steps ----
        xlim, ylim = SLAMVisualizer._compute_global_meas_limits_from_history(
            slam,
            steps,
            margin_fraction=margin_fraction,
            min_margin_r=min_margin_r,
            min_margin_b=min_margin_b,
        )

        # ---- grid layout ----
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for plot_idx, step in enumerate(steps):
            ax = axes_flat[plot_idx]
            rec = slam.history.get_or_raise(step)

            ax.set_title(f"Step {step} ({plot_idx}/{K-1})")
            ax.set_xlabel("range [m]")
            ax.set_ylabel("bearing [rad]")

            if rec.measurements is None:
                ax.text(0.5, 0.5, "No measurements", transform=ax.transAxes,
                        ha="center", va="center")
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                continue

            z = np.array([[r, b.theta()] for (r, b) in rec.measurements], dtype=float)  # (M,2)

            zhat = rec.predicted_measurements
            if zhat is None:
                zhat = np.empty((0, 2), dtype=float)
            else:
                zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)

            # predicted + measured
            if len(zhat) > 0:
                ax.scatter(zhat[:, 0], zhat[:, 1], marker="o", label="predicted")
            if len(z) > 0:
                ax.scatter(z[:, 0], z[:, 1], marker="x", label="measured")

            # association lines measured -> predicted
            assoc = rec.associations if rec.associations is not None else []
            local_ids = getattr(rec, "local_landmark_ids", None)

            if (
                show_lines
                and local_ids is not None
                and len(local_ids) == len(zhat)
                and len(assoc) == len(z)
            ):
                id_to_i = {int(lm_id): i for i, lm_id in enumerate(local_ids)}
                assoc_arr = np.asarray(assoc, dtype=int)
                new_mask = assoc_arr == -1

                for j, a_j in enumerate(assoc_arr):
                    if a_j == -1:
                        continue
                    i = id_to_i.get(int(a_j), None)
                    if i is None:
                        continue

                    ax.plot(
                        [z[j, 0], zhat[i, 0]],
                        [z[j, 1], zhat[i, 1]],
                        linewidth=1,
                        alpha=0.6,
                    )
                    if show_labels:
                        ax.text(z[j, 0], z[j, 1], f"{a_j}", fontsize=8, alpha=0.8)

                if np.any(new_mask):
                    ax.scatter(z[new_mask, 0], z[new_mask, 1], marker="x", label="unassociated (-1)")

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

            # only show legend if something was plotted
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(fontsize=8)

        # Hide unused axes
        for i in range(K, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def _compute_global_meas_limits_from_history(
        slam,
        steps,
        margin_fraction: float = 0.1,
        min_margin_r: float = 0.5,
        min_margin_b: float = 0.1,
    ):
        import numpy as np

        rs, bs = [], []

        for step in steps:
            rec = slam.history.get(step)
            if rec is None:
                continue

            if rec.measurements is not None:
                z = np.array([[r, b.theta()] for (r, b) in rec.measurements], dtype=float)
                if z.size > 0:
                    rs.extend(z[:, 0].tolist())
                    bs.extend(z[:, 1].tolist())

            zhat = rec.predicted_measurements
            if zhat is not None:
                zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)
                if zhat.size > 0:
                    rs.extend(zhat[:, 0].tolist())
                    bs.extend(zhat[:, 1].tolist())

        if len(rs) == 0:
            return (-1, 1), (-1, 1)

        rmin, rmax = float(np.min(rs)), float(np.max(rs))
        bmin, bmax = float(np.min(bs)), float(np.max(bs))

        rspan = max(1e-6, rmax - rmin)
        bspan = max(1e-6, bmax - bmin)

        r_margin = max(min_margin_r, margin_fraction * rspan)
        b_margin = max(min_margin_b, margin_fraction * bspan)

        return (rmin - r_margin, rmax + r_margin), (bmin - b_margin, bmax + b_margin)
