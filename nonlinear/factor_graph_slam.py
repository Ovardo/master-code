import gtsam
import numpy as np

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from association import JCBB, make_psd
from models.dynamicmodels import OdometrySE2
from models.measurementmodels import RangeBearing
from utilities.cov_reorder import reorder_covariance_auto
from utilities.utils import pose2_to_array, rotmat2
from gtsam.symbol_shorthand import X, L
from tuning import NonlinearFactorGraphParams


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
        self.graph.add(gtsam.PriorFactorPose2(X(0), prior_pose, self.prior_noise))
        self.values.insert(X(0), prior_pose)
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
       

    def compute_association(self, measurements, predicted_measurements, S) -> List[int]: # predicted_measurements, S, alpha_individ, alpha_joint
        """Placeholder for JCBB data association logic"""
        # For now, return dummy associations (all -1)

        alpha_ind = self.config.alpha_individual
        alpha_jnt = self.config.alpha_joint

        z = np.zeros((len(measurements), 2)) 
        for j, z_j in enumerate(measurements): # turn List[(float, Rot2)] into (Mx2) np array 
            z[j] = np.array([z_j[0], z_j[1].theta()])

        association_hypothesis = JCBB(z, predicted_measurements, S, alpha_ind, alpha_jnt)
   
        return association_hypothesis

    
    def process_step(self, 
                     odometry: Optional[gtsam.Pose2],
                     z_range_bearing: List[Tuple[float, gtsam.Rot2]]) -> gtsam.Values:
        
        local_predicted_measurements = []
        pose_pred = None

        if self.config.association_type == "jcbb":
            if self.current_step == 0:
                associations = [-1] * len(z_range_bearing)  # No associations at first step
            else:
                local_landmarks_keys, pose_pred   = self.local_feature_filtering(odometry)
                Sigma_prev_W, _                   = self.covariance_extraction(local_landmarks_keys)
                Sigma_pred_W                      = self.covariance_propagation(Sigma_prev_W, pose_pred, odometry)
                S_k, local_predicted_measurements = self.innovation_covariance_computation(local_landmarks_keys, pose_pred, Sigma_pred_W)
                associations                      = self.compute_association(z_range_bearing, local_predicted_measurements, S_k)
        elif self.config.association_type == "known":
            associations = self.gt_associations[self.current_step]
        
        self.update_graph(odometry, z_range_bearing, associations)
        self.optimize_graph()
        
        # Step 4: Store results
        self.history.add_estimate(self.current_step, self.values)
        self.history.add_measurements(self.current_step, z_range_bearing)
        self.history.add_predicted_measurements(self.current_step, local_predicted_measurements)
        self.history.add_predicted_pose(self.current_step, pose_pred)
        self.history.add_associations(self.current_step, associations)

        # Increment step
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
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values, self.config.optimizer_params
        )
        self.values = optimizer.optimize()

    
    def _add_odometry(self, odometry: gtsam.Pose2):
        """Add odometry factor between consecutive poses"""
        from_idx = self.current_step - 1
        to_idx = self.current_step
        
        # Predict next pose for initialization
        prev_pose = self.values.atPose2(X(from_idx))
        predicted_pose = prev_pose.compose(odometry)
        self.values.insert(X(to_idx), predicted_pose)
        
        odom_factor = gtsam.BetweenFactorPose2( 
            X(from_idx), X(to_idx), odometry, self.odometry_noise
        )
        
        self.graph.add(odom_factor)
        
        # Update graph for ISAM2
        self.new_factors.add(odom_factor)
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
                self.graph.add(gtsam.BearingRangeFactor2D(
                    current_pose_key, L(a_j), z_bearing, z_range, self.measurement_noise
                ))
            else: # a_j = -1, i.e measurement j not associated with any landmark
                # TODO; add logic for false alarms if needed
                lm_key = L(self.num_landmarks)
                self.landmark_keys.add(lm_key) # this updates num_landmarks property btw
               
                meas_factor = gtsam.BearingRangeFactor2D(
                    current_pose_key, lm_key, z_bearing, z_range, self.measurement_noise
                )
                
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)

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
        self.new_values.insert(lm_key, landmark_global) # TODO: check if needed
    
    def _compute_prediction(self) -> Dict:
        """
        Compute predicted measurements for data association
        (Currently using ground truth associations, but this prepares for JCBB)
        """
        marginals = gtsam.Marginals(self.graph, self.values)
        
        prev_idx = self.current_step - 1
        prev_pose = self.values.atPose2(X(prev_idx))
        
        # Get relevant covariance
        state_keys = [X(prev_idx)] + list(self.landmark_keys)
        P_prev = marginals.jointMarginalCovariance(state_keys).fullMatrix()
        
        # Reorder covariance to match state ordering
        P_prev = reorder_covariance_auto(
            P_prev, 
            source_keys=sorted(state_keys),
            target_keys=state_keys,
            values=self.values
        )
        
        # Rotate to global frame
        E_mat = self._build_rotation_matrix(prev_pose.theta(), len(self.landmark_keys))
        g_P_prev = E_mat @ P_prev @ E_mat.T
        
        # Predict measurements
        state_vector = self._build_state_vector(prev_pose, self.landmark_keys)
        z_pred, S = self.sensor_model.predict_measurements(state_vector, g_P_prev)
        
        return {
            'predicted_measurements': z_pred,
            'innovation_covariance': S,
            'state_covariance': g_P_prev
        }
    
    def _build_state_vector(self, pose: gtsam.Pose2, lm_keys: set) -> np.ndarray:
        """Build state vector [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]"""
        pose_array = pose2_to_array(pose)
        
        landmarks_array = []
        for lm_key in sorted(lm_keys):
            lm_pos = self.values.atPoint2(lm_key)
            landmarks_array.extend([lm_pos[0], lm_pos[1]])
        
        return np.hstack([pose_array, landmarks_array])
    
    def _build_rotation_matrix(self, theta: float, num_landmarks: int) -> np.ndarray:
        """Build rotation matrix for covariance transformation"""
        dim = 3 + 2 * num_landmarks
        E_mat = np.eye(dim)
        E_mat[:2, :2] = rotmat2(theta) 
        return E_mat
    
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



class SLAMHistory:
    """Store SLAM estimation history for visualization and analysis"""
    
    def __init__(self):
        self.estimates: List[gtsam.Values] = []
        self.measurements: List[List] = []
        self.predicted_measurements: List[List] = []
        self.predicted_pose: List[List] = []
        self.associations: List[List[int]] = []
        self.timestamps: List[int] = []
    
    def add_estimate(self, step: int, values: gtsam.Values):
        """Store optimized estimate"""
        import copy
        self.estimates.append(copy.deepcopy(values))
        self.timestamps.append(step)

    def add_measurements(self, step: int, measurements: List):
        """Store actual measurements"""
        self.measurements.append(measurements) 

    def add_predicted_pose(self, step: int, measurements: List):
        """Store predicted measurements"""
        self.predicted_pose.append(measurements)
    
    def add_predicted_measurements(self, step: int, z_pred: List):
        """Store predicted measurements"""
        self.predicted_measurements.append(z_pred)

    def add_associations(self, step: int, associations: List[int]):
        """Store data associations"""
        self.associations.append(associations)

    def get_measurements_at_step(self, step: int) -> Optional[List]:
        """Retrieve measurements at specific step"""
        if step < len(self.measurements):
            return self.measurements[step]
        return None
    
    
    def get_predicted_measurements_at_step(self, step: int) -> Optional[List]:
        """Retrieve predicted measurements at specific step"""
        if step < len(self.predicted_measurements):
            return self.predicted_measurements[step]
        return None
    
    def get_estimate_at_step(self, step: int) -> Optional[gtsam.Values]:
        """Retrieve estimate at specific step"""
        if step < len(self.estimates):
            return self.estimates[step]
        return None
    
    def get_all_estimates(self) -> List[gtsam.Values]:
        """Get all stored estimates"""
        return self.estimates
    
    def get_all_measurements(self) -> List[List]:
        """Get all stored measurements"""
        return self.measurements
    
    def get_all_predicted_measurements(self) -> List[List]:
        """Get all stored predicted measurements"""
        return self.predicted_measurements


class SLAMVisualizer:
    """Handle SLAM visualization"""
    
    @staticmethod
    def plot_final_result(slam: FactorGraphSLAM, 
                         marginals: Optional[gtsam.Marginals] = None,
                         figsize=(22, 6)):
        """Plot final SLAM result with covariances"""
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot
        
        if marginals is None:
            marginals = slam.get_marginals()
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect('equal')
        ax.set_title("Nonlinear 2D SLAM with Marginals")
        
        # Plot poses
        for k in range(slam.num_poses):
            pose_key = X(k)
            pose = slam.values.atPose2(pose_key)
            cov = marginals.marginalCovariance(pose_key)
            gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5, covariance=cov)
        
        # Plot landmarks
        for lm_key in slam.landmark_keys:
            lm_pos = slam.values.atPoint2(lm_key)
            cov = marginals.marginalCovariance(lm_key)
            gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='b', P=cov)
        
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_ground_truth(poses_gt: List[gtsam.Pose2], 
                          landmarks_gt: List[gtsam.Point2],
                          figsize=(22, 6)):
        """Plot ground truth trajectory and landmarks"""
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot  
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect('equal')
        ax.set_title("Ground Truth Trajectory and Landmarks")
        # Plot ground truth poses
        for k, pose in enumerate(poses_gt):
            gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5)
        # Plot ground truth landmarks
        for lm_pos in landmarks_gt:
            gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='go')
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_measurements_space(slam: FactorGraphSLAM,
                          step: int,
                          axis_length: float = 0.5,
                          figsize=(8, 8)):
        """Plot measurements at a specific step"""
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect('equal')
        ax.set_title(f"Measurements at Step {step}")    

        z_list = slam.history.get_measurements_at_step(step)
        z_pred = slam.history.get_predicted_measurements_at_step(step)

    
    @staticmethod
    def plot_step_by_step(slam: FactorGraphSLAM, 
                         subplot_size: float = 4.0,
                         axis_length: float = 0.5,
                         margin_fraction: float = 0.2,
                         min_margin: float = 0.5):
        """
        Plot SLAM evolution step-by-step in a grid of subplots
        
        Args:
            slam: FactorGraphSLAM object with history
            subplot_size: Size of each subplot in inches
            axis_length: Length of pose axis arrows
            margin_fraction: Fraction of span to add as margin
            min_margin: Minimum margin in meters
        """
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot
        
        plt.ioff()  # Turn off interactive mode for static multi-plot display
        
        estimates = slam.history.get_all_estimates()
        
        K = len(estimates)
        
        if K == 0:
            print("No estimates to plot!")
            return None, None
        
        # Compute grid layout
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))
        
        # Compute global axis limits across all estimates
        xlim, ylim = SLAMVisualizer._compute_global_limits(
            estimates, slam, margin_fraction, min_margin
        )
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
        
        # Flatten axes for easy indexing
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        # Plot each estimate
        for idx, est in enumerate(estimates):
            ax = axes_flat[idx]
            ax.set_aspect('equal')
            ax.set_title(f"Step {idx}/{K-1}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            
            # Plot poses up to current step
            for k in range(idx + 1):
                pose_key = X(k)
                if est.exists(pose_key):
                    pose = est.atPose2(pose_key)
                    gtsam_plot.plot_pose2_on_axes(ax, pose, axis_length=axis_length)
            
            # Plot all observed landmarks up to current step
            for lm_key in slam.landmark_keys:
                if est.exists(lm_key):
                    lm_pos = est.atPoint2(lm_key)
                    gtsam_plot.plot_point2_on_axes(ax, lm_pos, linespec='b')
                      
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
    def _compute_global_limits(estimates: List[gtsam.Values], 
                              slam: FactorGraphSLAM,
                              margin_fraction: float = 0.2,
                              min_margin: float = 0.5) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute global axis limits across all estimates
        
        Returns:
            ((xmin, xmax), (ymin, ymax))
        """
        xs = []
        ys = []
        
        for est in estimates:
            # Collect pose positions
            for k in range(slam.num_poses):
                pose_key = X(k)
                if est.exists(pose_key):
                    pose = est.atPose2(pose_key)
                    xs.append(pose.x())
                    ys.append(pose.y())
            
            # Collect landmark positions
            for lm_key in slam.landmark_keys:
                if est.exists(lm_key):
                    lm_pos = est.atPoint2(lm_key)
                    xs.append(lm_pos[0])
                    ys.append(lm_pos[1])
        
        # Fallback if nothing collected
        if len(xs) == 0:
            return ((-1, 1), (-1, 1))
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # Add margin
        xspan = max(1e-3, xmax - xmin)
        yspan = max(1e-3, ymax - ymin)
        x_margin = max(min_margin, margin_fraction * xspan)
        y_margin = max(min_margin, margin_fraction * yspan)
        
        xlim = (xmin - x_margin, xmax + x_margin)
        ylim = (ymin - y_margin, ymax + y_margin)
        
        return xlim, ylim
    
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