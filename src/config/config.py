from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    """Uncertainty parameters for SLAM.

    The odometry model is:
        ΔT_k = f(u_k + γ_u,k) + γ_o,k
    where:
        γ_u,k ~ N(0, Q_u): noise on raw odometry inputs (velocity, steering)
        γ_o,k ~ N(0, Q_o): additive noise on the resulting relative pose

    The resulting covariance used in preintegration is:
        Q_k = J_u Q_u J_u^T + Q_o

    The measurement model is:
        z_kj = h(x_k, m_j) + η_kj
    where:
        η_kj ~ N(0, R): noise on landmark observations (range, bearing)

    Attributes:
        odom_input_vel_std (float): Standard deviation of raw odometry input noise in velocity [m/s].
        odom_input_alpha_deg_std (float): Standard deviation of raw odometry input noise in steering angle [deg].
        odom_output_x_std (float): Standard deviation of additive odometry output noise in relative x [m].
        odom_output_y_std (float): Standard deviation of additive odometry output noise in relative y [m].
        odom_output_yaw_deg_std (float): Standard deviation of additive odometry output noise in relative heading [deg].
        landmark_range_std (float): Standard deviation of landmark observation noise in range [m].
        landmark_bearing_deg_std (float): Standard deviation of landmark observation noise in bearing [deg].
        init_pose_x_std (float): Standard deviation of initial pose uncertainty in x [m].
        init_pose_y_std (float): Standard deviation of initial pose uncertainty in y [m].
        init_pose_yaw_deg_std (float): Standard deviation of initial pose uncertainty in heading [deg].
    """
    odom_input_vel_std: float = 0.1
    odom_input_alpha_deg_std: float = 0.5

    odom_output_x_std: float = 0.1
    odom_output_y_std: float = 0.1
    odom_output_yaw_deg_std: float = 0.1

    landmark_range_std: float = 0.2
    landmark_bearing_deg_std: float = 1.0

    init_pose_x_std: float = 0.05
    init_pose_y_std: float = 0.05
    init_pose_yaw_deg_std: float = 0.5

    def __post_init__(self) -> None:
        """Validate noise parameters."""
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"Noise parameter {field_name} must be > 0, got {value}")

    @property
    def odom_input_alpha_rad_std(self) -> float:
        return np.deg2rad(self.odom_input_alpha_deg_std)

    @property
    def odom_output_yaw_rad_std(self) -> float:
        return np.deg2rad(self.odom_output_yaw_deg_std)

    @property
    def landmark_bearing_rad_std(self) -> float:
        return np.deg2rad(self.landmark_bearing_deg_std)

    @property
    def init_pose_yaw_rad_std(self) -> float:
        return np.deg2rad(self.init_pose_yaw_deg_std)

    @property
    def odom_input_cov(self) -> np.ndarray:
        """Return diagonal covariance matrix of raw odometry input noise in [v, alpha]."""
        return np.diag([
            self.odom_input_vel_std ** 2,
            self.odom_input_alpha_rad_std ** 2,
        ])

    @property
    def odom_output_cov(self) -> np.ndarray:
        """Return diagonal covariance matrix of additive odometry output noise in [x, y, yaw]."""
        return np.diag([
            self.odom_output_x_std ** 2,
            self.odom_output_y_std ** 2,
            self.odom_output_yaw_rad_std ** 2,
        ])

    @property
    def landmark_cov(self) -> np.ndarray:
        """Return diagonal landmark observation covariance matrix in [range, bearing]."""
        return np.diag([
            self.landmark_range_std ** 2,
            self.landmark_bearing_rad_std ** 2,
        ])

    @property
    def init_pose_cov(self) -> np.ndarray:
        """Return diagonal initial pose covariance matrix in [x, y, yaw]."""
        return np.diag([
            self.init_pose_x_std ** 2,
            self.init_pose_y_std ** 2,
            self.init_pose_yaw_rad_std ** 2,
        ])

    @property
    def gtsam_landmark_sigmas(self) -> np.ndarray:
        """Return landmark noise sigmas for GTSAM in [bearing, range] order."""
        return np.array([
            self.landmark_bearing_rad_std,
            self.landmark_range_std,
        ], dtype=np.float64)

    @property
    def gtsam_prior_pose_sigmas(self) -> np.ndarray:
        """Return prior pose sigmas for GTSAM in [x, y, yaw] order."""
        return np.array([
            self.init_pose_x_std,
            self.init_pose_y_std,
            self.init_pose_yaw_rad_std,
        ], dtype=np.float64)

      
@dataclass
class TentativeLandmarkManagerConfig:
    """Configuration for landmark management in SLAM.
    
    Attributes:
        M (int): Hits needed to confirm a landmark.
        N (int): Max lifetime (in steps) of tentative landmark.
        gate (float): Euclidean distance threshold for associating unassociated 
            measurements to existing tentative landmarks.
    """
    M: int = 3
    N: int = 4
    gate: float = 0.1

    def __post_init__(self):
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.M > self.N:
            raise ValueError(f"M must be <= N, got M={self.M} and N={self.N}")
        if self.gate <= 0:
            raise ValueError(f"gate must be positive, got {self.gate}")


@dataclass
class SensorConfig:
    """Sensor configuration parameters.
    
    Attributes:
        max_range (float): Maximum sensor range in meters. Must be positive.
        fov_deg (float): Sensor field of view in degrees. Must be in (0, 360].
    """
    max_range: float = 50.0
    fov_deg: float = 250.0

    def __post_init__(self):
        if self.max_range <= 0:
            raise ValueError(f"max_range must be positive, got {self.max_range}")
        if not (0 < self.fov_deg <= 360):
            raise ValueError(f"fov_deg must be in (0, 360], got {self.fov_deg}")


@dataclass
class AssociationConfig:
    """Data association configuration for landmark matching.
    
    Attributes:
        method (str): Association method. Options: "gt" (only for sim), "jcbb", "ml", "nn", "cnn".
        alpha_individual (float): Confidence level for individual compatibility test.
        alpha_joint (float): Confidence level for joint compatibility test.
        range_gate (float): Local feature filtering radius [m].
        fov_gate_deg (float): Local feature filtering field of view [deg].
    """
    method: str = "jcbb"
    alpha_individual: float = 0.999
    alpha_joint: float = 0.9999

    def __post_init__(self):
        methods_options = ["jcbb", "ml"] # TODO: include "gt" and "nn"
        if self.method not in methods_options:
            raise ValueError(f"Invalid association_type {self.method}, must be one of {methods_options}")
        if not (0 < self.alpha_individual < 1):
            raise ValueError(f"alpha_individual must be in (0, 1), got {self.alpha_individual}")
        if not (0 < self.alpha_joint < 1):
            raise ValueError(f"alpha_joint must be in (0, 1), got {self.alpha_joint}")

        






