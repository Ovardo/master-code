from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    """Uncertainty parameters for SLAM.

    The relative pose process model is:
        ΔT_k = f(u_k + γ_u,k) + γ_o,k
    where:
        γ_u,k ~ N(0, Q_u): noise on control inputs (velocity, steering)
        γ_o,k ~ N(0, Q_o): additive noise on the resulting relative pose

    The resulting covariance used in preintegration is:
        Q_k = J_u Q_u J_u^T + Q_o % TODO: change if not the case

    The measurement model is:
        z_kj = h(x_k, m_j) + η_kj
    where:
        η_kj ~ N(0, R): noise on landmark observations (range, bearing)

    Attributes:
        sigma_velocity (float): Standard deviation of control input velocity [m/s].
        sigma_steer_deg (float): Standard deviation of control input steering angle [deg].
        sigma_odom_x (float): Standard deviation of additive odometry noise in relative x [m].
        sigma_odom_y (float): Standard deviation of additive odometry noise in relative y [m].
        sigma_odom_yaw_deg (float): Standard deviation of additive odometry noise in relative heading [deg].
        sigma_range (float): Standard deviation of landmark observation noise in range [m].
        sigma_bearing_deg (float): Standard deviation of landmark observation noise in bearing [deg].
        sigma_init_pose_x (float): Standard deviation of initial pose uncertainty in x [m].
        sigma_init_pose_y (float): Standard deviation of initial pose uncertainty in y [m].
        sigma_init_pose_yaw_deg (float): Standard deviation of initial pose uncertainty in heading [deg].
    """
    sigma_velocity: float = 0.1
    sigma_steer_deg: float = 0.5

    sigma_odom_x: float = 0.1
    sigma_odom_y: float = 0.1
    sigma_odom_yaw_deg: float = 0.1

    sigma_range: float = 0.2
    sigma_bearing_deg: float = 1.0

    sigma_init_pose_x: float = 0.05
    sigma_init_pose_y: float = 0.05
    sigma_init_pose_yaw_deg: float = 0.5

    def __post_init__(self) -> None:
        """Validate noise parameters."""
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"Noise parameter {field_name} must be > 0, got {value}")

    @property
    def sigma_steer_rad(self) -> float:
        return np.deg2rad(self.sigma_steer_deg)

    @property
    def sigma_odom_yaw_rad(self) -> float:
        return np.deg2rad(self.sigma_odom_yaw_deg)

    @property
    def sigma_bearing_rad(self) -> float:
        return np.deg2rad(self.sigma_bearing_deg)

    @property
    def sigma_init_pose_yaw_rad(self) -> float:
        return np.deg2rad(self.sigma_init_pose_yaw_deg)

    @property
    def control_input_cov_matrix(self) -> np.ndarray:
        """Return diagonal covariance matrix of raw odometry input noise in [velocity, steering]."""
        return np.diag([
            self.sigma_velocity ** 2,
            self.sigma_steer_rad ** 2,
        ])

    @property
    def odom_cov_matrix(self) -> np.ndarray:
        """Return diagonal covariance matrix of additive odometry output noise in [x, y, yaw]."""
        return np.diag([
            self.sigma_odom_x ** 2,
            self.sigma_odom_y ** 2,
            self.sigma_odom_yaw_rad ** 2,
        ])

    @property
    def range_bearing_cov_matrix(self) -> np.ndarray:
        """Return diagonal landmark observation covariance matrix in [range, bearing]."""
        return np.diag([
            self.sigma_range ** 2,
            self.sigma_bearing_rad ** 2,
        ])

    @property
    def init_pose_cov_matrix(self) -> np.ndarray:
        """Return diagonal initial pose covariance matrix in [x, y, yaw]."""
        return np.diag([
            self.sigma_init_pose_x ** 2,
            self.sigma_init_pose_y ** 2,
            self.sigma_init_pose_yaw_rad ** 2,
        ])


      
@dataclass
class TentativeLandmarkManagerConfig:
    """Configuration for landmark management in SLAM.
    
    Attributes:
        M (int): Hits needed to confirm a landmark.
        N (int): Max lifetime (in steps) of tentative landmark.
        gate (float): Euclidean distance threshold for associating unassociated 
            measurements to existing tentative landmarks. Quick fix to handle 
            dynamic objects.
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

        






