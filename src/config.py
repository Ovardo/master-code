"""
Code for converting yaml configuration files into structured dataclasses utlizing OmegaConf.

Convenient for type checking, data validation, and IDE support.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import OmegaConf


@dataclass
class NoiseConfig:
    """Noise paramters.
    
    Attributes:
        x_std (float): Standard deviation of process noise in x [m].
        y_std (float): Standard deviation of process noise in y [m].
        theta_std_deg (float): Standard deviation of process noise in orientation [degrees].
        bearing_std_deg (float): Standard deviation of measurement noise in bearing [degrees].
        range_std (float): Standard deviation of measurement noise in range [m].
        x0_std (float): Standard deviation of initial x position [m].
        y0_std (float): Standard deviation of initial y position [m].
        theta0_std_deg (float): Standard deviation of initial orientation [degrees].
    """


    x_std: float = 0.1
    y_std: float = 0.1
    theta_std_deg: float = 0.1
    bearing_std_deg: float = 1.0
    range_std: float = 0.2
    x0_std: float = 0.05
    y0_std: float = 0.05
    theta0_std_deg: float = 1.0
 
    def __post_init__(self):
        """Validate noise parameters."""
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"Noise parameter {field_name} must be > 0, got {value}")

    @property
    def theta_std_rad(self):
        """Convert theta_std from degrees to radians."""
        return np.deg2rad(self.theta_std_deg)

    @property
    def theta0_std_rad(self):
        """Convert theta0_std from degrees to radians."""
        return np.deg2rad(self.theta0_std_deg)

    @property
    def bearing_std_rad(self):
        """Convert bearing_std from degrees to radians."""
        return np.deg2rad(self.bearing_std_deg)

    @property
    def odometry_std(self):
        """Return process noise vector (3,) [x, y, theta]."""
        return np.array([self.x_std, self.y_std, self.theta_std_rad], dtype=np.float64)

    @property
    def measurement_std(self):
        """Return measurement noise vector (2,) [bearing, range]."""
        return np.array([self.bearing_std_rad, self.range_std], dtype=np.float64)

    @property
    def prior_std(self):
        """Return initial state noise vector (3,) [x0, y0, theta0]."""
        return np.array([self.x0_std, self.y0_std, self.theta0_std_rad], dtype=np.float64)

      
@dataclass
class LandmarkManagerConfig:
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
    range_gate: float = 50.0
    fov_gate_deg: float = 250.0

    def __post_init__(self):
        methods_options = ["jcbb", "ml"] # TODO: include "gt" and "nn"
        if self.method not in methods_options:
            raise ValueError(f"Invalid association_type {self.method}, must be one of {methods_options}")
        if not (0 < self.alpha_individual < 1):
            raise ValueError(f"alpha_individual must be in (0, 1), got {self.alpha_individual}")
        if not (0 < self.alpha_joint < 1):
            raise ValueError(f"alpha_joint must be in (0, 1), got {self.alpha_joint}")
        if self.range_gate <= 0:
            raise ValueError(f"local_filtering_range must be positive, got {self.range_gate}")
        if not (0 <= self.fov_gate_deg <= 360):
            raise ValueError(f"fov_gate_deg must be in [0, 360], got {self.fov_gate_deg}")
        

@dataclass
class InferenceConfig:
    """Inference parameters.
    
    Attributes:
        algorithm (str): The inference algorithm to use. Must be one of 'ekf', 'isam2', or 'batch'.
            Defaults to 'isam2'.
        prior_pose (tuple[float, float, float]): Initial pose of the robot as (x, y, theta).
            Defaults to (0.0, 0.0, 0.0).
        noise (NoiseConfig): Configuration for noise parameters.
        landmark_manager (LandmarkManagerConfig): Configuration for landmark management.
        association (AssociationConfig): Configuration for data association.
    """
    algorithm: str = 'isam2' 
    prior_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    landmark_manager: LandmarkManagerConfig = field(default_factory=LandmarkManagerConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)

    def __post_init__(self):
        """Validate inference parameters."""
        algorithm_options = ["ekf", "isam2", "batch"]
        if self.algorithm not in algorithm_options:
            raise ValueError(f"Invalid algorithm {self.algorithm}, must be one of 'ekf', 'isam2', 'batch'")
        

@dataclass
class VisualizationConfig:
    """Visualization and plotting settings.
    
    Attributes:
        output_dir (str): Directory path for saving output files. Defaults to "./results".
        save_format (str): Output file format for saved plots. Defaults to "svg".
            Must be one of 'png', 'pdf', or 'svg'.
    """

    output_dir: str = "./results"
    save_format: str = "svg"  # png, pdf, svg


@dataclass
class SimulationConfig:
    """Simulation parameters.
    
    Configuration for robot simulation including timing, trajectory, landmarks, and sensor settings.
    
    Attributes:
        dt (float): Time step in seconds. Must be positive.
        duration (float): Total simulation duration in seconds. Must be positive.
        init_pose (tuple[float, float, float]): Robot initial pose (x, y, theta). Units: [m, m, deg].
        noise (NoiseConfig): Noise configuration for simulation.
        path_type (str): Type of trajectory path.
        num_poses (int): Number of poses in trajectory. Must be positive.
        num_landmarks (int): Number of landmarks to simulate. Must be positive.
        landmark_bounds (tuple[float, float, float, float]): Landmark spatial bounds (x_min, x_max, y_min, y_max).
        max_sensor_range (float): Maximum sensor range in meters. Must be positive.
        sensor_fov_deg (float): Sensor field of view in degrees. Must be in (0, 360].
    """

    # Time settings
    dt: float = 0.1 # time step in seconds
    duration: float = 20.0 # total simulation duration in seconds
    init_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    path_type: str = "circle"
    num_poses: int = 10
    num_landmarks: int = 15
    landmark_bounds: tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0)
    max_sensor_range: float = 15.0  # meters
    sensor_fov_deg: float = 360.0  # degrees

    def __post_init__(self):
        """Validate simulation parameters."""
        if self.dt <= 0:
            msg = f"dt must be positive, got {self.dt}"
            raise ValueError(msg)
        if self.duration <= 0:
            msg = f"duration must be positive, got {self.duration}"
            raise ValueError(msg)
        if self.num_poses <= 0:
            msg = f"num_poses must be positive, got {self.num_poses}"
            raise ValueError(msg)
        if self.num_landmarks <= 0:
            msg = f"num_landmarks must be positive, got {self.num_landmarks}"
            raise ValueError(msg)
        if self.max_sensor_range <= 0:
            msg = f"max_sensor_range must be positive, got {self.max_sensor_range}"
            raise ValueError(msg)
        if not (0 < self.sensor_fov_deg <= 360):
            msg = f"sensor_fov_deg must be in (0, 360], got {self.sensor_fov_deg}"
            raise ValueError(msg)



# -------------------------------
# Main configuration dataclass
# -------------------------------
@dataclass
class SLAMConfig:
    """Main configuration class for SLAM experiment.

    Attributes:
        name: Experiment name identifier. Defaults to "default_experiment".
        description: Experiment description. Defaults to "SLAM experiment configuration".
        seed: Random seed for reproducibility. Defaults to 42. None for non-deterministic.
        profilinfg_enabled: Whether to enable timing profiling. Defaults to False.
        simulation: Simulation configuration parameters.
        inference: Inference configuration parameters.
        visualization: Visualization configuration parameters.
    """
    
    # Experiment metadata
    name: str = "default_experiment"
    description: str = "SLAM experiment configuration"
    seed: Optional[int] = 42 # random seed for reproducibility
    profilinfg_enabled: bool = False  # whether to enable timing profiling
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        """Validate top-level configuration."""
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative or None, got {self.seed}")
        if self.inference.association.range_gate < self.simulation.max_sensor_range:
            warnings.warn(
                "range_gate should be larger than max_sensor_range for effective "
                f"data association, got {self.inference.association.range_gate} "
                f"and {self.simulation.max_sensor_range}"
            )

    def summary(self) -> str: # TODO: adjust
        """Return a human-readable summary of the configuration."""
        lines = [
            f"SLAM Configuration: {self.name}"
            f"Description: {self.description}",
            f"Algorithm: {self.inference.algorithm}",
            f"Association method: {self.inference.association.method}",
        ]
        return "\n".join(lines)



def load_config(config_path: str) -> SLAMConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        SLAMConfig object with validated parameters.

    """
    from omegaconf import OmegaConf

    # Check if file exists
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    yaml_conf = OmegaConf.load(config_path)

    # Convert to structured config (validates and runs __post_init__)
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(SLAMConfig), # Default values
            yaml_conf, # Override with YAML values
        )
    )

    return config  # type: ignore


def save_config(config: SLAMConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: SLAMConfig object to save
        output_path: Path where YAML file will be saved
    """
    
    
    # Convert to OmegaConf and save
    omega_config = OmegaConf.structured(config)
    OmegaConf.save(omega_config, output_path)
    print(f"Configuration saved to {output_path}")


def merge_configs(base_config: SLAMConfig, override_config_path: str) -> SLAMConfig:
    """
    Merge a base configuration with overrides from a YAML file.
    
    Useful for having a default config and scenario-specific overrides.
    
    Args:
        base_config: Base SLAMConfig object
        override_config_path: Path to YAML file with overrides
        
    Returns:
        Merged SLAMConfig object
    """
    from omegaconf import OmegaConf

    # Load override config
    override_yaml = OmegaConf.load(override_config_path)
    
    # Merge with base config
    base_omega = OmegaConf.structured(base_config)
    merged = OmegaConf.merge(base_omega, override_yaml)
    
    # Convert back to dataclass
    return OmegaConf.to_object(merged) # type: ignore



if __name__ == "__main__":
    
    # Example: Create and save a default configuration
    default_config = SLAMConfig(
        name="example_config",
        description="Example SLAM configuration for testing"
    )
    
    print(default_config.summary())
    print("\n" + "="*50 + "\n")
    
    # Save to YAML
    save_config(default_config, "src/conf/default_config.yaml")
    
    # Load it back
    loaded_config = load_config("src/conf/default_config.yaml")
    print("Successfully loaded configuration!")
    print(f"Name: {loaded_config.name}")
    print(f"Algorithm: {loaded_config.inference.algorithm}")
   


