"""
Code for converting yaml configuration files into structured dataclasses utlizing OmegaConf.
Convenient for type checking, data validation, and IDE support
"""
import numpy as np
import warnings

from dataclasses import dataclass, field
from typing import Optional, Tuple 
from pathlib import Path


@dataclass
class NoiseConfig:
    """Noise paramters for simulation and inference."""

    # Process noise (odometry)
    x_std: float = 0.1 # meters
    y_std: float = 0.1 # meters
    theta_std_deg: float = 0.1 # degrees
    
    # Measurement noise 
    bearing_std_deg: float = 1.0  # degrees
    range_std: float = 0.2 # meters
    
    # Initial state uncertainty
    x0_std: float = 0.05 # meters
    y0_std: float = 0.05 # meters
    theta0_std_deg: float = 3.0  # degrees

    def __post___init__(self):
        """Validate noise parameters"""
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
    def odometry_vec(self):
        """Return process noise vector."""
        return np.array([self.x_std, self.y_std, self.theta_std_rad], dtype=np.float64)
    
    @property
    def measurement_vec(self):
        """Return measurement noise vector."""
        return np.array([self.bearing_std_rad, self.range_std], dtype=np.float64)
    
    @property
    def initial_vec(self):
        """Return initial state noise vector."""
        return np.array([self.x0_std, self.y0_std, self.theta0_std_rad], dtype=np.float64)
    


@dataclass
class SimulationConfig:
    """Simulation paramters."""
    
    # Time settings
    dt: float = 0.1  # time step in seconds
    duration: float = 20.0  # total simulation duration in seconds
    
    # Robot initial pose
    init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, theta) in meters and radians

    # Noise configuration
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # Trajectory settings
    path_type: str = 'circle'
    num_poses: int = 10

    # Landmark settings
    num_landmarks: int = 15
    landmark_bounds: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0) # (x_min, x_max, y_min, y_max)

    # Sensor settings
    max_sensor_range: float = 15.0   # meters
    sensor_fov_deg: float = 360.0  # degrees

    def __post_init__(self):
        """Validate simulation parameters."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.num_poses <= 0:
            raise ValueError(f"num_poses must be positive, got {self.num_poses}")
        if self.num_landmarks <= 0:
            raise ValueError(f"num_landmarks must be positive, got {self.num_landmarks}")
        if self.max_sensor_range <= 0:
            raise ValueError(f"max_sensor_range must be positive, got {self.max_sensor_range}")
        if not (0 < self.sensor_fov_deg <= 360):
            raise ValueError(f"sensor_fov_deg must be in (0, 360], got {self.sensor_fov_deg}")
      

@dataclass
class InferenceConfig:
    """Inference parameters."""
    
    # Algorithm selection
    algorithm: str = "isam2"  # Options: "ekf", "isam2", "batch"

    # EKF-specific settings
    
    # Noise model (can differ from simulation)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dead_reckoning: bool = False  # whether to use dead reckoning only (no landmark factors) # TODO: IDE hover support
    
    # Data association settings
    association_type: str = "jcbb"  # "ground_truth", "jcbb", "maximum_likelihood", "neareast_neighbour", "Constrained Nearnest Neighboor data association
    alpha_individual: float = 0.999  # confidence levels for individual compatibility test
    alpha_joint: float = 0.9999  # confidence levels for joint compatibility test

    local_filtering_range: float = 30  # local feature filtering radius for data association (should be larger than lidar range)
    sensor_offset: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) offset of sensor wrt robot body frame NOTE: not used

    def __post_init__(self):
        """Validate inference parameters."""
        
        algorithm_options = ["ekf", "isam2", "batch"]
        if self.algorithm not in algorithm_options:
            raise ValueError(f"Invalid algorithm {self.algorithm}, must be one of 'ekf', 'isam2', 'batch'")
        
        association_options = ["ground_truth", "jcbb", "maximum_likelihood"]
        if self.association_type not in association_options:
            raise ValueError(f"Invalid association_type {self.association_type}, must be one of {association_options}")
       
        if not (0 < self.alpha_individual < 1):
            raise ValueError(f"alpha_individual must be in (0, 1), got {self.alpha_individual}")
        if not (0 < self.alpha_joint < 1):
            raise ValueError(f"alpha_joint must be in (0, 1), got {self.alpha_joint}")
        if self.local_filtering_range <= 0:
            raise ValueError(f"local_filtering_range must be positive, got {self.local_filtering_range}")
        

@dataclass
class VisualizationConfig:
    """Visualization and plotting settings."""
    
    enabled: bool = True
    real_time: bool = True  # Update plot during simulation
    update_interval: int = 10  # Update every N steps
    
    # Plot settings
    figure_size: Tuple[int, int] = (12, 8)
    show_trajectory: bool = True
    show_landmarks: bool = True
    show_uncertainty: bool = True
    show_observations: bool = False
    
    # Save settings
    save_plots: bool = False
    output_dir: str = "./results"
    save_format: str = "png"  # png, pdf, svg
    
    def __post_init__(self):
        """Validate visualization parameters."""
        if self.update_interval <= 0:
            raise ValueError(f"update_interval must be > 0, got {self.update_interval}")
        
        valid_formats = ["png", "pdf", "svg"]
        if self.save_format not in valid_formats:
            raise ValueError(f"save_format must be one of {valid_formats}, got {self.save_format}")



# -------------------------------
# Main configuration dataclass
# -------------------------------
@dataclass
class SLAMConfig:
    """Main configuration class for SLAM experiment."""
    
    # Experiment metadata
    name: str = "default_experiment"
    description: str = "SLAM experiment configuration"
    seed: Optional[int] = 42  # random seed for reproducibility

    # Sub-configurations
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        """Validate top-level configuration."""
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative or None, got {self.seed}")
        if self.inference.local_filtering_range < self.simulation.max_sensor_range:
            warnings.warn(f"local_filtering_range should be larger than max_sensor_range for effective data association, got {self.inference.local_filtering_range} and {self.simulation.max_sensor_range}")

    def summary(self) -> str: # TODO: adjust
        """Return a human-readable summary of the configuration."""
        lines = [
            f"Experiment Name: {self.name}",
            f"Description: {self.description}",
            f"Random Seed: {self.seed}",
            "Simulation Config:",
            f"  Time Step (dt): {self.simulation.dt} s",
            f"  Duration: {self.simulation.duration} s",
            f"  Initial Pose: {self.simulation.init_pose}",
            f"  Path Type: {self.simulation.path_type}",
            f"  Number of Poses: {self.simulation.num_poses}",
            f"  Number of Landmarks: {self.simulation.num_landmarks}",
            f"  Landmark Bounds: {self.simulation.landmark_bounds}",
            f"  Max Sensor Range: {self.simulation.max_sensor_range} m",
            f"  Sensor FOV: {self.simulation.sensor_fov_deg} degrees",
            "Inference Config:",
            f"  Algorithm: {self.inference.algorithm}",
            f"  Data Association Type: {self.inference.association_type}",
            f"  Alpha Individual: {self.inference.alpha_individual}",
            f"  Alpha Joint: {self.inference.alpha_joint}",
            f"  Local Filtering Range: {self.inference.local_filtering_range} m",
            "Visualization Config:",
            f"  Enabled: {self.visualization.enabled}",
            f"  Real-time: {self.visualization.real_time}",
            f"  Update Interval: {self.visualization.update_interval} steps",
            f"  Figure Size: {self.visualization.figure_size}",
            f"  Show Trajectory: {self.visualization.show_trajectory}",
            f"  Show Landmarks: {self.visualization.show_landmarks}",
            f"  Show Uncertainty: {self.visualization.show_uncertainty}",
            f"  Show Observations: {self.visualization.show_observations}",
            f"  Save Plots: {self.visualization.save_plots}",
            f"  Output Directory: {self.visualization.output_dir}",
            f"  Save Format: {self.visualization.save_format}",
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
            yaml_conf # Override with YAML values
        )
    )

    return config

def save_config(config: SLAMConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: SLAMConfig object to save
        output_path: Path where YAML file will be saved
    """
    from omegaconf import OmegaConf
    
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
    return OmegaConf.to_object(merged)



if __name__ == "__main__":
    # Example: Create and save a default configuration
    default_config = SLAMConfig(
        name="example_config",
        description="Example SLAM configuration for testing"
    )
    
    print(default_config.summary())
    print("\n" + "="*50 + "\n")
    
    # Save to YAML
    save_config(default_config, "src/conf/sandbox/default_config.yaml")
    
    # Load it back
    loaded_config = load_config("src/conf/sandbox/default_config.yaml")
    print("Successfully loaded configuration!")
    print(f"Algorithm: {loaded_config.inference.algorithm}")
    print(f"Simulation dt: {loaded_config.simulation.dt}")




