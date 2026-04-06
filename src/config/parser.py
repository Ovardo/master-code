from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from config.config import (
    AssociationConfig,
    NoiseConfig,
    SensorConfig,
    TentativeLandmarkManagerConfig,
)

CONFIG_FILES_DIR = Path(__file__).resolve().parent / "files"

# -------------------------------
# Main configuration dataclass
# -------------------------------
@dataclass
class SLAMConfig:
    """Main configuration class for SLAM experiment.

    Attributes:

    """
    name: str = "base_config" # TODO: perhaps remove and let the filename be the identifier instead, or when not filename is defined in save_config, use name as filename?
    description: str = "Base config description"

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    tentative: TentativeLandmarkManagerConfig = field(default_factory=TentativeLandmarkManagerConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)



def load_config(config_file: Optional[Path]) -> SLAMConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): YAML file name or path.

    Returns:
        SLAMConfig object with validated parameters.

    """

    default_config = OmegaConf.structured(SLAMConfig)

    if config_file is None:
        config = OmegaConf.to_object(default_config)
        return config  # type: ignore

    path = CONFIG_FILES_DIR / Path(config_file).name
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    yaml_conf = OmegaConf.load(path)
    config = OmegaConf.to_object(
        OmegaConf.merge(
            default_config,
            yaml_conf,
        )
    )

    return config  # type: ignore


def save_config(config: SLAMConfig, filename: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: SLAMConfig object to save
        output_path: Path where YAML file will be saved
    """
    
    path = CONFIG_FILES_DIR / Path(filename).name
    omega_config = OmegaConf.structured(config)
    OmegaConf.save(omega_config, path)
    print(f"Configuration saved to {path}")


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

    # Load override config
    override_yaml = OmegaConf.load(override_config_path)
    
    # Merge with base config
    base_omega = OmegaConf.structured(base_config)
    merged = OmegaConf.merge(base_omega, override_yaml)
    
    # Convert back to dataclass
    return OmegaConf.to_object(merged) # type: ignore



if __name__ == "__main__":
    
    # Example: Create and save a default configuration
    config = SLAMConfig(name="Hellow World")
    
    print(config)
    print("\n" + "="*50 + "\n")
    
    # Save to YAML
    filename = "default_config.yaml"
    save_config(config, filename)
    
    # Load it back
    loaded_config = load_config(filename)
    print(f"Loaded config \"{loaded_config.name}\" from {filename}.")
   
