from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
class SlamConfig:
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    tentative: TentativeLandmarkManagerConfig = field(default_factory=TentativeLandmarkManagerConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)

    @classmethod
    def load(cls, filename: Path) -> SlamConfig:
        path = CONFIG_FILES_DIR / Path(filename).name
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        default_conf = OmegaConf.structured(SlamConfig)
        loaded_conf = OmegaConf.load(path)
        config = OmegaConf.to_object(
            OmegaConf.merge(default_conf, loaded_conf)
        )
    
        print(f"Loaded configuration from {path}")
        return config # type: ignore

    def save(self, filename: str) -> None:
        path = CONFIG_FILES_DIR / Path(filename).name
        OmegaConf.save(OmegaConf.structured(self), path)
        print(f"Configuration saved to {path}")



if __name__ == "__main__":
    
    # Example: Create and save a default configuration
    default_config = SlamConfig()
    default_config.save("default_config.yaml")
    
    # Example: Load configuration from specified YAML file
    loaded_config = SlamConfig.load("default_config.yaml")
    
   
