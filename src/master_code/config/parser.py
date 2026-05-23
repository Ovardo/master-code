from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from master_code.config.config import (
    AssociationConfig,
    LoggingConfig,
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
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def resolve_path(filename: str | Path) -> Path:
        path = Path(filename)
        if path.is_absolute() or path.parent != Path("."):
            return path
        return CONFIG_FILES_DIR / path.name

    @classmethod
    def load(cls, filename: str | Path) -> SlamConfig:
        path = cls.resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        default_conf = OmegaConf.structured(SlamConfig)
        loaded_conf = OmegaConf.load(path)
        config = OmegaConf.to_object(
            OmegaConf.merge(default_conf, loaded_conf)
        )
    
        print(f"Loaded configuration from {path}")
        return config # type: ignore

    def save(self, filename: str | Path) -> None:
        path = self.resolve_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(self), path)
        print(f"Configuration saved to {path}")



if __name__ == "__main__":
    
    # Example: Create and save a default configuration
    default_config = SlamConfig()
    default_config.save("default_config.yaml")
    
    # Example: Load configuration from specified YAML file
    loaded_config = SlamConfig.load("default_config.yaml")
    
   
