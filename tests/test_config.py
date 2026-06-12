from pathlib import Path

import pytest

from master_code.config import NoiseConfig, SlamConfig


def test_load_root_config() -> None:
    config = SlamConfig.load(Path("configs/default_sim.yaml"))

    assert config.association.method == "jcbb"
    assert config.tentative.M == 1
    assert config.tentative.N == 1


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    expected = SlamConfig()

    expected.save(path)
    actual = SlamConfig.load(path)

    assert actual == expected


def test_default_config_matches_dataclass_defaults() -> None:
    assert SlamConfig.load(Path("configs/default_config.yaml")) == SlamConfig()


def test_noise_parameters_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        NoiseConfig(sigma_range=0.0)
