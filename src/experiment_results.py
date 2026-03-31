from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import SLAMConfig
from history import SLAMHistory
from timing_profiler import TimingProfiler


CURRENT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class ExperimentReferenceData:
    """Optional run-level reference data stored alongside an experiment."""

    ground_truth_poses: np.ndarray | None = None
    ground_truth_landmarks: np.ndarray | None = None
    gps_track: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentResult:
    """Versioned experiment artifact for later analysis and visualization."""

    created_at_utc: str
    config: SLAMConfig
    history: SLAMHistory
    profiler: TimingProfiler | None = None
    reference_data: ExperimentReferenceData | None = None
    schema_version: int = CURRENT_SCHEMA_VERSION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_created_at(timestamp: datetime) -> str:
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_filename_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def _sanitize_config_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "experiment"


def _default_result_path(config: SLAMConfig, timestamp: datetime) -> Path:
    base_dir = Path(config.visualization.output_dir) / "experiments"
    filename = f"{_format_filename_timestamp(timestamp)}_{_sanitize_config_name(config.name)}.pkl"
    return base_dir / filename


def save_result(
    config: SLAMConfig,
    history: SLAMHistory,
    profiler: TimingProfiler | None = None,
    reference_data: ExperimentReferenceData | None = None,
    path: str | Path | None = None,
) -> Path:
    """Serialize a SLAM experiment bundle to disk."""

    timestamp = _utc_now()
    artifact = ExperimentResult(
        created_at_utc=_format_created_at(timestamp),
        config=config,
        history=history,
        profiler=profiler if (profiler is not None and profiler.enabled) else None,
        reference_data=reference_data,
    )

    artifact_path = Path(path) if path is not None else _default_result_path(config, timestamp)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return artifact_path


def load_result(path: str | Path) -> ExperimentResult:
    """Load and validate a previously saved experiment bundle."""

    artifact_path = Path(path)

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, ExperimentResult):
        raise TypeError(
            f"Expected {ExperimentResult.__name__} payload, got {type(payload).__name__}."
        )

    if payload.schema_version != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported experiment result schema_version={payload.schema_version}. "
            f"Expected {CURRENT_SCHEMA_VERSION}."
        )

    return payload


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ExperimentReferenceData",
    "ExperimentResult",
    "load_result",
    "save_result",
]
