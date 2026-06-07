"""
SlamLogger — per-step diagnostics, snapshot saving, and offline reloading.

Saves to:
    <run_dir>/
        metadata.json          ← scalar metadata
        reference.npz          ← optional dataset references for plotting
        steps.npz              ← per-step diagnostics (NaN for absent fields)
        associations/
            assoc_00050.npz    ← ragged association diagnostics for selected scans
        snapshots/
            snap_00050.npz     ← full state at step 50
            snap_00100.npz     ← full state at step 100
            snap_final.npz     ← full final state (always written by save())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from master_code.config import LoggingConfig


@dataclass
class AssociationDiagnostics:
    """Raw per-scan association data saved as numeric npz arrays."""
    scan_step: int
    scan_time: float
    pose_index: int
    pose: np.ndarray
    measurements: np.ndarray
    predicted_measurements: np.ndarray
    association: np.ndarray
    local_landmarks: np.ndarray
    local_landmark_keys: np.ndarray
    prior_joint_covariance: np.ndarray
    innovation_covariance: np.ndarray

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "scan_step": np.asarray([self.scan_step], dtype=np.int64),
            "scan_time": np.asarray([self.scan_time], dtype=float),
            "pose_index": np.asarray([self.pose_index], dtype=np.int64),
            "pose": np.asarray(self.pose, dtype=float).reshape(3),
            "measurements": np.asarray(self.measurements, dtype=float).reshape(-1, 2),
            "predicted_measurements": np.asarray(self.predicted_measurements, dtype=float).reshape(-1, 2),
            "association": np.asarray(self.association, dtype=np.int64).reshape(-1),
            "local_landmarks": np.asarray(self.local_landmarks, dtype=float).reshape(-1, 2),
            "local_landmark_keys": np.asarray(self.local_landmark_keys, dtype=np.int64).reshape(-1),
            "prior_joint_covariance": np.asarray(self.prior_joint_covariance, dtype=float),
            "innovation_covariance": np.asarray(self.innovation_covariance, dtype=float),
        }

@dataclass
class StepDiagnostics:
    """Per-step scalar diagnostics for logging and plotting."""
    scan_step: int = -1
    scan_time: float = np.nan
    num_landmarks: int = 0
    num_local_landmarks: int = 0
    num_associated_measurement: int = 0
    num_unassociated_measurement: int = 0
    num_support_cliques: int = 0
    duration_step: float = 0.0
    duration_optimization: float = 0.0
    duration_association: float = 0.0
    duration_covariance_extraction: float = 0.0
    duration_local_landmark_extraction: float = 0.0

    def clear(self) -> None:
        """Reset all diagnostics to their default values."""
        fresh = type(self)()
        for field in fields(self):
            setattr(self, field.name, getattr(fresh, field.name))

    def copy(self) -> StepDiagnostics:
        """Return a detached copy of these diagnostics."""
        return replace(self)

    def add_time(self, name: str, dt: float) -> None:
        """Accumulate timing diagnostics by field name."""
        setattr(self, name, getattr(self, name) + dt)



class SlamLogger:
    def __init__(self, run_dir: Path, config: LoggingConfig):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_dir = self.run_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.association_dir = self.run_dir / "associations"
        self.association_dir.mkdir(parents=True, exist_ok=True)

        self.configure_logging(config)

    def configure_logging(self, config: LoggingConfig) -> None:
        """Configure optional diagnostic logging from the project config."""
        self.log_association_diagnostics = config.log_association_diagnostics
        self.association_stride = config.association_stride
        self.association_steps = {step for step in config.association_steps}

        self.log_snapshot = config.log_snapshot
        self.snapshot_stride = config.snapshot_stride
        self.snapshot_steps = {step for step in config.snapshot_steps}

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def should_save_association_diagnostics(self, scan_step: int) -> bool:
        """Return whether this scan should get a full association diagnostic file."""
        if not self.log_association_diagnostics:
            return False
        if scan_step in self.association_steps:
            return True
        return scan_step % self.association_stride == 0

    def should_save_snapshot(self, scan_step: int) -> bool:
        """Return whether this scan should get a full snapshot."""
        if not self.log_snapshot:
            return False
        if scan_step in self.snapshot_steps:
            return True
        return scan_step % self.snapshot_stride == 0

    def _convert(self, steps_list: list[StepDiagnostics]) -> dict[str, np.ndarray]:
        """ Convert per-step diagnostics to columnar step arrays. """
        steps_dict_arr = {
            field.name: np.asarray([getattr(step, field.name) for step in steps_list])
            for field in fields(StepDiagnostics)
        }
        return steps_dict_arr

    def save_steps_diagnostics(self, steps: list[StepDiagnostics]) -> dict[str, np.ndarray]:
        """ Save per-step diagnostics to a compressed npz file. """
        path = self.run_dir / "steps.npz"
        steps_dict = self._convert(steps)
        np.savez_compressed(path, **steps_dict)
        return steps_dict

    def save_snapshot(self, step: int, snapshot: dict[str, np.ndarray], final: bool = False) -> None:
        """ Write a full-state snapshot to disk"""
        snapshot = dict(snapshot)
        snapshot["step"] = np.asarray([step], dtype=np.int64)

        if final:
            filename = "snap_final.npz"
        else:
            filename = f"snap_{step:05d}.npz"

        path = self.snapshot_dir / filename
        np.savez_compressed(path, **snapshot)

    def save_association_diagnostics(self, diagnostics: AssociationDiagnostics) -> None:
        """Save one scan's raw association data to a compressed npz file."""
        path = self.association_dir / f"assoc_{diagnostics.scan_step:05d}.npz"
        np.savez_compressed(path, **diagnostics.as_npz_dict())

    def save_reference(self, **arrays: np.ndarray) -> None:
        """Save optional dataset reference data for plotting and evaluation."""
        if not arrays:
            return

        path = self.run_dir / "reference.npz"
        np.savez_compressed(path, **arrays)
    
    def save_metadata(
        self,
        diagnostics: StepDiagnostics,
        total_time: float,
        dataset: str,
        algebraic_connectivity: float | None = None,
        verbose: bool = True,
    ) -> None:

        metadata = {
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "dataset":       dataset,
            "num_poses":     diagnostics.scan_step + 1,
            "num_landmarks": diagnostics.num_landmarks,
            "run_time_s":    total_time,
        }

        if algebraic_connectivity is not None:
            metadata["algebraic_connectivity"] = algebraic_connectivity

        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        if verbose:
            lines = [
                "[SlamLogger] Run saved",
                f"  Path        : {self.run_dir.resolve()}",
                f"  Dataset     : {metadata['dataset']}",
                f"  Poses       : {metadata['num_poses']}",
                f"  Landmarks   : {metadata['num_landmarks']}",
                f"  Run time    : {metadata['run_time_s']:.2f}s",
            ]
            if algebraic_connectivity is not None:
                lines.append(f"  Connectivity: {algebraic_connectivity:.4f}")
            print("\n".join(lines))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_npz_dict(path: Path | str) -> dict[str, np.ndarray]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        # Could consider doing lazy loading, returning np.load() instead of dict
        return np.load(path, allow_pickle=False)
     

    @staticmethod
    def load_steps(run_path: Path | str) -> dict[str, np.ndarray]:
        return SlamLogger._load_npz_dict(Path(run_path) / "steps.npz")

    @staticmethod
    def load_metadata(run_path: Path | str) -> dict:
        path = Path(run_path) / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    @staticmethod
    def _resolve_step_path(
        run_path: Path | str,
        step: int,
        directory: str,
        prefix: str,
        description: str,
    ) -> Path:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"step must be an int, got {type(step).__name__}")
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")

        step_dir = Path(run_path) / directory
        path = step_dir / f"{prefix}_{step:05d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No {description} saved for step {step} in {step_dir}"
            )
        return path

    @staticmethod
    def load_snapshot(
        run_path: Path | str,
        step: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Load a snapshot by run directory and step, or directly by file path."""
        if step is None:
            return SlamLogger._load_npz_dict(run_path)

        path = SlamLogger._resolve_step_path(
            run_path,
            step,
            directory="snapshots",
            prefix="snap",
            description="snapshot",
        )
        return SlamLogger._load_npz_dict(path)

    @staticmethod
    def load_association_diagnostics(
        run_path: Path | str,
        step: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Load association diagnostics by run directory and step, or directly by file path."""
        if step is None:
            return SlamLogger._load_npz_dict(run_path)

        path = SlamLogger._resolve_step_path(
            run_path,
            step,
            directory="associations",
            prefix="assoc",
            description="association diagnostics",
        )
        return SlamLogger._load_npz_dict(path)

    @staticmethod
    def load_reference(run_path: Path | str) -> dict[str, np.ndarray]:
        path = Path(run_path) / "reference.npz"
        if not path.exists():
            return {}

        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    @staticmethod
    def load_all_snapshots(run_dir: Path | str) -> list[dict[str, np.ndarray]]:  
        run_dir = Path(run_dir)
        snap_dir = run_dir / "snapshots"
        if not snap_dir.exists():
            raise FileNotFoundError(snap_dir)
        # TODO: handle memory concerns for large number snapshots
        paths = sorted(snap_dir.glob("snap_*.npz"))
        return [SlamLogger.load_snapshot(p) for p in paths]

    @staticmethod
    def load_all_association_diagnostics(run_dir: Path | str) -> list[dict[str, np.ndarray]]:
        run_dir = Path(run_dir)
        assoc_dir = run_dir / "associations"
        if not assoc_dir.exists():
            raise FileNotFoundError(assoc_dir)

        paths = sorted(assoc_dir.glob("assoc_*.npz"))
        return [SlamLogger.load_association_diagnostics(path) for path in paths]
