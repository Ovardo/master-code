"""
SlamLogger — integrated logging, snapshot saving, and offline reloading.

Saves to:
    <run_dir>/
        metadata.json                  ← scalar metadata
        step_data.npz                  ← per-step timing and counts
        snapshots/
            step_000050.npz            ← full state at step 50
            step_000100.npz            ← full state at step 100
            ...
            step_final.npz             ← final state (always written on save())

Usage (during a run):
    logger = SlamLogger(Path("results") / run_name, snapshot_every=50)
    slam   = FactorGraphSLAM(cfg=config, logger=logger)
    ...run slam...
    logger.save(slam.get_snapshot())

Usage (offline plotting):
    data      = SlamLogger.load(Path("results/xxx.yaml"))
    snapshots = SlamLogger.load_snapshots(Path("results/xxx.yaml"))
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


class SlamLogger:
    """
    Receives per-step callbacks from FactorGraphSLAM and persists them.

    Parameters
    ----------
    run_dir:
        Root directory for this run's output (created on construction).
    snapshot_every:
        How often (in scan steps) to write a full-state snapshot.
        Set to 0 or None to disable intermediate snapshots (only the
        final snapshot written by save() will exist).
    """

    def __init__(self, run_dir: Path, snapshot_every: int = 50):
        self.run_dir        = Path(run_dir)
        self.snapshot_dir   = self.run_dir / "snapshots"
        self.snapshot_every = snapshot_every or 0

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Per-step accumulators  {metric_name -> list of values}
        self._steps:  list[int]              = []
        self._times:  dict[str, list[float]] = {}
        self._counts: dict[str, list[int | float]] = {}
        self._values: dict[str, list]        = {}

        # Metrics collected during the current SLAM update.
        self._pending_times:  dict[str, float] = {}
        self._pending_counts: dict[str, int]   = {}
        self._pending_values: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Callbacks — called by FactorGraphSLAM
    # ------------------------------------------------------------------

    def log_time(self, name: str, value: float, accumulate: bool = False) -> None:
        """Record a timing metric for the current step."""
        value = float(value)
        if accumulate:
            self._pending_times[name] = self._pending_times.get(name, 0.0) + value
        else:
            self._pending_times[name] = value

    def log_count(self, name: str, value: int) -> None:
        """Record an integer count metric for the current step."""
        self._pending_counts[name] = int(value)

    def log_value(self, name: str, value) -> None:
        """Record a generic per-step metric for the current step."""
        self._pending_values[name] = value

    def flush_step(
        self,
        step: int,
        times: dict[str, float] | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        """
        Flush pending diagnostics into one per-step record.

        ``times`` and ``counts`` are accepted for backwards compatibility with
        the old call style, but new code should use ``log_time`` and
        ``log_count`` at the measurement site.
        """
        if times is not None:
            for name, value in times.items():
                self.log_time(name, value)
        if counts is not None:
            for name, value in counts.items():
                self.log_count(name, value)

        previous_steps = len(self._steps)
        self._steps.append(step)

        self._flush_pending_group(
            destination=self._times,
            pending=self._pending_times,
            previous_steps=previous_steps,
            missing_value=0.0,
        )
        self._flush_pending_group(
            destination=self._counts,
            pending=self._pending_counts,
            previous_steps=previous_steps,
            missing_value=0.0,
        )
        self._flush_pending_group(
            destination=self._values,
            pending=self._pending_values,
            previous_steps=previous_steps,
            missing_value=0.0,
        )

    @staticmethod
    def _flush_pending_group(
        destination: dict[str, list],
        pending: dict[str, object],
        previous_steps: int,
        missing_value: object,
    ) -> None:
        """Append one step of pending metrics, padding absent columns."""
        for name in pending:
            if name not in destination:
                destination[name] = [missing_value] * previous_steps

        for name, values in destination.items():
            values.append(pending.get(name, missing_value))

        pending.clear()

    def log_snapshot(self, step: int, snapshot: dict[str, np.ndarray]) -> None:
        """
        Write a full-state snapshot to disk immediately.

        Called automatically by FactorGraphSLAM every ``snapshot_every`` steps.
        Can also be called manually at any time, passing ``slam.get_snapshot()``.
        """
        path = self.snapshot_dir / f"step_{step:06d}.npz"
        np.savez_compressed(path, step=np.array(step), **snapshot)

    def maybe_log_snapshot(
        self,
        step: int,
        snapshot_fn: Callable[[], dict[str, np.ndarray]],
    ) -> None:
        """Write a snapshot when ``snapshot_every`` says this step is due."""
        if self.snapshot_every > 0 and step % self.snapshot_every == 0:
            self.log_snapshot(step, snapshot_fn())

    # ------------------------------------------------------------------
    # Final save — call once after the run is complete
    # ------------------------------------------------------------------

    def save(self, snapshot: dict[str, np.ndarray], error: float) -> Path:
        """
        Flush all accumulated step data and write the final snapshot.

        Parameters
        ----------
        snapshot:
            The dict returned by ``slam.get_snapshot()``.
        error:
            float returned by ``slam.get_error()``

        Returns
        -------
        Path
            The run directory.
        """
        # ── step-level data ───────────────────────────────────────────
        step_arrays: dict[str, np.ndarray] = {
            "steps": np.asarray(self._steps, dtype=int),
        }
        for k, v in self._times.items():
            step_arrays[f"time_{k}"] = np.asarray(v, dtype=float)
        for k, v in self._counts.items():
            arr = np.asarray(v)
            if arr.dtype.kind == "f" and np.isnan(arr).any():
                step_arrays[f"count_{k}"] = arr.astype(float)
            else:
                step_arrays[f"count_{k}"] = arr.astype(int)
        for k, v in self._values.items():
            step_arrays[f"value_{k}"] = np.asarray(v)

        np.savez_compressed(self.run_dir / "step_data.npz", **step_arrays)

        # ── final snapshot — written into snapshots/ alongside the others ──
        final_step = len(snapshot["poses"])
        np.savez_compressed(
            self.snapshot_dir / "step_final.npz",   # ← same dir as periodic snapshots
            step=np.array(final_step),
            **snapshot,
        )

        # ── metadata ───────────────────────────────────────────────────
        total_time = np.nansum(self._times.get("total", [0.0]))
        metadata = {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "snapshot_every": self.snapshot_every,
            "num_poses":      int(len(snapshot["poses"])),
            "num_landmarks":  int(len(snapshot["landmarks"])),
            "total_error":    error,
            "total_time_s":   round(total_time, 3),
            
        }
        (self.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"[SlamLogger] Saved → {self.run_dir.resolve()}")
        print(f"             poses={metadata['num_poses']}  "
              f"landmarks={metadata['num_landmarks']}  "
              f"total_time={metadata['total_time_s']:.1f}s")
        return self.run_dir

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load(run_dir: Path) -> dict:
        """
        Load step-level data and metadata for a saved run.

        Returns a flat dict whose keys include:
            ``steps``, 
            ``scan_time``, 
            ``time_total``, 
            ``time_association``,
            ``count_local_landmarks``, 
            ``count_total_landmarks``,
            ...
            ``metadata``

        For full-state arrays (poses, landmarks, covariances) load a
        snapshot::

            snap = SlamLogger.load_snapshot(run_dir / "snapshots" / "step_final.npz")
        """
        run_dir  = Path(run_dir)
        npz_path = run_dir / "step_data.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No step_data.npz in {run_dir}")

        archive = np.load(npz_path, allow_pickle=False)
        data    = {k: archive[k] for k in archive.files}

        meta_path = run_dir / "metadata.json"
        data["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        return data

    @staticmethod
    def load_snapshot(path: Path) -> dict:
        """
        Load a single snapshot file.

        Returns a dict with keys:
            ``step``, ``poses``, ``poses_covariance``, ``landmarks``,
            ``landmarks_covariance``
        """
        archive = np.load(Path(path), allow_pickle=False)
        return {k: archive[k] for k in archive.files}

    @staticmethod
    def load_snapshots(run_dir: Path, include_final: bool = True) -> list[dict]:
        """
        Load all snapshots from a run, sorted by step number.

        Parameters
        ----------
        run_dir:
            The top-level run directory (parent of ``snapshots/``).
        include_final:
            Whether to append ``step_final.npz`` after the numbered snapshots.

        Returns
        -------
        list[dict]
            Each element is the dict returned by ``load_snapshot()``.
        """
        snap_dir = Path(run_dir) / "snapshots"
        if not snap_dir.exists():
            return []

        # Numbered periodic snapshots, sorted ascending
        paths = sorted(snap_dir.glob("step_[0-9]*.npz"))

        # Final snapshot appended last (not glob'd to preserve ordering)
        final_path = snap_dir / "step_final.npz"
        if include_final and final_path.exists():
            paths.append(final_path)

        return [SlamLogger.load_snapshot(p) for p in paths]
