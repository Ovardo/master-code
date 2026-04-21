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
    slam   = FactorGraphSLAM(cfg=config, pose0=pose0, logger=logger)
    ...run slam...
    logger.save(slam.get_snapshot())

Usage (offline plotting):
    data      = SlamLogger.load(Path("results/vp1_20240417_120000"))
    snapshots = SlamLogger.load_snapshots(Path("results/vp1_20240417_120000"))
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

        # Per-step accumulators  {metric_name → list of values}
        self._steps:  list[int]              = []
        self._times:  dict[str, list[float]] = defaultdict(list)
        self._counts: dict[str, list[int]]   = defaultdict(list)

    # ------------------------------------------------------------------
    # Callbacks — called by FactorGraphSLAM
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        times: dict[str, float],
        counts: dict[str, int],
    ) -> None:
        """
        Record per-step diagnostics.  Called once per scan step.

        Parameters
        ----------
        step:
            Current pose index (``slam._n_poses - 1``).
        times:
            Timing measurements in seconds, e.g.::

                {
                    "total":                 0.042,
                    "covariance_extraction": 0.011,
                    "association":           0.008,
                    "optimization":          0.018,
                }

        counts:
            Integer diagnostics, e.g.::

                {
                    "local_landmarks": 12,
                    "total_landmarks": 47,
                }
        """
        self._steps.append(step)
        for k, v in times.items():
            self._times[k].append(float(v))
        for k, v in counts.items():
            self._counts[k].append(int(v))

    def log_snapshot(self, step: int, snapshot: dict[str, np.ndarray]) -> None:
        """
        Write a full-state snapshot to disk immediately.

        Called automatically by FactorGraphSLAM every ``snapshot_every`` steps.
        Can also be called manually at any time, passing ``slam.get_snapshot()``.
        """
        path = self.snapshot_dir / f"step_{step:06d}.npz"
        np.savez_compressed(path, step=np.array(step), **snapshot)

    # ------------------------------------------------------------------
    # Final save — call once after the run is complete
    # ------------------------------------------------------------------

    def save(self, snapshot: dict[str, np.ndarray]) -> Path:
        """
        Flush all accumulated step data and write the final snapshot.

        Parameters
        ----------
        snapshot:
            The dict returned by ``slam.get_snapshot()``.

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
            step_arrays[f"count_{k}"] = np.asarray(v, dtype=int)

        np.savez_compressed(self.run_dir / "step_data.npz", **step_arrays)

        # ── final snapshot — written into snapshots/ alongside the others ──
        final_step = len(snapshot["poses"]) - 1
        np.savez_compressed(
            self.snapshot_dir / "step_final.npz",   # ← same dir as periodic snapshots
            step=np.array(final_step),
            **snapshot,
        )

        # ── metadata ───────────────────────────────────────────────────
        total_time = sum(self._times.get("total", [0.0]))
        metadata = {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "snapshot_every": self.snapshot_every,
            "num_poses":      int(len(snapshot["poses"])),
            "num_landmarks":  int(len(snapshot["landmarks"])),
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
            ``steps``, ``time_total``, ``time_association``,
            ``count_local_landmarks``, ``count_total_landmarks``,
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
            ``landmarks_covariance``, ``poses_dr``
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