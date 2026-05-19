"""
SlamLogger — per-step diagnostics, snapshot saving, and offline reloading.

Saves to:
    <run_dir>/
        metadata.json          ← scalar metadata
        steps.npz              ← per-step diagnostics (NaN for absent fields)
        snapshots/
            snap_00050.npz     ← full state at step 50
            snap_00100.npz     ← full state at step 100
            snap_final.npz     ← final state (always written by save())
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

class SlamLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_dir = self.run_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def convert_records_to_steps(self, records: list[dict[str, object]]) -> dict[str, np.ndarray]:
        """ Convert append-friendly per-step records to columnar step arrays. """
        all_keys = {k for record in records for k in record}
        steps = {
            key: np.asarray([record.get(key, np.nan) for record in records])
            for key in all_keys
        }
        return steps


    def save_steps(self, steps: dict[str, np.ndarray]) -> None:
        """ Save per-step diagnostics to a compressed npz file. """
        path = self.run_dir / "steps.npz"
        np.savez_compressed(path, **steps)
        

    def save_snapshot(self, step: int, snapshot: dict[str, np.ndarray], final: bool = False) -> None:
        """ Write a full-state snapshot to disk"""
        snapshot = dict(snapshot)
        snapshot["step"] = np.array([step]) # Include step in snapshot for easier loading and debugging

        if final:
            filename = "snap_final.npz"
        else:
            filename = f"snap_{step:05d}.npz"
        path = self.snapshot_dir / filename
        np.savez_compressed(path, **snapshot)
    
    
    def save_metadata(self, steps: dict[str, np.ndarray], snapshot: dict[str, np.ndarray], verbose: bool = True) -> None:
                
        metadata = {
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "num_poses":     int(len(snapshot["poses"])),
            "num_landmarks": int(len(snapshot["landmarks"])),
            "fg_error":      steps["fg_error"][-1],
            "total_time_s":  round(float(np.nansum(steps["t_update"])), 3),
        }
        
        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        if verbose:
            print(
                "\n".join(
                    [
                        "[SlamLogger] Run saved",
                        f"  Path        : {self.run_dir.resolve()}",
                        f"  Poses       : {metadata['num_poses']}",
                        f"  Landmarks   : {metadata['num_landmarks']}",
                        f"  Final error : {metadata['fg_error']:.3f}",
                        f"  Total time  : {metadata['total_time_s']:.1f}s",
                    ]
                )
            )


    def save(self, 
             step: int,
             steps: dict[str, np.ndarray],
             snapshot: dict[str, np.ndarray], 
             verbose=True) -> None:
        
        self.save_steps(steps)
        self.save_snapshot(step, snapshot, final=True)
        self.save_metadata(steps, snapshot, verbose=verbose)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_steps(run_dir: Path | str) -> dict[str, np.ndarray]:
        run_dir = Path(run_dir)
        records_path = run_dir / "steps.npz"
        if not records_path.exists():
            raise FileNotFoundError(f"No steps.npz in {run_dir}")

        archive = np.load(records_path, allow_pickle=False)
        return {k: archive[k] for k in archive.files}

    @staticmethod
    def load_snapshot(snapshot_path: Path | str) -> dict[str, np.ndarray]:
        snapshot_path = Path(snapshot_path)
        archive = np.load(snapshot_path, allow_pickle=False)
        return {k: archive[k] for k in archive.files}

    @staticmethod
    def load_snapshots(run_dir: Path | str, include_final: bool = True) -> list[dict[str, np.ndarray]]:
        run_dir = Path(run_dir)
        snap_dir = run_dir / "snapshots"
        if not snap_dir.exists():
            return []

        # Numbered periodic snapshots, sorted ascending
        paths = sorted(snap_dir.glob("snap_[0-9]*.npz"))

        # Final snapshot appended last (not glob'd to preserve ordering)
        final_path = snap_dir / "snap_final.npz"
        if include_final and final_path.exists():
            paths.append(final_path)

        return [SlamLogger.load_snapshot(p) for p in paths]
