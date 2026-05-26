from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from tqdm import tqdm

from master_code.utils import rotmat2


DEFAULT_RUN_DIR = Path("runs/20260521_203732_all")
SNAPSHOT_PATTERN = re.compile(r"snap_(\d{5})\.npz$")
ASSOCIATION_PATTERN = re.compile(r"assoc_(\d{5})\.npz$")


@dataclass(frozen=True)
class SnapshotFrame:
    step: int
    poses: np.ndarray
    landmarks: np.ndarray


@dataclass(frozen=True)
class AssociationFrame:
    scan_step: int
    scan_time: float
    pose: np.ndarray
    measurements: np.ndarray
    association: np.ndarray
    local_landmarks: np.ndarray


def _as_xy(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    return array.reshape(-1, 2)


def _scatter_offsets(points: np.ndarray) -> np.ndarray:
    points = _as_xy(points)
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    return points


def _measurement_points_world(pose: np.ndarray, measurements: np.ndarray) -> np.ndarray:
    """Transform range-bearing measurements to world-frame endpoints."""
    measurements = _as_xy(measurements)
    if len(measurements) == 0:
        return np.empty((0, 2), dtype=float)

    ranges = measurements[:, 0]
    bearings = measurements[:, 1]
    local_points = np.column_stack((ranges * np.cos(bearings), ranges * np.sin(bearings)))
    return pose[:2] + local_points @ rotmat2(pose[2]).T


def _discover_step_files(directory: Path, pattern: re.Pattern[str], label: str) -> dict[int, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing {label} directory: {directory}")

    files: dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        files[int(match.group(1))] = path

    if not files:
        raise FileNotFoundError(f"No numbered {label} files found in {directory}")

    return dict(sorted(files.items()))


def discover_frame_steps(run_dir: Path, start: int, stop: int | None, stride: int) -> tuple[list[int], dict[int, Path], dict[int, Path]]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if stop is not None and stop <= start:
        raise ValueError(f"stop must be greater than start, got start={start}, stop={stop}")

    snapshot_paths = _discover_step_files(run_dir / "snapshots", SNAPSHOT_PATTERN, "snapshot")
    association_paths = _discover_step_files(run_dir / "associations", ASSOCIATION_PATTERN, "association")

    snapshot_steps = set(snapshot_paths)
    association_steps = set(association_paths)
    missing_associations = sorted(snapshot_steps - association_steps)
    missing_snapshots = sorted(association_steps - snapshot_steps)
    if missing_associations or missing_snapshots:
        raise RuntimeError(
            "Snapshot/association step mismatch: "
            f"{len(missing_associations)} snapshots lack associations, "
            f"{len(missing_snapshots)} associations lack snapshots. "
            f"First missing associations: {missing_associations[:5]}; "
            f"first missing snapshots: {missing_snapshots[:5]}"
        )

    all_steps = sorted(snapshot_steps)
    stop_value = all_steps[-1] + 1 if stop is None else stop
    selected_steps = [
        step
        for step in all_steps
        if start <= step < stop_value and (step - start) % stride == 0
    ]

    if not selected_steps:
        raise ValueError(
            f"No frames selected for start={start}, stop={stop_value}, stride={stride}. "
            f"Available step range is {all_steps[0]}..{all_steps[-1]}."
        )

    return selected_steps, snapshot_paths, association_paths


def load_snapshot_frame(path: Path) -> SnapshotFrame:
    with np.load(path, allow_pickle=False) as data:
        return SnapshotFrame(
            step=int(data["step"][0]),
            poses=np.asarray(data["poses"], dtype=float).copy(),
            landmarks=_as_xy(data["landmarks"]).copy(),
        )


def load_association_frame(path: Path) -> AssociationFrame:
    with np.load(path, allow_pickle=False) as data:
        return AssociationFrame(
            scan_step=int(data["scan_step"][0]),
            scan_time=float(data["scan_time"][0]),
            pose=np.asarray(data["pose"], dtype=float).reshape(3).copy(),
            measurements=_as_xy(data["measurements"]).copy(),
            association=np.asarray(data["association"], dtype=int).reshape(-1).copy(),
            local_landmarks=_as_xy(data["local_landmarks"]).copy(),
        )


def compute_axis_limits(run_dir: Path, fallback_snapshot: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    final_snapshot = run_dir / "snapshots" / "snap_final.npz"
    snapshot_path = final_snapshot if final_snapshot.exists() else fallback_snapshot

    with np.load(snapshot_path, allow_pickle=False) as data:
        poses_xy = _as_xy(data["poses"][:, :2])
        landmarks = _as_xy(data["landmarks"])

    point_sets = [points for points in (poses_xy, landmarks) if len(points) > 0]
    if not point_sets:
        return (-1.0, 1.0), (-1.0, 1.0)

    points = np.vstack(point_sets)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return (-1.0, 1.0), (-1.0, 1.0)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    half_width = max(span / 2.0, 1.0)
    padding = max(20.0, 0.05 * span)
    half_width += padding

    xlim = (float(center[0] - half_width), float(center[0] + half_width))
    ylim = (float(center[1] - half_width), float(center[1] + half_width))
    return xlim, ylim


def _association_segments(measurement_world: np.ndarray, association: np.ndarray, local_landmarks: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for measurement_xy, landmark_index in zip(measurement_world, association):
        if landmark_index < 0 or landmark_index >= len(local_landmarks):
            continue
        segments.append(np.vstack((measurement_xy, local_landmarks[landmark_index])))
    return segments


def render_video(
    run_dir: Path,
    output_path: Path,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    fps: int = 20,
    dpi: int = 150,
) -> None:
    steps, snapshot_paths, association_paths = discover_frame_steps(run_dir, start, stop, stride)

    if not animation.writers.is_available("ffmpeg"):
        available = ", ".join(animation.writers.list())
        raise RuntimeError(
            "Matplotlib cannot access the ffmpeg writer. "
            f"Available writers: {available or 'none'}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    xlim, ylim = compute_axis_limits(run_dir, snapshot_paths[steps[-1]])
    heading_length = max(3.0, 0.015 * (xlim[1] - xlim[0]))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.set_title("SLAM Trajectory and Data Associations")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.4, alpha=0.7)

    trajectory_line, = ax.plot([], [], color="steelblue", lw=1.2, label="Estimated trajectory", zorder=3)
    heading_line, = ax.plot([], [], color="black", lw=1.2, zorder=7)
    start_marker = ax.scatter([], [], s=40, marker="o", facecolors="white", edgecolors="green", label="Start", zorder=8)
    pose_marker = ax.scatter([], [], s=45, marker="o", c="black", label="Current pose", zorder=8)
    landmarks_scatter = ax.scatter([], [], s=14, marker=".", c="tomato", alpha=0.7, label="Confirmed landmarks", zorder=2)
    local_scatter = ax.scatter([], [], s=36, marker="x", c="orange", label="Local candidates", zorder=4)
    unassociated_scatter = ax.scatter([], [], s=30, marker=".", c="dimgray", label="Unassociated measurements", zorder=5)
    associated_scatter = ax.scatter([], [], s=36, marker=".", c="steelblue", label="Associated measurements", zorder=6)
    association_lines = LineCollection([], colors="seagreen", linewidths=0.9, alpha=0.75, zorder=5)
    ax.add_collection(association_lines)
    frame_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.86, "boxstyle": "round,pad=0.35"},
    )

    ax.legend(loc="lower right", fontsize=8)

    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": "SLAM trajectory associations", "artist": "master_code"},
        bitrate=2400,
    )

    with writer.saving(fig, str(output_path), dpi=dpi):
        for step in tqdm(steps, desc="Rendering SLAM video", unit="frame"):
            snapshot = load_snapshot_frame(snapshot_paths[step])
            association = load_association_frame(association_paths[step])

            if len(association.association) != len(association.measurements):
                raise ValueError(
                    f"Association length mismatch at step {step}: "
                    f"{len(association.association)} associations for "
                    f"{len(association.measurements)} measurements."
                )

            poses = snapshot.poses
            current_pose = poses[-1] if len(poses) > 0 else association.pose
            measurement_world = _measurement_points_world(association.pose, association.measurements)
            is_associated = association.association >= 0
            associated_points = measurement_world[is_associated]
            unassociated_points = measurement_world[~is_associated]
            segments = _association_segments(
                measurement_world,
                association.association,
                association.local_landmarks,
            )

            trajectory_line.set_data(poses[:, 0], poses[:, 1])
            start_marker.set_offsets(_scatter_offsets(poses[:1, :2]))
            pose_marker.set_offsets(_scatter_offsets(current_pose[:2].reshape(1, 2)))
            landmarks_scatter.set_offsets(_scatter_offsets(snapshot.landmarks))
            local_scatter.set_offsets(_scatter_offsets(association.local_landmarks))
            associated_scatter.set_offsets(_scatter_offsets(associated_points))
            unassociated_scatter.set_offsets(_scatter_offsets(unassociated_points))
            association_lines.set_segments(segments)

            heading = heading_length * np.array([np.cos(current_pose[2]), np.sin(current_pose[2])])
            heading_line.set_data(
                [current_pose[0], current_pose[0] + heading[0]],
                [current_pose[1], current_pose[1] + heading[1]],
            )

            frame_text.set_text(
                "\n".join(
                    [
                        f"step: {association.scan_step}",
                        f"time: {association.scan_time:.2f} s",
                        f"poses: {len(poses)}",
                        f"landmarks: {len(snapshot.landmarks)}",
                        f"associated: {int(np.sum(is_associated))}",
                        f"unassociated: {int(np.sum(~is_associated))}",
                    ]
                )
            )

            writer.grab_frame()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a streaming SLAM trajectory association video.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None, help="Exclusive stop step. Defaults to the last available step + 1.")
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output = args.output
    if output is None:
        output = run_dir / "figures" / "trajectory_associations.mp4"

    render_video(
        run_dir=run_dir,
        output_path=output,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        fps=args.fps,
        dpi=args.dpi,
    )
    print(f"Saved video to {output}")


if __name__ == "__main__":
    main()
