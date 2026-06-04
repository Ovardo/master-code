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
from matplotlib.patches import Ellipse, Patch
from tqdm import tqdm

from master_code.utils import rotmat2


DEFAULT_RUN_DIR = Path("runs/20260521_203732_all")
SNAPSHOT_PATTERN = re.compile(r"snap_(\d{5})\.npz$")
ASSOCIATION_PATTERN = re.compile(r"assoc_(\d{5})\.npz$")


@dataclass(frozen=True)
class SnapshotFrame:
    step: int
    poses: np.ndarray
    poses_covariance: np.ndarray | None
    landmarks: np.ndarray
    landmarks_covariance: np.ndarray | None


@dataclass(frozen=True)
class AssociationFrame:
    scan_step: int
    scan_time: float
    pose: np.ndarray
    measurements: np.ndarray
    predicted_measurements: np.ndarray
    association: np.ndarray
    local_landmarks: np.ndarray
    innovation_covariance: np.ndarray


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


def _covariance_array(data: np.lib.npyio.NpzFile, key: str, shape_tail: tuple[int, int]) -> np.ndarray | None:
    if key not in data.files:
        return None

    covariance = np.asarray(data[key], dtype=float)
    if covariance.size == 0:
        return np.empty((0, *shape_tail), dtype=float)
    return covariance.reshape(-1, *shape_tail).copy()


def _confidence_ellipse(
    center: np.ndarray,
    covariance: np.ndarray,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int,
) -> Ellipse | None:
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(covariance)):
        return None

    try:
        eigvals, eigvecs = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    scale_95 = 2.447746830681
    width, height = 2.0 * scale_95 * np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return Ellipse(
        xy=tuple(center),
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=0.7,
        zorder=zorder,
    )


def _clear_patches(patches: list[Ellipse]) -> None:
    for patch in patches:
        patch.remove()
    patches.clear()


def _add_covariance_ellipses(
    ax: plt.Axes,
    centers: np.ndarray,
    covariances: np.ndarray | None,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int,
) -> list[Ellipse]:
    if covariances is None or len(centers) == 0:
        return []

    patches: list[Ellipse] = []
    for center, covariance in zip(centers, covariances):
        ellipse = _confidence_ellipse(
            center,
            covariance,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=zorder,
        )
        if ellipse is None:
            continue
        ax.add_patch(ellipse)
        patches.append(ellipse)
    return patches


def _pose_xy_covariances_world(poses: np.ndarray, poses_covariance: np.ndarray | None) -> np.ndarray | None:
    if poses_covariance is None or len(poses) == 0:
        return None

    rotations = rotmat2(poses[:, 2])
    xy_covariances = poses_covariance[:, :2, :2]
    return rotations @ xy_covariances @ rotations.transpose((0, 2, 1))


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
            poses_covariance=_covariance_array(data, "poses_covariance", (3, 3)),
            landmarks=_as_xy(data["landmarks"]).copy(),
            landmarks_covariance=_covariance_array(data, "landmarks_covariance", (2, 2)),
        )


def load_association_frame(path: Path) -> AssociationFrame:
    with np.load(path, allow_pickle=False) as data:
        if "predicted_measurements" in data.files:
            predicted_measurements = _as_xy(data["predicted_measurements"]).copy()
        else:
            predicted_measurements = np.empty((0, 2), dtype=float)

        if "innovation_covariance" in data.files:
            innovation_covariance = np.asarray(data["innovation_covariance"], dtype=float).copy()
        else:
            innovation_covariance = np.empty((0, 0), dtype=float)

        return AssociationFrame(
            scan_step=int(data["scan_step"][0]),
            scan_time=float(data["scan_time"][0]),
            pose=np.asarray(data["pose"], dtype=float).reshape(3).copy(),
            measurements=_as_xy(data["measurements"]).copy(),
            predicted_measurements=predicted_measurements,
            association=np.asarray(data["association"], dtype=int).reshape(-1).copy(),
            local_landmarks=_as_xy(data["local_landmarks"]).copy(),
            innovation_covariance=innovation_covariance,
        )


def compute_axis_limits(
    run_dir: Path,
    fallback_snapshot: Path,
    extra_points: tuple[np.ndarray, ...] = (),
) -> tuple[tuple[float, float], tuple[float, float]]:
    final_snapshot = run_dir / "snapshots" / "snap_final.npz"
    snapshot_path = final_snapshot if final_snapshot.exists() else fallback_snapshot

    with np.load(snapshot_path, allow_pickle=False) as data:
        poses_xy = _as_xy(data["poses"][:, :2])
        landmarks = _as_xy(data["landmarks"])

    point_sets = [
        points
        for points in (poses_xy, landmarks, *(_as_xy(points) for points in extra_points))
        if len(points) > 0
    ]
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


def _marginal_innovation_covariances(innovation_covariance: np.ndarray, num_predicted: int) -> np.ndarray:
    """Extract the per-prediction 2x2 marginal blocks from the (2L, 2L) innovation covariance."""
    if num_predicted == 0:
        return np.empty((0, 2, 2), dtype=float)

    covariance = np.asarray(innovation_covariance, dtype=float)
    if covariance.shape != (2 * num_predicted, 2 * num_predicted):
        return np.empty((0, 2, 2), dtype=float)

    blocks = covariance.reshape(num_predicted, 2, num_predicted, 2)
    indices = np.arange(num_predicted)
    return blocks[indices, :, indices, :].copy()


def _innovation_segments(measurements: np.ndarray, predicted: np.ndarray, association: np.ndarray) -> list[np.ndarray]:
    """Connect each associated measurement to its predicted measurement in (range, bearing) space."""
    segments: list[np.ndarray] = []
    for measurement, predicted_index in zip(measurements, association):
        if predicted_index < 0 or predicted_index >= len(predicted):
            continue
        segments.append(np.vstack((measurement, predicted[predicted_index])))
    return segments


def compute_innovation_limits(
    association_paths: dict[int, Path],
    steps: list[int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fixed (range, bearing) axis limits spanning every measurement, prediction and gate ellipse."""
    scale_95 = 2.447746830681
    range_values: list[np.ndarray] = []
    bearing_values: list[np.ndarray] = []

    for step in steps:
        frame = load_association_frame(association_paths[step])
        measurements = frame.measurements
        predicted = frame.predicted_measurements
        if len(measurements) > 0:
            range_values.append(measurements[:, 0])
            bearing_values.append(measurements[:, 1])
        if len(predicted) > 0:
            covariances = _marginal_innovation_covariances(frame.innovation_covariance, len(predicted))
            half_range = np.zeros(len(predicted), dtype=float)
            half_bearing = np.zeros(len(predicted), dtype=float)
            if len(covariances) == len(predicted):
                half_range = scale_95 * np.sqrt(np.maximum(covariances[:, 0, 0], 0.0))
                half_bearing = scale_95 * np.sqrt(np.maximum(covariances[:, 1, 1], 0.0))
            range_values.append(predicted[:, 0] - half_range)
            range_values.append(predicted[:, 0] + half_range)
            bearing_values.append(predicted[:, 1] - half_bearing)
            bearing_values.append(predicted[:, 1] + half_bearing)

    if not range_values or not bearing_values:
        return (0.0, 1.0), (-np.pi, np.pi)

    ranges = np.concatenate(range_values)
    bearings = np.concatenate(bearing_values)
    ranges = ranges[np.isfinite(ranges)]
    bearings = bearings[np.isfinite(bearings)]
    if len(ranges) == 0 or len(bearings) == 0:
        return (0.0, 1.0), (-np.pi, np.pi)

    range_pad = max(0.05 * (ranges.max() - ranges.min()), 0.5)
    bearing_pad = max(0.05 * (bearings.max() - bearings.min()), 0.05)
    range_lim = (float(ranges.min() - range_pad), float(ranges.max() + range_pad))
    bearing_lim = (float(bearings.min() - bearing_pad), float(bearings.max() + bearing_pad))
    return range_lim, bearing_lim


def render_video(
    run_dir: Path,
    output_path: Path,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    fps: int = 20,
    dpi: int = 150,
    reference_poses: np.ndarray | None = None,
    reference_landmarks: np.ndarray | None = None,
    title: str = "SLAM Trajectory and Data Associations",
    show_covariances: bool = False,
    pose_cov_stride: int = 10,
    show_innovation: bool = False,
) -> None:
    steps, snapshot_paths, association_paths = discover_frame_steps(run_dir, start, stop, stride)
    if pose_cov_stride <= 0:
        raise ValueError(f"pose_cov_stride must be positive, got {pose_cov_stride}")

    if not animation.writers.is_available("ffmpeg"):
        available = ", ".join(animation.writers.list())
        raise RuntimeError(
            "Matplotlib cannot access the ffmpeg writer. "
            f"Available writers: {available or 'none'}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_poses_xy = np.empty((0, 2), dtype=float)
    if reference_poses is not None:
        reference_poses = np.asarray(reference_poses, dtype=float)
        if reference_poses.size > 0:
            reference_poses_xy = _as_xy(reference_poses[:, :2])

    reference_landmarks_xy = (
        _as_xy(reference_landmarks)
        if reference_landmarks is not None
        else np.empty((0, 2), dtype=float)
    )

    xlim, ylim = compute_axis_limits(
        run_dir,
        snapshot_paths[steps[-1]],
        extra_points=(reference_poses_xy, reference_landmarks_xy),
    )
    heading_length = max(3.0, 0.015 * (xlim[1] - xlim[0]))

    if show_innovation:
        fig, (ax, ax_innov) = plt.subplots(
            1,
            2,
            figsize=(15, 8),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.25, 1.0]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax_innov = None
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.4, alpha=0.7)

    if len(reference_poses_xy) > 0:
        ax.plot(
            reference_poses_xy[:, 0],
            reference_poses_xy[:, 1],
            color="black",
            linestyle="--",
            lw=1.0,
            alpha=0.75,
            label="Ground truth trajectory",
            zorder=1,
        )
    if len(reference_landmarks_xy) > 0:
        ax.scatter(
            reference_landmarks_xy[:, 0],
            reference_landmarks_xy[:, 1],
            s=18,
            marker="x",
            c="0.25",
            alpha=0.65,
            label="Ground truth landmarks",
            zorder=1,
        )

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
    pose_covariance_patches: list[Ellipse] = []
    landmark_covariance_patches: list[Ellipse] = []
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

    handles, labels = ax.get_legend_handles_labels()
    if show_covariances:
        covariance_handles = [
            Patch(facecolor="steelblue", edgecolor="steelblue", alpha=0.18, label=f"Pose covariance ({pose_cov_stride} poses)"),
            Patch(facecolor="tomato", edgecolor="tomato", alpha=0.16, label="Landmark covariance"),
        ]
        handles.extend(covariance_handles)
        labels.extend(handle.get_label() for handle in covariance_handles)
    ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=8)

    innovation_predicted_scatter = None
    innovation_associated_scatter = None
    innovation_unassociated_scatter = None
    innovation_lines = None
    innovation_gate_patches: list[Ellipse] = []
    if ax_innov is not None:
        range_lim, bearing_lim = compute_innovation_limits(association_paths, steps)
        ax_innov.set_title("Measurement innovation space")
        ax_innov.set_xlabel("range [m]")
        ax_innov.set_ylabel("bearing [rad]")
        ax_innov.set_xlim(*range_lim)
        ax_innov.set_ylim(*bearing_lim)
        ax_innov.grid(True, lw=0.4, alpha=0.7)

        innovation_predicted_scatter = ax_innov.scatter([], [], s=40, marker="x", c="orange", label="Predicted measurements", zorder=4)
        innovation_associated_scatter = ax_innov.scatter([], [], s=30, marker=".", c="steelblue", label="Associated measurements", zorder=6)
        innovation_unassociated_scatter = ax_innov.scatter([], [], s=28, marker=".", c="dimgray", label="Unassociated measurements", zorder=5)
        innovation_lines = LineCollection([], colors="seagreen", linewidths=0.9, alpha=0.75, zorder=3)
        ax_innov.add_collection(innovation_lines)

        innovation_handles, innovation_labels = ax_innov.get_legend_handles_labels()
        gate_handle = Patch(facecolor="none", edgecolor="orange", label="Marginal innovation gate (95%)")
        innovation_handles.append(gate_handle)
        innovation_labels.append(gate_handle.get_label())
        ax_innov.legend(handles=innovation_handles, labels=innovation_labels, loc="lower right", fontsize=8)

    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": title, "artist": "master_code"},
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
            _clear_patches(pose_covariance_patches)
            _clear_patches(landmark_covariance_patches)
            if show_covariances:
                pose_indices = np.arange(0, len(poses), pose_cov_stride)
                pose_covariances_world = _pose_xy_covariances_world(poses, snapshot.poses_covariance)
                pose_covariance_patches.extend(
                    _add_covariance_ellipses(
                        ax,
                        poses[pose_indices, :2],
                        None if pose_covariances_world is None else pose_covariances_world[pose_indices],
                        facecolor="steelblue",
                        edgecolor="steelblue",
                        alpha=0.18,
                        zorder=2,
                    )
                )
                landmark_covariance_patches.extend(
                    _add_covariance_ellipses(
                        ax,
                        snapshot.landmarks,
                        snapshot.landmarks_covariance,
                        facecolor="tomato",
                        edgecolor="tomato",
                        alpha=0.16,
                        zorder=2,
                    )
                )

            if ax_innov is not None:
                predicted_measurements = association.predicted_measurements
                _clear_patches(innovation_gate_patches)
                innovation_predicted_scatter.set_offsets(_scatter_offsets(predicted_measurements))
                innovation_associated_scatter.set_offsets(_scatter_offsets(association.measurements[is_associated]))
                innovation_unassociated_scatter.set_offsets(_scatter_offsets(association.measurements[~is_associated]))
                innovation_lines.set_segments(
                    _innovation_segments(association.measurements, predicted_measurements, association.association)
                )
                marginal_covariances = _marginal_innovation_covariances(
                    association.innovation_covariance, len(predicted_measurements)
                )
                innovation_gate_patches.extend(
                    _add_covariance_ellipses(
                        ax_innov,
                        predicted_measurements,
                        marginal_covariances if len(marginal_covariances) == len(predicted_measurements) else None,
                        facecolor="none",
                        edgecolor="orange",
                        alpha=0.8,
                        zorder=3,
                    )
                )

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
    parser.add_argument("--show-covariances", action="store_true")
    parser.add_argument("--pose-cov-stride", type=int, default=10)
    parser.add_argument(
        "--show-innovation",
        action="store_true",
        help="Add a side panel plotting measurements and predicted measurements in innovation space with marginal gates.",
    )
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
        show_covariances=args.show_covariances,
        pose_cov_stride=args.pose_cov_stride,
        show_innovation=args.show_innovation,
    )
    print(f"Saved video to {output}")


if __name__ == "__main__":
    main()
