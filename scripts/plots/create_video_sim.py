from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from master_code.loaders.simulated import SimulatedDataLoader
from master_code.logger import SlamLogger
from create_video import render_video


DEFAULT_RUN_DIR = Path("runs/sim/20260603_130915_all")


def load_sim_references(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = SlamLogger.load_metadata(run_dir)
    if metadata.get("dataset") != "sim":
        raise ValueError(f"Expected a simulated run, got dataset={metadata.get('dataset')!r}")

    loader = SimulatedDataLoader()
    poses_gt = loader.poses_gt

    final_snapshot = run_dir / "snapshots" / "snap_final.npz"
    if final_snapshot.exists():
        with np.load(final_snapshot, allow_pickle=False) as data:
            poses_gt = poses_gt[: len(data["poses"])]

    return poses_gt, loader.landmarks_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simulated SLAM association video.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None, help="Exclusive stop step. Defaults to the last available step + 1.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-covariances", action="store_true")
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
        output = run_dir / "figures" / "trajectory_associations_sim.mp4"

    poses_gt, landmarks_gt = load_sim_references(run_dir)
    render_video(
        run_dir=run_dir,
        output_path=output,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        fps=args.fps,
        dpi=args.dpi,
        reference_poses=poses_gt,
        reference_landmarks=landmarks_gt,
        title="Simulated SLAM Trajectory and Data Associations",
        show_covariances=not args.no_covariances,
        pose_cov_stride=args.pose_cov_stride,
        show_innovation=args.show_innovation,
    )
    print(f"Saved video to {output}")


if __name__ == "__main__":
    main()
