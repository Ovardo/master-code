#!/usr/bin/env python3
# Created by ChatGPT 5.5.
"""
Batch optimization of iSAM/GTSAM victoriaPark.txt using GTSAM Python. 

Usage:
    python optimize_victoria_park.py victoriaPark.txt

Install:
    pip install gtsam numpy matplotlib
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import gtsam


# ----------------------------
# Basic SE(2) helper functions
# ----------------------------

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def pose2_transform_from(pose: gtsam.Pose2, p_local: np.ndarray) -> np.ndarray:
    """World point = T * local point."""
    c = math.cos(pose.theta())
    s = math.sin(pose.theta())
    x = pose.x() + c * p_local[0] - s * p_local[1]
    y = pose.y() + s * p_local[0] + c * p_local[1]
    return np.array([x, y], dtype=float)


def pose2_transform_to(pose: gtsam.Pose2, p_world: np.ndarray) -> np.ndarray:
    """Local point = T^{-1} * world point."""
    dx = p_world[0] - pose.x()
    dy = p_world[1] - pose.y()
    c = math.cos(pose.theta())
    s = math.sin(pose.theta())
    return np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
    ], dtype=float)


def point2_to_np(p) -> np.ndarray:
    """GTSAM Point2 is usually numpy-like in Python."""
    return np.array([float(p[0]), float(p[1])], dtype=float)


def covariance_noise(C: np.ndarray):
    C = np.asarray(C, dtype=float)
    return gtsam.noiseModel.Gaussian.Covariance(C)


# ----------------------------
# Custom local xy landmark factor
# ----------------------------

def make_pose_point_local_xy_factor(
    pose_key: int,
    landmark_key: int,
    z_local: np.ndarray,
    cov: np.ndarray,
):
    """
    LANDMARK factor measurement model:

        z = R_i^T * (l_j - t_i) + noise

    where z is a 2D point in the robot/pose frame.

    Error:

        e = R_i^T * (l_j - t_i) - z

    This matches the iSAM Pose2d_Point2d relative landmark measurement.
    """

    noise = covariance_noise(cov)
    z_local = np.asarray(z_local, dtype=float)

    def error_func(this, values: gtsam.Values, H):
        pose = values.atPose2(pose_key)
        landmark = point2_to_np(values.atPoint2(landmark_key))

        q = pose2_transform_to(pose, landmark)
        e = q - z_local

        if H is not None:
            theta = pose.theta()
            c = math.cos(theta)
            s = math.sin(theta)

            # q = R^T (l - t)
            qx, qy = q[0], q[1]

            # Jacobian wrt Pose2 right perturbation [dx_body, dy_body, dtheta]
            # q' ≈ q - rho - dtheta * Omega q
            H_pose = np.array([
                [-1.0,  0.0,  qy],
                [ 0.0, -1.0, -qx],
            ], dtype=float)

            # Jacobian wrt landmark world coordinates
            H_landmark = np.array([
                [ c, s],
                [-s, c],
            ], dtype=float)

            # GTSAM CustomFactor expects Fortran-contiguous matrices.
            H[0] = np.asfortranarray(H_pose)
            H[1] = np.asfortranarray(H_landmark)

        return e

    return gtsam.CustomFactor(
        noise,
        [int(pose_key), int(landmark_key)],
        error_func,
    )


# ----------------------------
# Parser
# ----------------------------

def parse_victoria_park(path: Path):
    odom = []
    lms = []

    pose_keys = set()
    landmark_keys = set()

    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "ODOMETRY":
                if len(parts) != 12:
                    raise ValueError(f"Bad ODOMETRY line {line_no}: {line}")

                i = int(parts[1])
                j = int(parts[2])

                dx = float(parts[3])
                dy = float(parts[4])
                dtheta = float(parts[5])

                sxx = float(parts[6])
                sxy = float(parts[7])
                sxt = float(parts[8])
                syy = float(parts[9])
                syt = float(parts[10])
                stt = float(parts[11])

                cov = np.array([
                    [sxx, sxy, sxt],
                    [sxy, syy, syt],
                    [sxt, syt, stt],
                ], dtype=float)

                odom.append((i, j, dx, dy, dtheta, cov))
                pose_keys.add(i)
                pose_keys.add(j)

            elif tag == "LANDMARK":
                if len(parts) != 8:
                    raise ValueError(f"Bad LANDMARK line {line_no}: {line}")

                i = int(parts[1])
                l = int(parts[2])

                zx = float(parts[3])
                zy = float(parts[4])

                sxx = float(parts[5])
                sxy = float(parts[6])
                syy = float(parts[7])

                cov = np.array([
                    [sxx, sxy],
                    [sxy, syy],
                ], dtype=float)

                lms.append((i, l, zx, zy, cov))
                pose_keys.add(i)
                landmark_keys.add(l)

            else:
                raise ValueError(f"Unsupported tag on line {line_no}: {tag}")

    overlap = pose_keys.intersection(landmark_keys)
    if overlap:
        raise RuntimeError(
            f"Some keys are both poses and landmarks: {sorted(list(overlap))[:10]}"
        )

    return odom, lms, pose_keys, landmark_keys


# ----------------------------
# Initialization
# ----------------------------

def initialize_values(odom, lms, pose_keys, landmark_keys):
    """
    Initialize poses by dead-reckoning through odometry.
    Initialize landmarks by averaging projected landmark observations.
    """

    if not odom:
        raise RuntimeError("No odometry factors found.")

    first_pose_key = odom[0][0]

    pose_init = {
        first_pose_key: gtsam.Pose2(0.0, 0.0, 0.0)
    }

    # Dead-reckon through odometry factors.
    for i, j, dx, dy, dtheta, _ in odom:
        if i not in pose_init:
            raise RuntimeError(
                f"Pose {i} was not initialized before odometry {i}->{j}. "
                "The odometry chain may not be ordered as expected."
            )

        if j not in pose_init:
            delta = gtsam.Pose2(dx, dy, dtheta)
            pose_init[j] = pose_init[i].compose(delta)

    missing_poses = sorted(pose_keys.difference(pose_init.keys()))
    if missing_poses:
        raise RuntimeError(
            f"{len(missing_poses)} poses were not initialized. "
            f"First few: {missing_poses[:10]}"
        )

    # Project each local landmark observation into the odometry world frame.
    landmark_observations_world = defaultdict(list)

    for pose_key, landmark_key, zx, zy, _ in lms:
        pose = pose_init[pose_key]
        z = np.array([zx, zy], dtype=float)
        p_world = pose2_transform_from(pose, z)
        landmark_observations_world[landmark_key].append(p_world)

    landmark_init = {}

    for landmark_key in landmark_keys:
        obs = landmark_observations_world[landmark_key]
        landmark_init[landmark_key] = np.mean(np.vstack(obs), axis=0)

    values = gtsam.Values()

    for key in sorted(pose_keys):
        values.insert(int(key), pose_init[key])

    for key in sorted(landmark_keys):
        p = landmark_init[key]
        values.insert(int(key), gtsam.Point2(float(p[0]), float(p[1])))

    return values, first_pose_key


# ----------------------------
# Graph construction
# ----------------------------

def build_graph(odom, lms, first_pose_key: int):
    graph = gtsam.NonlinearFactorGraph()

    # Fix gauge freedom.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-6, 1e-6, 1e-8], dtype=float)
    )

    graph.add(
        gtsam.PriorFactorPose2(
            int(first_pose_key),
            gtsam.Pose2(0.0, 0.0, 0.0),
            prior_noise,
        )
    )

    for i, j, dx, dy, dtheta, cov in odom:
        graph.add(
            gtsam.BetweenFactorPose2(
                int(i),
                int(j),
                gtsam.Pose2(dx, dy, dtheta),
                covariance_noise(cov),
            )
        )

    for pose_key, landmark_key, zx, zy, cov in lms:
        z = np.array([zx, zy], dtype=float)
        graph.add(
            make_pose_point_local_xy_factor(
                int(pose_key),
                int(landmark_key),
                z,
                cov,
            )
        )

    return graph


# ----------------------------
# Plotting
# ----------------------------

def extract_poses(values: gtsam.Values, pose_order):
    poses = []
    for key in pose_order:
        p = values.atPose2(int(key))
        poses.append([p.x(), p.y(), p.theta()])
    return np.array(poses)


def extract_landmarks(values: gtsam.Values, landmark_keys):
    landmarks = []
    for key in sorted(landmark_keys):
        p = point2_to_np(values.atPoint2(int(key)))
        landmarks.append(p)
    return np.array(landmarks)

def get_pose_order(odom):
    pose_order = [odom[0][0]]
    for i, j, *_ in odom:
        pose_order.append(j)
    return pose_order

def plot_result(initial, result, odom, landmark_keys, out_path=None):
    pose_order = get_pose_order(odom)

    init_poses = extract_poses(initial, pose_order)
    opt_poses = extract_poses(result, pose_order)
    opt_landmarks = extract_landmarks(result, landmark_keys)

    plt.figure(figsize=(10, 8))

    plt.plot(
        init_poses[:, 0],
        init_poses[:, 1],
        "--",
        linewidth=1.0,
        label="initial odometry trajectory",
    )

    plt.plot(
        opt_poses[:, 0],
        opt_poses[:, 1],
        "-",
        linewidth=1.5,
        label="optimized trajectory",
    )

    plt.scatter(
        opt_landmarks[:, 0],
        opt_landmarks[:, 1],
        s=20,
        marker="x",
        label="optimized landmarks",
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Victoria Park batch SLAM from victoriaPark.txt")
    plt.legend()

    if out_path is not None:
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")

    plt.show()


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Path to victoriaPark.txt")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("victoria_park_batch_result.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="SUMMARY",
        choices=["SILENT", "SUMMARY", "TERMINATION", "LAMBDA", "TRYLAMBDA", "TRYCONFIG"],
    )
    args = parser.parse_args()

    odom, lms, pose_keys, landmark_keys = parse_victoria_park(args.file)

    print("Parsed dataset:")
    print(f"  odometry factors:  {len(odom)}")
    print(f"  landmark factors:  {len(lms)}")
    print(f"  pose variables:    {len(pose_keys)}")
    print(f"  landmark variables:{len(landmark_keys)}")
    print(f"  total variables:   {len(pose_keys) + len(landmark_keys)}")

    initial, first_pose_key = initialize_values(
        odom,
        lms,
        pose_keys,
        landmark_keys,
    )

    graph = build_graph(odom, lms, first_pose_key)

    print(f"Graph has {graph.size()} factors.")
    print("Initial graph error:", graph.error(initial))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(args.max_iterations)
    params.setVerbosityLM(args.verbosity)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    print("Final graph error:", graph.error(result))

    pose_order = get_pose_order(odom)
    print("Pose order equals sorted pose keys:", pose_order == sorted(pose_keys))
    print("Unique poses in order:", len(set(pose_order)))
    print("Total poses:", len(pose_order))

    plot_result(
        initial,
        result,
        odom,
        landmark_keys,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()