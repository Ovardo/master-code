import numpy as np
import matplotlib.pyplot as plt

import gtsam
from gtsam import Pose2, Point2, Rot2
from gtsam.symbol_shorthand import X, L

DATA_PATH = "/Users/ovar/gtsam-src/examples/Data/victoria_park.txt"
NUM_LINES = 4000  # adjust as needed for testing

def parse_file(file_path: str, num_lines):
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            parts = line.split()
            tag = parts[0]
            if tag == "ODOMETRY":
                i, j = int(parts[1]), int(parts[2])
                dx, dy, dth = map(float, parts[3:6])
                # Information matrix upper triangular:
                I11, I12, I13, I22, I23, I33 = map(float, parts[6:12])
                yield ("ODOM", i, j, dx, dy, dth, (I11, I12, I13, I22, I23, I33))
            elif tag == "LANDMARK":
                pose_id, lm_id = int(parts[1]), int(parts[2])
                r, b = float(parts[3]), float(parts[4])
                # Information matrix upper triangular 2x2:
                J11, J12, J22 = map(float, parts[5:8])
                yield ("LM", pose_id, lm_id, r, b, (J11, J12, J22))
            else:
                raise ValueError(f"Unknown tag: {tag}")

def info_to_noise3(I):
    """Convert 3x3 information upper-tri to a GTSAM Gaussian noise model."""
    I11, I12, I13, I22, I23, I33 = I
    sigmas = np.sqrt(np.array([I11, I22, I33]))
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

def info_to_noise2(J):
    """Convert 2x2 information upper-tri to a GTSAM Gaussian noise model."""
    J11, J12, J22 = J
    sigmas = np.sqrt(np.array([J11, J22]))
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

def dead_reckon_initial(odom_factors):
    """Return dict pose_id -> Pose2 by chaining odometry from pose 0."""
    poses = {0: Pose2(0.0, 0.0, 36 * np.pi / 180)}
    for (i, j, dx, dy, dth, _) in odom_factors:
        if i not in poses:
            continue
        poses[j] = poses[i].compose(Pose2(dx, dy, dth))
    return poses

def backproject_landmarks(initial_poses, lm_meas):
    """Return dict lm_id -> Point2 by averaging back-projections."""
    samples = {}
    for pose_id, lm_id, r, b, _ in lm_meas:
        if pose_id not in initial_poses:
            continue
        pose = initial_poses[pose_id]
        ang = pose.theta() + b
        lx = pose.x() + r * np.cos(ang)
        ly = pose.y() + r * np.sin(ang)
        samples.setdefault(lm_id, []).append((lx, ly))
    out = {}
    for lm_id, pts in samples.items():
        pts = np.array(pts)
        out[lm_id] = Point2(float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    return out

def values_to_arrays(values, pose_ids, lm_ids):
    traj = np.array([[values.atPose2(X(i)).x(), values.atPose2(X(i)).y()] for i in pose_ids])
    lms = np.array([[values.atPoint2(L(j))[0], values.atPoint2(L(j))[1]] for j in lm_ids])
    return traj, lms

# ---------------------------
# Build graph + initial guess
# ---------------------------
graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

odom_factors = []
lm_meas = []

for item in parse_file(DATA_PATH, num_lines=NUM_LINES):
    if item[0] == "ODOM":
        _, i, j, dx, dy, dth, I = item
        odom_factors.append((i, j, dx, dy, dth, I))
    else:
        _, pose_id, lm_id, r, b, J = item
        lm_meas.append((pose_id, lm_id, r, b, J))

# Prior on pose 0 (fix gauge freedom)
prior_sigmas = np.array([1e-3, 1e-3, 1e-3])  # fairly tight, adjust if you want
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
graph.add(gtsam.PriorFactorPose2(X(0), Pose2(0.0, 0.0, 36 * np.pi / 180), prior_noise))

# Add odometry between factors
for i, j, dx, dy, dth, I in odom_factors:
    noise = info_to_noise3(I)
    graph.add(gtsam.BetweenFactorPose2(X(i), X(j), Pose2(dx, dy, dth), noise))

# Add landmark bearing-range factors
for pose_id, lm_id, r, b, J in lm_meas:
    noise = info_to_noise2(J)
    graph.add(gtsam.BearingRangeFactor2D(X(pose_id), L(lm_id), Rot2(b), r, noise))

# Initial guess: dead-reckoned poses + landmark backprojection
init_poses = dead_reckon_initial(odom_factors)
for pid, pose in init_poses.items():
    initial.insert(X(pid), pose)

init_lms = backproject_landmarks(init_poses, lm_meas)
for lm_id, pt in init_lms.items():
    initial.insert(L(lm_id), pt)

pose_ids = sorted(init_poses.keys())
lm_ids = sorted(init_lms.keys())

# ---------------------------
# Optimize
# ---------------------------
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("ERROR")  # "SUMMARY" if you want iteration output
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

# ---------------------------
# Plot
# ---------------------------
traj0, lms0 = values_to_arrays(initial, pose_ids, lm_ids)
traj1, lms1 = values_to_arrays(result, pose_ids, lm_ids)

plt.figure(figsize=(7, 7))
plt.plot(traj0[:, 0], traj0[:, 1], marker="o", label="Initial (dead-reckoned)")
plt.plot(traj1[:, 0], traj1[:, 1], marker="o", label="Optimized")

if len(lm_ids) > 0:
    plt.scatter(lms0[:, 0], lms0[:, 1], marker="x", s=80, label="Landmarks init")
    plt.scatter(lms1[:, 0], lms1[:, 1], marker="x", s=80, label="Landmarks opt")

# for i, (x, y) in zip(pose_ids, traj1):
#     plt.text(x, y, str(i), fontsize=8, ha="right")

# for lm_id, (x, y) in zip(lm_ids, lms1):
#     plt.text(x, y, f"L{lm_id}", fontsize=9, ha="left")

plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("GTSAM SLAM: initial vs optimized")
plt.legend()
plt.show()