import matplotlib.pyplot as plt
import numpy as np

np.random.seed(43)

T = 400
dt = 0.1
time_steps = np.arange(0, T, dt)

L = 2.83

u_acc = 0 * np.ones_like(time_steps)  # constant acceleration

# Timed steering profile: left(+), right(-), straight(0)
turn_angle = 0.1
u_alpha = np.zeros_like(time_steps)
steering_schedule = [
    (20, +turn_angle),  # left
    (20, 0.0),          # straight
    (20, -turn_angle),  # right
    (10, 0.0),          # straight
    (10, -turn_angle),  # right
    (20, +turn_angle),  # left
    (10, 0.0),          # straight
    (30, +turn_angle),  # left
    (10, -turn_angle),  # right
    (20, 0.0),          # straight
    (15, +turn_angle),  # extra left
    (15, 0.0),          # extra straight
    (10, -turn_angle),  # extra right
    (10, 0.0),          # extra straight
    (20, -turn_angle),  # added right
    (10, 0.0),          # added straight
    (10, +turn_angle),  # added left
    (10, 0.0),          # added straight
    (5, -turn_angle),   # added short right
    (5, 0.0),           # added short straight
    (25, -turn_angle),  # return attempt right
    (20, 0.0),          # return attempt straight
    (20, -turn_angle),  # return attempt right
    (15, 0.0),          # return attempt straight
    (10, +turn_angle),  # return attempt left
    (10, 0.0),          # return attempt straight
    (10, -turn_angle),  # return attempt right
    (20, +turn_angle),  
]

start_idx = 0
for duration_sec, alpha_cmd in steering_schedule:
    end_idx = min(start_idx + int(duration_sec / dt), len(time_steps))
    u_alpha[start_idx:end_idx] = alpha_cmd
    start_idx = end_idx

   

state = np.zeros((len(time_steps), 4))  # x, y, psi, u
state[0, :] = [0.0, 0.0, 0.0, 3.0]  # initial state

for i in range(1, len(time_steps)):
    x, y, psi, u = state[i-1, :]
    a = u_acc[i-1]
    alpha = u_alpha[i-1]

    # Update state using simple kinematic bicycle model
    x += u * np.cos(psi) * dt
    y += u * np.sin(psi) * dt
    psi += (u / L) * np.tan(alpha) * dt
    u += a * dt

    state[i, :] = [x, y, psi, u] + np.random.normal(0, 0.01, size=4)  # add some noise

# Sample 2D landmarks within a radius of the trajectory.
landmark_radius = 30.0
num_landmarks = 80
trajectory_xy = state[:, :2]
anchor_indices = np.random.choice(len(trajectory_xy), size=num_landmarks, replace=True)
anchor_points = trajectory_xy[anchor_indices]

angles = 2.0 * np.pi * np.random.rand(num_landmarks)
radial_distances = landmark_radius * np.sqrt(np.random.rand(num_landmarks))
landmark_offsets = np.column_stack((radial_distances * np.cos(angles), radial_distances * np.sin(angles)))
landmarks = anchor_points + landmark_offsets

# Generate range-bearing measurements to landmarks within sensor radius.
sensor_radius = 40.0
range_noise_std = 0.1
bearing_noise_std = np.deg2rad(1.0)

measurements_per_pose = []
total_measurements = 0
for pose_idx, (x, y, psi, _) in enumerate(state):
    dx = landmarks[:, 0] - x
    dy = landmarks[:, 1] - y
    ranges = np.hypot(dx, dy)

    visible_mask = ranges <= sensor_radius
    visible_landmark_ids = np.flatnonzero(visible_mask)

    if visible_landmark_ids.size == 0:
        measurements_per_pose.append((
            pose_idx,
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        ))
        continue

    visible_ranges = ranges[visible_mask]
    visible_bearings = np.arctan2(dy[visible_mask], dx[visible_mask]) - psi
    visible_bearings = (visible_bearings + np.pi) % (2.0 * np.pi) - np.pi

    measured_ranges = visible_ranges + np.random.normal(0.0, range_noise_std, size=visible_ranges.shape)
    measured_bearings = visible_bearings + np.random.normal(0.0, bearing_noise_std, size=visible_bearings.shape)
    measured_bearings = (measured_bearings + np.pi) % (2.0 * np.pi) - np.pi

    measurements_per_pose.append((
        pose_idx,
        visible_landmark_ids,
        measured_ranges,
        measured_bearings,
    ))
    total_measurements += visible_landmark_ids.size

    

print(f"Generated {total_measurements} range-bearing measurements across {len(state)} poses")


# Plot a subset of measurement rays for readability.
pose_stride_for_plot = 200
max_rays_per_pose = 6
plotted_measurements = 0

for pose_idx in range(0, len(measurements_per_pose), pose_stride_for_plot):
    entry = measurements_per_pose[pose_idx]
    pose_id, landmark_ids, measured_ranges, measured_bearings = entry
    if landmark_ids.size == 0:
        continue

    pose_x, pose_y, pose_psi = state[pose_id, :3]
    n_rays = min(max_rays_per_pose, landmark_ids.size)

    for j in range(n_rays):
        meas_range = measured_ranges[j]
        meas_bearing = measured_bearings[j]

        end_x = pose_x + meas_range * np.cos(pose_psi + meas_bearing)
        end_y = pose_y + meas_range * np.sin(pose_psi + meas_bearing)
        plt.plot([pose_x, end_x], [pose_y, end_y], color="tab:green", alpha=0.25, linewidth=0.8)
        plotted_measurements += 1

print(f"Plotted {plotted_measurements} measurement rays")





plt.plot(state[:, 0], state[:, 1])
plt.scatter(landmarks[:, 0], landmarks[:, 1], s=18, c="tab:orange", label="landmarks")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Simulated Vehicle Trajectory")
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()