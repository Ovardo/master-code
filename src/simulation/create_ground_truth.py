import numpy as np
import os

# ---- helpers ---------------------------------------------------------------
def heading_from_xy(x, y):
    """Compute headings from parametric curve samples (finite differences)."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.arctan2(dy, dx)

def make_observations(poses_xy, landmarks, max_range=7.5):
    """For each pose index i, list landmark indices within max_range."""
    obs = {}
    for i, p in enumerate(poses_xy):
        dists = np.linalg.norm(landmarks - p, axis=1)
        obs[i] = np.where(dists <= max_range)[0].tolist()
    return obs

def sample_landmarks(n_landmarks, bounds, rng):
    """Uniform landmarks in axis-aligned bounds ((xmin,xmax),(ymin,ymax))."""
    (xmin, xmax), (ymin, ymax) = bounds
    xs = rng.uniform(xmin, xmax, size=n_landmarks)
    ys = rng.uniform(ymin, ymax, size=n_landmarks)
    return np.column_stack([xs, ys])

# =============================================================================
# 1) CIRCULAR PATH -> returns (poses, landmarks, observations)
# =============================================================================
def make_circle_data(
    radius=5.0,
    n_poses=20,
    n_landmarks=20,
    max_range=7.5,
    center=(0.0, 0.0),
    pose_seed=1,        # kept for symmetry; not used unless you later add noise here
    landmark_seed=42,
):
    # Parametric circle: angle t in [0, 2π)
    t = np.linspace(0, 2*np.pi, n_poses, endpoint=True)
    cx, cy = center
    x = cx + radius * np.cos(t)
    y = cy + radius * np.sin(t)
    #theta = heading_from_xy(x, y)          # tangent heading
    theta = t + np.pi/2                     # tangent heading (alternative)
    poses = [np.array([x[i], y[i], theta[i]]) for i in range(n_poses)]

    # Landmarks: random in a box around the circle
    margin = 0.75 * radius
    bounds = ((cx - radius - margin, cx + radius + margin),
              (cy - radius - margin, cy + radius + margin))
    rng = np.random.default_rng(landmark_seed)
    landmarks = sample_landmarks(n_landmarks, bounds, rng)

    observations = make_observations(np.column_stack([x, y]), landmarks, max_range)

    # Keep landmarks as list[np.array] for compatibility with your earlier code
    landmarks_list = [lm for lm in landmarks]
    return poses, landmarks_list, observations

# =============================================================================
# 2) FIGURE-EIGHT (Lemniscate of Gerono) -> returns (poses, landmarks, observations)
#    x = a sin t,  y = a sin t cos t   with t in [0, 2π)
# =============================================================================
def make_figure8_data(
    a=5.0,
    n_poses=25,
    n_landmarks=20,
    max_range=7.5,
    center=(0.0, 0.0),
    pose_seed=2,        # kept for symmetry
    landmark_seed=24,
):
    t = np.linspace(0, 2*np.pi, n_poses, endpoint=True)
    cx, cy = center

    # Lemniscate of Gerono centered at (cx, cy)
    x = cx + a * np.sin(t)
    y = cy + a * np.sin(t) * np.cos(t)
    theta = heading_from_xy(x, y)
    poses = [np.array([x[i], y[i], theta[i]]) for i in range(n_poses)]

    # Landmarks: box covering the path with a margin
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    margin = 0.5 * a
    bounds = ((xmin - margin, xmax + margin), (ymin - margin, ymax + margin))

    rng = np.random.default_rng(landmark_seed)
    landmarks = sample_landmarks(n_landmarks, bounds, rng)

    observations = make_observations(np.column_stack([x, y]), landmarks, max_range)

    landmarks_list = [lm for lm in landmarks]
    return poses, landmarks_list, observations


if __name__ == "__main__":
    #poses, landmarks, observations = make_circle_data(radius=7.5, n_poses=20, n_landmarks=20, max_range=7.5)
    poses, landmarks, observations = make_figure8_data(a=10, n_poses=30, n_landmarks=25, max_range=7.5)
    import matplotlib.pyplot as plt

    # Convert poses and landmarks to arrays
    poses_arr = np.vstack(poses)  # shape (n_poses, 3) columns: x,y,theta
    xs, ys, thetas = poses_arr[:, 0], poses_arr[:, 1], poses_arr[:, 2]
    landmarks_arr = np.vstack(landmarks) if len(landmarks) > 0 else np.empty((0, 2))

    # Set figure width to half the width of an A4 paper (A4 width = 8.27 inches)
    A4_WIDTH_INCH = 8.27
    fig_width = A4_WIDTH_INCH * (2/3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))

    # Plot pose positions and connect them
    ax.plot(xs, ys, '-o', color='tab:blue', label='x', linewidth=1.0, alpha=0.5)

    # Plot headings as lines
    scale = 1
    dx = np.cos(thetas) * scale
    dy = np.sin(thetas) * scale
    for i in range(len(xs)):
        # Add x-axis
        ax.plot([xs[i], xs[i] + dx[i]], [ys[i], ys[i] + dy[i]], color='tab:red', linewidth=1.5, label='_x-axis' if i == 0 else "")
        # Add y-axis (x-axis rotated 90 degrees)
        ax.plot([xs[i], xs[i] - dy[i]], [ys[i], ys[i] + dx[i]], color='tab:green', linewidth=1.5, label='_y-axis' if i == 0 else "")

    # Plot landmarks and lines to observed landmarks
    if landmarks_arr.size:
        ax.scatter(landmarks_arr[:, 0], landmarks_arr[:, 1], marker='x', color='tab:red', label='m')

        # Draw light gray lines from each pose to its measured landmarks
        for i, lm_indices in observations.items():
            px, py = xs[i], ys[i]
            for j in lm_indices:
                lx, ly = landmarks_arr[j]
                ax.plot([px, lx], [py, ly], color='gray', linewidth=0.8, alpha=0.3, zorder=0, label='observations' if i == 0 and j == lm_indices[0] else "")

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Poses and Landmarks')
    ax.legend()
    ax.grid(True)
    plt.show()
    # out_path = os.path.join(os.path.dirname(__file__), "data_circle_zoomed.pdf")
    # fig.savefig(out_path, dpi=300, bbox_inches="tight")
    # print(f"Saved plot to {out_path}")