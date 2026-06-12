from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry as sg
from shapely.ops import unary_union

from master_code.logger import SlamLogger
from master_code.config import SlamConfig
from master_code.plotting.plotting_funcs import confidence_ellipse_2d, plot_association, plot_estimate
from master_code.utils import rotmat2
from master_code.plotting.thesis_style import apply_thesis_style
from master_code.paths import FIGURES_ROOT


def extract_polygon_slices(grid_2d: np.ndarray) -> list[sg.Polygon]:
    """Convert adjacent points in a mapped surface grid into quadrilaterals."""
    p_a = grid_2d[:-1, :-1]
    p_b = grid_2d[:-1, 1:]
    p_c = grid_2d[1:, 1:]
    p_d = grid_2d[1:, :-1]
    quads = np.stack((p_a, p_b, p_c, p_d), axis=2).reshape(-1, 4, 2)

    return [
        sg.Polygon(quad).buffer(0.0001, cap_style=2, join_style=2)
        for quad in quads
    ]


def plot_pose_covariance_in_body_frame(
    ax,
    mean: np.ndarray,
    covariance: np.ndarray,
    n: int = 50,
    chi2_val: float = 11.345,
    right_perturbation: bool = True,
    **kwargs,
) -> None:
    """Project an SE(2) tangent-space confidence ellipsoid into body-frame translation."""
    eigvals, eigvecs = np.linalg.eigh(covariance)
    scales = np.sqrt(chi2_val * np.maximum(eigvals, 0.0))

    azimuth = np.linspace(0.0, 2.0 * np.pi, n + 1)
    polar = np.linspace(0.0, np.pi, n + 1)
    azimuth_grid, polar_grid = np.meshgrid(azimuth, polar)
    unit_sphere = np.vstack((
        (np.sin(polar_grid) * np.cos(azimuth_grid)).ravel(),
        (np.sin(polar_grid) * np.sin(azimuth_grid)).ravel(),
        np.cos(polar_grid).ravel(),
    ))
    tangent_points = eigvecs @ (scales[:, None] * unit_sphere)

    mean_pose = gtsam.Pose2(*mean)
    body_translations = np.empty((tangent_points.shape[1], 2))
    for i, tangent_point in enumerate(tangent_points.T):
        perturbation = gtsam.Pose2.Expmap(tangent_point)
        if right_perturbation:
            sample_pose = mean_pose.compose(perturbation)
        else:
            sample_pose = perturbation.compose(mean_pose)
        body_translations[i] = mean_pose.transformTo(sample_pose.translation())

    plot_grid = body_translations[:, [1, 0]].reshape(n + 1, n + 1, 2)
    region = unary_union(extract_polygon_slices(plot_grid))
    if isinstance(region, sg.Polygon):
        polygons = [region]
    else:
        polygons = [
            geometry
            for geometry in region.geoms
            if isinstance(geometry, sg.Polygon)
        ]

    fill_alpha = kwargs.pop("fill_alpha", 0.0)
    color = kwargs.get("color", "black")
    for polygon in polygons:
        boundary = np.asarray(polygon.exterior.coords)
        ax.fill(boundary[:, 0], boundary[:, 1], color=color, alpha=fill_alpha)
        ax.plot(boundary[:, 0], boundary[:, 1], **kwargs)



# run = Path('/Users/ovar/Documents/Master/master_code/runs/sim/20260607_225558') # 357 (normal)
# run = Path('/Users/ovar/Documents/Master/master_code/runs/sim/20260607_182625') # 228, 229, 230
run = Path('/Users/ovar/Documents/Master/master_code/runs/real/20260607_120820_all') # real
COMPARISON_STEPS = [5194, 5195, 5196]

SCALE = 1.0
RANGE = 50.0
BEARING = 105.0

apply_thesis_style()

cfg = SlamConfig.load(run / "config.yaml")

# 3x2 grid: rows are consecutive steps, columns are the two views.
COLUMN_TITLES = ("MAP estimate", "Measurement space")

fig, axes = plt.subplots(3, 2, figsize=(8.5, 12), tight_layout=True)

# Linear map reused for every step.
range_bearing_to_plot = np.array([
    [0.0, np.rad2deg(1.0)],
    [1.0, 0.0],
])

for row, step in enumerate(COMPARISON_STEPS):
    snap = SlamLogger.load_snapshot(run, step)
    diag = SlamLogger.load_association_diagnostics(run, step)

    pose      = diag.get("pose")
    z         = diag.get("measurements")
    z_pred    = diag.get("predicted_measurements")
    assoc     = diag.get("association")
    local_lms = diag.get("local_landmarks")
    cov_Q     = diag.get("prior_joint_covariance")
    cov_S     = diag.get("innovation_covariance")

    if cov_Q is None:
        raise RuntimeError(
            "This association diagnostic predates prior_joint_covariance logging. "
            f"Rerun SLAM for step {step} to plot prior body-space covariances."
        )

    map_ax  = axes[row, 0]
    meas_ax = axes[row, 1]

    # --- Column 0: full global MAP estimate with marginal covariances ---
    plot_estimate(
        ax=map_ax,
        poses=snap["poses"],
        poses_cov=snap["poses_covariance"],
        poses_cov_stride=20,
        landmarks=snap["landmarks"],
        landmarks_cov=snap["landmarks_covariance"],
        title=None,
        show_legend=False,
    )

    # --- Column 2: measurement space ---
    z_range = z[:, 0]
    z_bearing = z[:, 1]
    z_pred_range = z_pred[:, 0]
    z_pred_bearing = z_pred[:, 1]

    for j, pred in enumerate(z_pred):
        cov_rb = cov_S[2*j : 2*j+2, 2*j : 2*j+2]
        cov_plot = range_bearing_to_plot @ cov_rb @ range_bearing_to_plot.T
        center = np.array([np.rad2deg(pred[1]), pred[0]])
        meas_ax.add_patch(confidence_ellipse_2d(center, cov_plot, fc="none", ec="tomato", confidence=cfg.association.alpha_individual, alpha=1, lw=1, zorder=2))

    for measurement, predicted_index in zip(z, assoc):
        if predicted_index < 0 or predicted_index >= len(z_pred):
            continue
        predicted = z_pred[predicted_index]
        meas_ax.plot(
            np.rad2deg([measurement[1], predicted[1]]),
            [measurement[0], predicted[0]],
            color="green",
            lw=0.8,
            alpha=0.8,
            zorder=3,
        )

    meas_ax.scatter(np.rad2deg(z_bearing), z_range, c="steelblue", s=30, lw=0.8, marker='x', label="Measurement", zorder=4)
    meas_ax.scatter(np.rad2deg(z_pred_bearing), z_pred_range, c="tomato", s=3, label="Predicted", zorder=5)
    meas_ax.set_xlabel("bearing [deg]")
    meas_ax.set_ylabel("range [m]")
    meas_ax.set_xlim(BEARING, -BEARING)
    meas_ax.set_ylim(0, RANGE)
    meas_ax.set_xticks(np.arange(-BEARING, BEARING + 1, 30))
    meas_ax.set_yticks(np.arange(0, RANGE + 1, 10))
    meas_ax.grid(True, linewidth=0.5, alpha=0.5)

    # --- Column 0 overlay: highlight the local landmarks and back-project this
    # step's measurements onto the estimated map (mirrors create_video.py). ---
    R_W_B = rotmat2(pose[2])
    meas_body = np.column_stack((z_range * np.cos(z_bearing), z_range * np.sin(z_bearing)))
    meas_world = pose[:2] + meas_body @ R_W_B.T
    is_associated = assoc >= 0

    for measurement_index, predicted_index in enumerate(assoc):
        if predicted_index < 0 or predicted_index >= len(local_lms):
            continue
        map_ax.plot(
            [meas_world[measurement_index, 0], local_lms[predicted_index, 0]],
            [meas_world[measurement_index, 1], local_lms[predicted_index, 1]],
            color="green",
            lw=0.8,
            alpha=0.8,
            zorder=5,
        )

    local_handle = map_ax.scatter(local_lms[:, 0], local_lms[:, 1], s=40, marker="x", c="orange", lw=0.9, label="Local landmarks", zorder=6)
    associated_handle = map_ax.scatter(meas_world[is_associated, 0], meas_world[is_associated, 1], s=12, marker=".", c="steelblue", label="Associated measurement", zorder=7)
    unassociated_handle = map_ax.scatter(meas_world[~is_associated, 0], meas_world[~is_associated, 1], s=12, marker=".", c="dimgray", label="Unassociated measurement", zorder=7)

    if row == 0:
        map_ax.legend(handles=[local_handle, associated_handle, unassociated_handle], fontsize=7, loc="best")

    # Highlight the large loop-closure measurement at step 5195.
    if step == 5195:
        map_ax.annotate(
            "Large loop closure",
            xy=(-125.0, -60.0),
            xytext=(-20, 50),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )
        meas_ax.annotate(
            "Large loop closure",
            xy=(37.0, 37.0),
            xytext=(-20, -50),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )

# Column headers on the top row.
axes[0, 0].set_title(COLUMN_TITLES[0])
axes[0, 1].set_title(COLUMN_TITLES[1])

fig.tight_layout(h_pad=3.0)
fig.canvas.draw()

# Centered "Step N" title above each row, spanning both columns.
for row, step in enumerate(COMPARISON_STEPS):
    pos_left = axes[row, 0].get_position()
    pos_right = axes[row, 1].get_position()
    center_x = 0.5 * (pos_left.x0 + pos_right.x1)
    top_y = max(pos_left.y1, pos_right.y1)
    offset = 0.035 if row == 0 else 0.012
    fig.text(center_x, top_y + offset, f"Step {step}", ha="center", va="bottom", fontsize=12, fontweight="bold")

fig.savefig(
    FIGURES_ROOT / f"real_association_grid_steps_{COMPARISON_STEPS[0]}_{COMPARISON_STEPS[-1]}.pdf",
    dpi=200,
    bbox_inches="tight",
)

plt.show()
