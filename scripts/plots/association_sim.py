from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry as sg
from shapely.ops import unary_union

from master_code.logger import SlamLogger
from master_code.config import SlamConfig
from master_code.plotting.plotting_funcs import confidence_ellipse_2d, plot_association
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
step = 5194
SCALE = 1.0
RANGE = 50.0  

apply_thesis_style()

cfg = SlamConfig.load(run / "config.yaml")
diag = SlamLogger.load_association_diagnostics(run, step)

fig, axes = plt.subplots(1,2, figsize=(9, 4.5), tight_layout=True)


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
        "Rerun SLAM for this step to plot prior body-space covariances."
    )


# Measurement space
z_range = z[:, 0]
z_bearing = z[:, 1]
z_pred_range = z_pred[:, 0]
z_pred_bearing = z_pred[:, 1]

range_bearing_to_plot = np.array([
    [0.0, np.rad2deg(1.0)],
    [1.0, 0.0],
])
for j, pred in enumerate(z_pred):
    cov_rb = cov_S[2*j : 2*j+2, 2*j : 2*j+2]
    cov_plot = range_bearing_to_plot @ cov_rb @ range_bearing_to_plot.T
    center = np.array([np.rad2deg(pred[1]), pred[0]])
    axes[1].add_patch(confidence_ellipse_2d(center, cov_plot, fc="none", ec="tomato", confidence=cfg.association.alpha_individual, alpha=1, lw=1, zorder=2))

for measurement, predicted_index in zip(z, assoc):
    if predicted_index < 0 or predicted_index >= len(z_pred):
        continue
    predicted = z_pred[predicted_index]
    axes[1].plot(
        np.rad2deg([measurement[1], predicted[1]]),
        [measurement[0], predicted[0]],
        color="green",
        lw=0.8,
        alpha=0.8,
        zorder=3,
    )

axes[1].scatter(np.rad2deg(z_bearing), z_range, c="steelblue", s=30, lw=0.8, marker='x', label="Measurement", zorder=4)
axes[1].scatter(np.rad2deg(z_pred_bearing), z_pred_range, c="tomato", s=3, label="Predicted", zorder=5)
axes[1].set_xlabel("bearing [deg]")
axes[1].set_ylabel("range [m]")
axes[1].set_xlim(120, -120)
axes[1].set_ylim(0, 90)
axes[1].set_xticks(np.arange(-120, 121, 30))
axes[1].set_yticks(np.arange(0, 91, 10))
axes[1].grid(True, linewidth=0.5, alpha=0.5)
axes[1].set_title("Measurement space")

# Body space
z_x = z_range * np.cos(z_bearing)
z_y = z_range * np.sin(z_bearing)
z_pred_x = z_pred_range * np.cos(z_pred_bearing)
z_pred_y = z_pred_range * np.sin(z_pred_bearing)

# The prior joint covariance is ordered [pose, lm_0, lm_1, ...].
R_W_B = rotmat2(pose[2])
landmarks_body = (local_lms - pose[:2]) @ R_W_B
body_to_plot = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
])
for j, landmark_body in enumerate(landmarks_body):
    start = 3 + 2 * j
    pair_indices = np.r_[0:3, start:start + 2]
    pair_cov = cov_Q[np.ix_(pair_indices, pair_indices)]
    x_body, y_body = landmark_body
    H_pose = np.array([
        [-1.0, 0.0, y_body],
        [0.0, -1.0, -x_body],
    ])
    H_landmark = R_W_B.T
    H_relative = np.hstack((H_pose, H_landmark))
    cov_body = H_relative @ pair_cov @ H_relative.T
    cov_plot = body_to_plot @ cov_body @ body_to_plot.T
    axes[0].add_patch(confidence_ellipse_2d(body_to_plot @ landmark_body, cov_plot,fc="none", ec="tomato", alpha=0.7, lw=0.8, zorder=2, scale=SCALE))

cov_Q[:3,:3] *= SCALE**2

plot_pose_covariance_in_body_frame(
    axes[0],
    pose,
    cov_Q[:3, :3],
    color="black",
    alpha=0.8,
    lw=1.0,
    zorder=4,
)
for measurement_index, predicted_index in enumerate(assoc):
    if predicted_index < 0 or predicted_index >= len(z_pred):
        continue
    axes[0].plot(
        [z_y[measurement_index], z_pred_y[predicted_index]],
        [z_x[measurement_index], z_pred_x[predicted_index]],
        color="green",
        lw=0.8,
        alpha=0.8,
        zorder=3,
    )

axes[0].scatter(z_y, z_x, c="steelblue", s=10, marker='x', label="Measurement", zorder=5)
axes[0].scatter(z_pred_y, z_pred_x, c="tomato", s=1, label="Predicted")
# axes[0].scatter(0, 0, c="black", s=80, marker="^", label="Robot", zorder=6)
axes[0].set_aspect("equal")
axes[0].set_xlabel("y [m]")
axes[0].set_ylabel("x [m]")
axes[0].set_xlim(RANGE, -RANGE)
axes[0].set_ylim(-RANGE, RANGE)
axes[0].set_title("Body space")
axes[0].patch.set_alpha(0)

# Finalize the subplot positions before placing the manual polar overlay.
fig.tight_layout()
fig.canvas.draw()
axp = fig.add_axes(axes[0].get_position().bounds, polar=True, frameon=False)
axp.set_theta_zero_location("N")
axp.set_theta_direction(1)
theta_degrees = np.arange(-180, 180, 30)
axp.set_thetagrids(theta_degrees % 360, labels=[f"{theta}°" for theta in theta_degrees])
axp.set_rlim(0, RANGE)
axp.patch.set_alpha(0)
axp.tick_params(axis="both", labelbottom=False, labelleft=False)
axp.set_zorder(-5)
axp.grid(zorder=-5)

fig.savefig(FIGURES_ROOT / f"sim_association_step_{step}.pdf", dpi=200, bbox_inches="tight")

# One-off thesis figure comparing consecutive scans in measurement space.
comparison_steps = [227, 228, 229]
comparison_fig, comparison_axes = plt.subplots(
    1,
    len(comparison_steps),
    figsize=(12, 4),
    sharex=True,
    sharey=True,
    tight_layout=True,
)

for panel_index, (comparison_step, ax) in enumerate(zip(comparison_steps, comparison_axes)):
    comparison_diag = SlamLogger.load_association_diagnostics(run, comparison_step)
    comparison_z = comparison_diag["measurements"]
    comparison_z_pred = comparison_diag["predicted_measurements"]
    comparison_assoc = comparison_diag["association"]
    comparison_cov_S = comparison_diag["innovation_covariance"]

    for j, predicted in enumerate(comparison_z_pred):
        cov_rb = comparison_cov_S[2*j : 2*j+2, 2*j : 2*j+2]
        cov_plot = range_bearing_to_plot @ cov_rb @ range_bearing_to_plot.T
        center = np.array([np.rad2deg(predicted[1]), predicted[0]])
        ax.add_patch(
            confidence_ellipse_2d(
                center,
                cov_plot,
                fc="none",
                ec="tomato",
                confidence=cfg.association.alpha_individual,
                alpha=1,
                lw=1,
                zorder=2,
            )
        )

    for measurement, predicted_index in zip(comparison_z, comparison_assoc):
        if predicted_index < 0 or predicted_index >= len(comparison_z_pred):
            continue
        predicted = comparison_z_pred[predicted_index]
        ax.plot(
            np.rad2deg([measurement[1], predicted[1]]),
            [measurement[0], predicted[0]],
            color="green",
            lw=0.8,
            alpha=0.8,
            zorder=3,
        )

    ax.scatter(
        np.rad2deg(comparison_z[:, 1]),
        comparison_z[:, 0],
        c="steelblue",
        s=30,
        lw=0.8,
        marker="x",
        label="Measurement",
        zorder=4,
    )
    ax.scatter(
        np.rad2deg(comparison_z_pred[:, 1]),
        comparison_z_pred[:, 0],
        c="tomato",
        s=3,
        label="Predicted",
        zorder=5,
    )
    ax.set_xlabel("bearing [deg]")
    if panel_index == 0:
        ax.set_ylabel("range [m]")
    ax.set_xlim(120, -120)
    ax.set_ylim(0, 90)
    ax.set_xticks(np.arange(-120, 121, 30))
    ax.set_yticks(np.arange(0, 91, 10))
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_title(f"Step {comparison_step}")

    # Highlight the unassociated measurement near (bearing -70 deg, range 70 m).
    if comparison_step == 228:
        bearings_deg = np.rad2deg(comparison_z[:, 1])
        ranges = comparison_z[:, 0]
        unassociated = np.where(comparison_assoc < 0)[0]
        target = unassociated[
            np.argmin(
                (bearings_deg[unassociated] + 70.0) ** 2
                + (ranges[unassociated] - 70.0) ** 2
            )
        ]
        ax.annotate(
            "Unassociated\nmeasurement",
            xy=(bearings_deg[target], ranges[target]),
            xytext=(-55.0, 55.0),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )

comparison_fig.savefig(
    FIGURES_ROOT / "sim_association_measurement_steps_227_229.pdf",
    dpi=200,
    bbox_inches="tight",
)

plt.show()
