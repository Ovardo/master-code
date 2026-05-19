import warnings
from dataclasses import dataclass
from typing import Any

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import visgeom as vg
from shapely import geometry as sg
from shapely import ops as so


@dataclass
class MultivariateNormalParameters:
    mean: Any
    covariance: np.ndarray


def plot_result(
    ax,
    poses,
    landmarks,
    poses_gt=None,
    gt_landmarks=None,
    sample_points=False,
    exact_map=True,
):
    # matplotlib.use('qt5agg')

    num_draws = 1000

    # Plot pose marginals.
    num_poses = len(poses)
    pose_cmap = plt.get_cmap("autumn")
    for dist, color in zip(
        poses, pose_cmap([number / num_poses for number in range(num_poses)]),
    ):
        if isinstance(dist.mean, gtsam.Pose2):
            mean_translation = dist.mean.translation()
        else:
            mean_translation = dist.mean[0:2]
        if (
            exact_map
        ):  # Plot covariance from manifold (takes rotation into consideration)
            plot_se2_covariance_on_manifold_gtsam(
                ax, dist, fill_alpha=0.1, fill_color="r",
            )
        else:  # Plot covariance of translation (marginalizes out rotation, i.e disregard it).
            dist = MultivariateNormalParameters(
                mean=mean_translation, covariance=dist.covariance[0:2, 0:2],
            )
            plot_ellipse(
                ax, dist, fill_alpha=0.1, fill_color="r",
            )  # 5.99146454711 approx 95% for 2D

        # Plot random points from distribution.
        if sample_points:
            if isinstance(dist.mean, gtsam.Pose2):
                random_points = np.random.multivariate_normal(
                    np.zeros(3), dist.covariance, num_draws,
                ).T
                random_translations = np.zeros([2, num_draws])
                for i in range(num_draws):
                    random_translations[:, i] = dist.mean.compose(
                        gtsam.Pose2.Expmap(random_points[:, i]),
                    ).translation()
                ax.plot(
                    random_translations[0, :],
                    random_translations[1, :],
                    marker=".",
                    markeredgewidth=0,
                    color=color,
                    linestyle="",
                    alpha=0.2,
                )
            else:
                print(
                    "Warning: Sampling points for pose marginals only implemented for gtsam.Pose2 means.",
                )
        # Plot mean.
        ax.plot(
            mean_translation[0],
            mean_translation[1],
            marker="o",
            color="r",
            linestyle="",
        )

    # Plot landmark marginals.
    num_landmarks = len(landmarks)
    landmark_cmap = plt.get_cmap("summer")
    for dist, color in zip(
        landmarks,
        landmark_cmap([number / num_landmarks for number in range(num_landmarks)]),
    ):
        plot_ellipse(ax, dist, fill_alpha=0.1, fill_color="b")
        ax.plot(dist.mean[0], dist.mean[1], marker="o", color="b", linestyle="")

        if sample_points:
            random_points = np.random.multivariate_normal(
                dist.mean, dist.covariance, num_draws,
            ).T
            ax.plot(
                random_points[0, :],
                random_points[1, :],
                marker=".",
                markeredgewidth=0,
                color=color,
                linestyle="",
                alpha=0.2,
            )

            ax.plot(dist.mean[0], dist.mean[1], marker="+", color="k", linestyle="")

    # Optionally plot ground truth landmarks
    if gt_landmarks is not None:
        ax.plot(
            gt_landmarks[0, :],
            gt_landmarks[1, :],
            marker="x",
            color="b",
            linestyle="",
            markersize=8,
        )

    # Optionally plot only the last ground truth pose
    if poses_gt is not None and len(poses_gt) > 0:
        last_pose = poses_gt[-1]
        last_translation = last_pose.translation()
        ax.plot(
            last_translation[0],
            last_translation[1],
            marker="x",
            color="g",
            linestyle="",
            markersize=10,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid("on")

    # plt.show()


# Code retrived and adapted from: https://github.com/borglab/gtsam)
def plot_se2_covariance_on_manifold_gtsam(
    ax,
    dist,
    n=50,
    chi2_val=7.815,
    right_perturbation=True,  # 99: 11.345, 95: 7.815,
    fill_alpha=0.0,
    fill_color="lightsteelblue",
    linestyle="-",
    axis_length: float = 0.5,
    axis_color: str = "k",
):
    u, s, _ = np.linalg.svd(dist.covariance)
    scale = np.sqrt(chi2_val * s)

    x, y, z = vg.utils.generate_ellipsoid(n, pose=(u, np.zeros([3, 1])), scale=scale)

    tangent_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    num_samples = tangent_points.shape[1]
    transl_points = np.zeros([num_samples, 2])

    if right_perturbation:
        for i in range(num_samples):
            transl_points[i, :] = dist.mean.compose(
                gtsam.Pose2.Expmap(tangent_points[:, i]),
            ).translation()
    else:
        for i in range(num_samples):
            transl_points[i, :] = (
                gtsam.Pose2.Expmap(tangent_points[:, i])
                .compose(dist.mean)
                .translation()
            )

    p_grid = np.reshape(transl_points, [(n + 1), (n + 1), 2])
    polygons = extract_polygon_slices(p_grid)
    union = so.unary_union(polygons)
    if union.geom_type != "Polygon":
        warnings.warn("Could not find a closed boundary", RuntimeWarning)
        return

    ax.fill(*union.exterior.xy, alpha=fill_alpha, facecolor=fill_color)
    ax.plot(*union.exterior.xy, color=fill_color, linewidth=1, linestyle=linestyle)


def plot_pose2_on_axes(
    ax,
    pose,
    axis_length: float = 0.25,
    show_axis: bool = False,
    linestyle: str = "-",
    color: str = "black",
    marker="",
):
    ax.plot(
        pose.x(),
        pose.y(),
        marker=marker,
        linestyle=linestyle,
        color=color,
        markersize=6,
    )

    # Optionally draw pose axes based on dist.mean.rotation
    if show_axis:
        try:
            if isinstance(pose, gtsam.Pose2):
                cx, cy = pose.x(), pose.y()
                th = pose.theta()
                c, s = np.cos(th), np.sin(th)
                # Body x-axis (forward)
                x_end = (cx + axis_length * c, cy + axis_length * s)
                # Body y-axis (left)
                y_end = (cx - axis_length * s, cy + axis_length * c)
                ax.plot(
                    [cx, x_end[0]],
                    [cy, x_end[1]],
                    color="r",
                    linewidth=0.8,
                    linestyle="-",
                )
                ax.plot(
                    [cx, y_end[0]],
                    [cy, y_end[1]],
                    color="g",
                    linewidth=0.8,
                    linestyle="-",
                )
            else:
                # If mean is not Pose2, skip axis drawing
                pass
        except Exception:
            # Avoid breaking plotting due to axes drawing
            pass


def plot_pose2_trajectory(
    ax,
    poses,
    color="black",
    linestyle="-",
    linewidth=1.0,
    show_axes=False,
    axis_length=0.25,
):
    xs = [p.x() for p in poses]
    ys = [p.y() for p in poses]

    # trajectory line
    ax.plot(xs, ys, linestyle=linestyle, color=color, linewidth=linewidth)

    # current pose marker
    ax.plot(xs[-1], ys[-1], marker="o", color=color, markersize=4)

    if show_axes:
        p = poses[-1]
        th = p.theta()
        c, s = np.cos(th), np.sin(th)
        x, y = p.x(), p.y()

        ax.plot(
            [x, x + axis_length * c], [y, y + axis_length * s], color="r", linewidth=0.8,
        )
        ax.plot(
            [x, x - axis_length * s], [y, y + axis_length * c], color="g", linewidth=0.8,
        )


def extract_polygon_slices(grid_2d):
    p_a = grid_2d[:-1, :-1]
    p_b = grid_2d[:-1, 1:]
    p_c = grid_2d[1:, 1:]
    p_d = grid_2d[1:, :-1]

    quads = np.concatenate((p_a, p_b, p_c, p_d), axis=2)

    m, n, _ = grid_2d.shape
    quads = quads.reshape(((m - 1) * (n - 1), 4, 2))

    return [sg.Polygon(t).buffer(0.0001, cap_style=2, join_style=2) for t in quads]


def plot_ellipse(
    ax,
    dist,
    n=50,
    chi2_val=5.991,
    fill_alpha=0.0,
    fill_color="lightsteelblue",
    linestyle="-",
    linewidth=1.0,
):
    u, s, _ = np.linalg.svd(dist.covariance)
    scale = np.sqrt(chi2_val * s)

    theta = np.linspace(0, 2 * np.pi, n + 1)
    x = np.cos(theta)
    y = np.sin(theta)

    R = u
    t = np.reshape(dist.mean, [2, 1])
    circle_points = (R * scale) @ np.vstack((x.flatten(), y.flatten())) + t

    ax.fill(
        circle_points[0, :], circle_points[1, :], alpha=fill_alpha, facecolor=fill_color,
    )
    ax.plot(
        circle_points[0, :],
        circle_points[1, :],
        color=fill_color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
