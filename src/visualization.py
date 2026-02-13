from typing import Optional

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import X
from scipy.stats.distributions import chi2

from association import NIS
from config import VisualizationConfig
from factor_graph_slam import FactorGraphSLAM
from result import SLAMHistory
from utils.utils_gtsam import pose2_to_array
from utils.utils_math import ssa


class SLAMVisualizer:
    """Handle SLAM visualization"""

    def __init__(self, config: VisualizationConfig, history: SLAMHistory,):
        self.cfg = config
        self.history = history

    def plot_measurement_space(
        self,
        step: int,
        show_lines: bool = True,
        show_labels: bool = True,
        figsize=(7, 5),
    ):

        rec = self.history.get_or_raise(step)

        if rec.measurements is None:
            raise ValueError(f"No measurements stored for step={step}")

        z = np.array(
            [[r, b.theta()] for (r, b) in rec.measurements], dtype=float
        )  # (M,2)

        zhat = rec.predicted_measurements
        if zhat is None:
            zhat = np.empty((0, 2), dtype=float)
        else:
            zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)

        assoc = rec.associations if rec.associations is not None else []
        local_ids = getattr(rec, "local_landmark_ids", None)

        fig, ax = plt.subplots(figsize=figsize)

        # predicted + measured
        if len(zhat) > 0:
            ax.scatter(zhat[:, 0], zhat[:, 1], marker="o", label="predicted")
        if len(z) > 0:
            ax.scatter(z[:, 0], z[:, 1], marker="x", label="measured")

        # association lines measured -> predicted
        if (
            show_lines
            and local_ids is not None
            and len(local_ids) == len(zhat)
            and len(assoc) == len(z)
        ):
            id_to_i = {lm_id: i for i, lm_id in enumerate(local_ids)}
            i_to_id = {i: lm_id for i, lm_id in enumerate(local_ids)}

            assoc_arr = np.asarray(assoc, dtype=int)
            new_mask = assoc_arr == -1

            for j, a_j in enumerate(assoc_arr):
                if a_j == -1:
                    continue
                i = id_to_i.get(int(a_j), None)
                if i is None:
                    continue

                ax.plot(
                    [z[j, 0], zhat[i, 0]], [z[j, 1], zhat[i, 1]], linewidth=1, alpha=0.6
                )

                if show_labels:
                    ax.text(z[j, 0], z[j, 1], f"{a_j}", fontsize=8, alpha=0.8)

            # mark unassociated
            if np.any(new_mask):
                ax.scatter(
                    z[new_mask, 0],
                    z[new_mask, 1],
                    marker="x",
                    label="unassociated (-1)",
                )

        ax.set_title(f"Measurement space (step {step})")
        ax.set_xlabel("range [m]")
        ax.set_ylabel("bearing [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig, ax

    def plot_NIS(self, figsize=(13, 3), ax=None, show_expected=True):

        steps = list(self.history.steps)
        N = len(steps)

        nis_sequence = np.full(N, np.nan, dtype=float)
        dof_sequence = np.zeros(N, dtype=int)
        lower_bounds = np.full(N, np.nan, dtype=float)
        upper_bounds = np.full(N, np.nan, dtype=float)

        for k, step in enumerate(steps):
            if step == 0:
                continue  # skip first step (no measurements)
            rec = self.history.get_or_raise(step)

            if rec.innovation_covariance is None:
                raise ValueError(f"No innovation covariance stored for step={step}")

            S = rec.innovation_covariance
            z = rec.measurements
            zhat = rec.predicted_measurements

            assoc = np.array(rec.associations_local)
            z = np.array([[r, b.theta()] for (r, b) in z], dtype=float)  # (M,2)

            # number of associated landmark measurements (each landmark gives 2D measurement)
            num_assoc = np.sum(assoc > -1)
            dof = 2 * num_assoc

            dof_sequence[k] = dof

            # If no associations, NIS is not meaningful (0 dof -> chi2 not defined nicely)
            if dof <= 0:
                continue

            nis_sequence[k] = NIS(z, zhat, S, assoc)

            lower, upper = chi2.interval(0.999999, df=dof) # TODO fix hardcoding 
            lower_bounds[k] = lower
            upper_bounds[k] = upper

        # ---- plotting ----
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(N)

        ax.plot(x, nis_sequence, label="NIS", linewidth=1.8)
        ax.plot(
            x,
            lower_bounds,
            "--",
            label=r"$\chi^2_{{dof},1-\alpha_{joint}}$",
            linewidth=1.2,
        )
        ax.plot(
            x,
            upper_bounds,
            "--",
            label=r"$\chi^2_{{dof},\alpha_{joint}}$",
            linewidth=1.2,
        )

        if show_expected:
            # E[chi2(dof)] = dof
            expected = np.where(dof_sequence > 0, dof_sequence.astype(float), np.nan)
            ax.plot(x, expected, ":", label="E[NIS] = dof", linewidth=1.2)

        ax.set_title("NIS consistency over time")
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("NIS")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def plot_NEES(self, gt_poses):
        steps = list(self.history.steps)
        N = len(steps)

        nees_sequence = np.full(N, np.nan, dtype=float)
        lower_bounds = np.full(N, np.nan, dtype=float)
        upper_bounds = np.full(N, np.nan, dtype=float)

        dof = 3  # Pose2 minimal dimension
        alpha = 0.95

        for k, step in enumerate(steps):
            rec = self.history.get_or_raise(step)
            est = rec.estimate
            cov = rec.cov_last_pose

            if est is None:
                continue

            if step >= len(gt_poses):
                continue

            pose_est = est.atPose2(X(step))
            pose_gt = gt_poses[step]

            error = pose2_to_array(pose_est.between(pose_gt))  # in minimal coordinates

            nees_sequence[k] = error.T @ np.linalg.inv(cov) @ error

            lower, upper = chi2.interval(alpha, df=dof)
            lower_bounds[k] = lower
            upper_bounds[k] = upper

        # ---- plotting ----
        fig, ax = plt.subplots(figsize=(13, 3))
        x = np.arange(N)
        ax.plot(x, nees_sequence, label="NEES", linewidth=1.8)
        ax.plot(
            x, lower_bounds, "--", label=f"Lower bound (α={alpha:g})", linewidth=1.2
        )
        ax.plot(
            x, upper_bounds, "--", label=f"Upper bound (α={alpha:g})", linewidth=1.2
        )
        ax.set_title("NEES consistency over time")
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("NEES")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax

    def plot_error(self, gt_poses):

        steps = list(self.history.steps)
        N = len(steps)

        # errors (x, y, theta) and sigmas
        err = np.full((N, 3), np.nan, dtype=float)
        sig = np.full((N, 3), np.nan, dtype=float)

        for k, step in enumerate(steps):
            rec = self.history.get_or_raise(step)
            est = rec.estimate
            cov = rec.cov_last_pose  # expected 3x3 in (x,y,theta) minimal coords

            if est is None or cov is None:
                continue
            if step >= len(gt_poses):
                continue

            pose_est = est.atPose2(X(step))
            pose_gt = gt_poses[step]

            # Minimal error coordinates: Pose2 "between" -> (dx, dy, dtheta)
            e = pose2_to_array(pose_est.between(pose_gt))
            e[2] = ssa(e[2])

            err[k, :] = e
            sig[k, :] = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

        # ---- plotting ----
        labels = ["x error [m]", "y error [m]", "yaw error [rad]"]
        fig, axs = plt.subplots(3, 1, figsize=(13, 5.5), sharex=True)

        x = np.arange(N)
        for i, ax in enumerate(axs):
            ax.plot(x, err[:, i], linewidth=1.6, label="Error")

            # envelopes
            ax.fill_between(x, -2 * sig[:, i], 2 * sig[:, i], alpha=0.25, label="±2σ")
            ax.fill_between(x, -3 * sig[:, i], 3 * sig[:, i], alpha=0.15, label="±3σ")

            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axs[-1].set_xlabel("Timestep index")
        fig.suptitle("Pose estimation error with 2σ/3σ covariance envelopes", y=0.98)
        fig.tight_layout()
        return fig, axs

    def plot_result_step(
        self,
        step: int,
        marginals: Optional[gtsam.Marginals] = None,
        poses_gt: Optional[list[gtsam.Pose2]] = None,
        landmarks_gt: Optional[list[gtsam.Point2]] = None,
        poses_dead_reckoning: Optional[list[gtsam.Pose2]] = None,
        show_covariances: bool = True,
        show_landmarks: bool = True,
        axis_length: float = 0.5,
        figsize=(22, 6),
        ax=None,
        title: Optional[str] = None,
        show_orientations: bool = True,
    ):
        """
        Plot estimate at a given step using history (StepRecord).

        Notes on covariances:
          - `marginals` must correspond to the same (graph, values) solution.
          - If you pass the current slam.get_marginals(), it usually corresponds to the final step.
        """
        from gtsam.utils import plot as gtsam_plot

        from utils.utils_plot import (
            MultivariateNormalParameters,
            plot_ellipse,
            plot_se2_covariance_on_manifold_gtsam,
        )

        rec = self.history.get(step)
        
        if rec is None:
            raise ValueError(f"No record for step={step}")
        
        est = rec.estimates

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        ax.set_aspect("equal")
        if title is None:
            title = f"SLAM result at step {step}"
            if show_covariances and marginals is not None:
                title += " (with marginals)"
        ax.set_title(title)

        # ----- Plot estimated poses up to step -----
        x_coords = []
        y_coords = []
        for k in range(step + 1):
            pose_key = X(k)
            if not est.exists(pose_key):
                continue
            pose = est.atPose2(pose_key)
            x_coords.append(pose.x())
            y_coords.append(pose.y())
        ax.plot(x_coords, y_coords, "-r", label=r"$\hat{x}$")

        for k in range(step + 1):
            pose_key = gtsam.symbol('x', k)
            if not est.exists(pose_key):
                continue

            pose = est.atPose2(pose_key)

            if show_covariances and (marginals is not None):
                try:
                    cov = marginals.marginalCovariance(pose_key)
                    plot_se2_covariance_on_manifold_gtsam(
                        ax,
                        dist=MultivariateNormalParameters(mean=pose, covariance=cov),
                        fill_alpha=0.2,
                        fill_color="red",
                        linestyle="none",
                    )
                    # plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, show_axis=show_orientations)
                    # gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, covariance=cov)
                except Exception:
                    gtsam_plot.plot_pose2_on_axes(
                        ax, pose=pose, axis_length=axis_length
                    )
            else:
                gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)

        # ----- Plot estimated landmarks (that exist in this estimate) -----
        if show_landmarks:
            # Count how many landmarks exist in this estimate
            est_landmark_count = 0
            for lm_key in slam.landmark_keys:
                if not est.exists(lm_key):
                    continue

                lm_pos = est.atPoint2(lm_key)
                est_landmark_count += 1

                if show_covariances and (marginals is not None):
                    try:
                        cov = marginals.marginalCovariance(lm_key)
                        ax.plot(lm_pos[0], lm_pos[1], "ob")
                        plot_ellipse(
                            ax,
                            MultivariateNormalParameters(mean=lm_pos, covariance=cov),
                            fill_alpha=0.2,
                            fill_color="blue",
                            linestyle="",
                            linewidth=0.8,
                        )
                        # gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b", P=cov)
                    except Exception:
                        gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b")
                else:
                    gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b")

            # Add a legend entry indicating the number of estimated landmarks
            try:
                ax.plot([], [], "ob", label=f"$\\hat{{m}}$ (#{est_landmark_count})")
            except Exception:
                pass

        # ----- Optional: overlay GT on same axes -----
        if poses_gt is not None:
            for pose in poses_gt[: step + 1]:
                plot_pose2_on_axes(
                    ax, pose=pose, axis_length=axis_length, marker="x", color="green"
                )
                # gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)
            ax.plot([], [], "gx", label="$x_{GT}$")

        if landmarks_gt is not None:
            for lm_pos in landmarks_gt:
                ax.plot(lm_pos[0], lm_pos[1], "x", color="orange")
                # gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="gx")
            ax.plot([], [], "x", color="orange", label=r"$m_{GT}$")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        return fig, ax

    @staticmethod
    def plot_final_result(
        slam,
        marginals: Optional[gtsam.Marginals] = None,
        poses_gt: Optional[list[gtsam.Pose2]] = None,
        landmarks_gt: Optional[list[gtsam.Point2]] = None,
        poses_dead_reckoning: Optional[list[gtsam.Pose2]] = None,
        **kwargs,
    ):
        if len(slam.history) == 0:
            raise ValueError("No history in slam.history")
        last_step = slam.history.steps[-1]
        return SLAMVisualizer.plot_result_step(
            slam,
            step=last_step,
            marginals=marginals,
            poses_dead_reckoning=poses_dead_reckoning,
            poses_gt=poses_gt,
            landmarks_gt=landmarks_gt,
            **kwargs,
        )

    # @staticmethod
    # def plot_final_result(slam: FactorGraphSLAM,
    #                      marginals: Optional[gtsam.Marginals] = None,
    #                      figsize=(22, 6)):
    #     """Plot final SLAM result with covariances"""
    #     import matplotlib.pyplot as plt
    #     from gtsam.utils import plot as gtsam_plot

    #     if marginals is None:
    #         marginals = slam.get_marginals()

    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     ax.set_aspect('equal')
    #     ax.set_title("Nonlinear 2D SLAM with Marginals")

    #     # Plot poses
    #     for k in range(slam.num_poses):
    #         pose_key = X(k)
    #         pose = slam.values.atPose2(pose_key)
    #         cov = marginals.marginalCovariance(pose_key)
    #         gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5, covariance=cov)

    #     # Plot landmarks
    #     for lm_key in slam.landmark_keys:
    #         lm_pos = slam.values.atPoint2(lm_key)
    #         cov = marginals.marginalCovariance(lm_key)
    #         gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='b', P=cov)

    #     plt.tight_layout()
    #     return fig, ax

    # @staticmethod
    # def plot_ground_truth(poses_gt: list[gtsam.Pose2],
    #                       landmarks_gt: list[gtsam.Point2],
    #                       figsize=(22, 6)):
    #     """Plot ground truth trajectory and landmarks"""
    #     import matplotlib.pyplot as plt
    #     from gtsam.utils import plot as gtsam_plot
    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     ax.set_aspect('equal')
    #     ax.set_title("Ground Truth Trajectory and Landmarks")
    #     # Plot ground truth poses
    #     for k, pose in enumerate(poses_gt):
    #         gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5)
    #     # Plot ground truth landmarks
    #     for lm_pos in landmarks_gt:
    #         gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='go')
    #     plt.tight_layout()
    #     return fig, ax

    @staticmethod
    def plot_step_by_step(
        slam,
        subplot_size: float = 4.0,
        axis_length: float = 0.5,
        margin_fraction: float = 0.2,
        min_margin: float = 0.5,
    ):
        """
        Plot SLAM evolution step-by-step in a grid of subplots using StepRecords.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from gtsam.utils import plot as gtsam_plot

        plt.ioff()

        steps = slam.history.steps
        K = len(steps)
        if K == 0:
            print("No estimates to plot!")
            return None, None

        # Compute grid layout
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))

        # Compute global axis limits across all stored estimates
        xlim, ylim = SLAMVisualizer._compute_global_limits_from_history(
            slam, steps, margin_fraction, min_margin
        )

        # Create subplots
        fig, axes = plt.subplots(
            rows, cols, figsize=(subplot_size * cols, subplot_size * rows)
        )
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # Plot each record
        for plot_idx, step in enumerate(steps):
            ax = axes_flat[plot_idx]
            rec = slam.history.get_or_raise(step)
            est = rec.estimate

            ax.set_aspect("equal")
            ax.set_title(f"Step {step} ({plot_idx}/{K - 1})")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            if est is None:
                ax.text(
                    0.5,
                    0.5,
                    "No estimate",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                continue

            # Plot poses up to current step
            for k in range(step + 1):
                pose_key = X(k)
                if est.exists(pose_key):
                    pose = est.atPose2(pose_key)
                    gtsam_plot.plot_pose2_on_axes(ax, pose, axis_length=axis_length)

            # Plot observed landmarks (that exist in this estimate)
            for lm_key in slam.landmark_keys:
                if est.exists(lm_key):
                    lm_pos = est.atPoint2(lm_key)
                    gtsam_plot.plot_point2_on_axes(ax, lm_pos, linespec="b")

            # Apply global limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(K, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        return fig, axes


    @staticmethod
    def _compute_global_limits_from_history(
        slam,
        steps,
        margin_fraction: float = 0.2,
        min_margin: float = 0.5,
    ):
        xs, ys = [], []

        for step in steps:
            rec = slam.history.get(step)
            if rec is None or rec.estimate is None:
                continue
            est = rec.estimate

            # poses up to this step
            for k in range(step + 1):
                pose_key = X(k)
                if est.exists(pose_key):
                    pose = est.atPose2(pose_key)
                    xs.append(pose.x())
                    ys.append(pose.y())

            # landmarks present in this estimate
            for lm_key in slam.landmark_keys:
                if est.exists(lm_key):
                    lm = est.atPoint2(lm_key)
                    xs.append(lm[0])
                    ys.append(lm[1])

        if len(xs) == 0:
            return (-1, 1), (-1, 1)

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        xspan = max(1e-3, xmax - xmin)
        yspan = max(1e-3, ymax - ymin)

        x_margin = max(min_margin, margin_fraction * xspan)
        y_margin = max(min_margin, margin_fraction * yspan)

        return (xmin - x_margin, xmax + x_margin), (ymin - y_margin, ymax + y_margin)

    @staticmethod
    def plot_trajectory_with_uncertainty(
        slam: FactorGraphSLAM,
        marginals: Optional[gtsam.Marginals] = None,
        show_landmarks: bool = True,
        figsize=(12, 8),
    ):
        """
        Plot robot trajectory with uncertainty ellipses

        Args:
            slam: FactorGraphSLAM object
            marginals: Pre-computed marginals (computed if None)
            show_landmarks: Whether to show landmarks
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        from gtsam.utils import plot as gtsam_plot

        if marginals is None:
            marginals = slam.get_marginals()

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_title("Robot Trajectory with Uncertainty")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # Plot trajectory line
        trajectory_x = []
        trajectory_y = []
        for k in range(slam.num_poses):
            pose = slam.values.atPose2(X(k))
            trajectory_x.append(pose.x())
            trajectory_y.append(pose.y())

        ax.plot(
            trajectory_x,
            trajectory_y,
            "r--",
            alpha=0.5,
            linewidth=1,
            label="Trajectory",
        )

        # Plot poses with covariance
        for k in range(slam.num_poses):
            pose_key = X(k)
            pose = slam.values.atPose2(pose_key)
            cov = marginals.marginalCovariance(pose_key)
            gtsam_plot.plot_pose2_on_axes(
                ax, pose=pose, axis_length=0.5, covariance=cov
            )

        # Plot landmarks if requested
        if show_landmarks:
            for lm_key in slam.landmark_keys:
                lm_pos = slam.values.atPoint2(lm_key)
                cov = marginals.marginalCovariance(lm_key)
                gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b", P=cov)

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig, ax

    @staticmethod
    def plot_measurement_space_step_by_step(
        slam,
        subplot_size: float = 4.0,
        show_lines: bool = True,
        show_labels: bool = False,
        margin_fraction: float = 0.1,
        min_margin_r: float = 0.5,
        min_margin_b: float = 0.1,
    ):
        """
        Plot measurement-space evolution step-by-step (range vs bearing) in a grid of subplots.

        Uses StepRecords in slam.history:
          - rec.measurements: list of (range, bearing_obj) where bearing_obj.theta() is used
          - rec.predicted_measurements: (N,2) array-like of [range, bearing]
          - rec.associations: list length M, with landmark ids or -1 for new/unassociated
          - rec.local_landmark_ids: list length N matching predicted_measurements
        """

        plt.ioff()

        steps = slam.history.steps
        K = len(steps)
        if K == 0:
            print("No history to plot!")
            return None, None

        # ---- global limits across all steps ----
        xlim, ylim = SLAMVisualizer._compute_global_meas_limits_from_history(
            slam,
            steps,
            margin_fraction=margin_fraction,
            min_margin_r=min_margin_r,
            min_margin_b=min_margin_b,
        )

        # ---- grid layout ----
        cols = int(np.ceil(np.sqrt(K)))
        rows = int(np.ceil(K / cols))

        fig, axes = plt.subplots(
            rows, cols, figsize=(subplot_size * cols, subplot_size * rows)
        )
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for plot_idx, step in enumerate(steps):
            ax = axes_flat[plot_idx]
            rec = slam.history.get_or_raise(step)

            ax.set_title(f"Step {step} ({plot_idx}/{K - 1})")
            ax.set_xlabel("range [m]")
            ax.set_ylabel("bearing [rad]")

            if rec.measurements is None:
                ax.text(
                    0.5,
                    0.5,
                    "No measurements",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                continue

            z = np.array(
                [[r, b.theta()] for (r, b) in rec.measurements], dtype=float
            )  # (M,2)

            zhat = rec.predicted_measurements
            if zhat is None:
                zhat = np.empty((0, 2), dtype=float)
            else:
                zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)

            # predicted + measured
            if len(zhat) > 0:
                ax.scatter(zhat[:, 0], zhat[:, 1], marker="o", label="predicted")
            if len(z) > 0:
                ax.scatter(z[:, 0], z[:, 1], marker="x", label="measured")

            # association lines measured -> predicted
            assoc = rec.associations if rec.associations is not None else []
            local_ids = getattr(rec, "local_landmark_ids", None)

            if (
                show_lines
                and local_ids is not None
                and len(local_ids) == len(zhat)
                and len(assoc) == len(z)
            ):
                id_to_i = {int(lm_id): i for i, lm_id in enumerate(local_ids)}
                assoc_arr = np.asarray(assoc, dtype=int)
                new_mask = assoc_arr == -1

                for j, a_j in enumerate(assoc_arr):
                    if a_j == -1:
                        continue
                    i = id_to_i.get(int(a_j), None)
                    if i is None:
                        continue

                    ax.plot(
                        [z[j, 0], zhat[i, 0]],
                        [z[j, 1], zhat[i, 1]],
                        linewidth=1,
                        alpha=0.6,
                    )
                    if show_labels:
                        ax.text(z[j, 0], z[j, 1], f"{a_j}", fontsize=8, alpha=0.8)

                if np.any(new_mask):
                    ax.scatter(
                        z[new_mask, 0],
                        z[new_mask, 1],
                        marker="x",
                        label="unassociated (-1)",
                    )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

            # only show legend if something was plotted
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(fontsize=8)

        # Hide unused axes
        for i in range(K, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def _compute_global_meas_limits_from_history(
        slam,
        steps,
        margin_fraction: float = 0.1,
        min_margin_r: float = 0.5,
        min_margin_b: float = 0.1,
    ):
        import numpy as np

        rs, bs = [], []

        for step in steps:
            rec = slam.history.get(step)
            if rec is None:
                continue

            if rec.measurements is not None:
                z = np.array(
                    [[r, b.theta()] for (r, b) in rec.measurements], dtype=float
                )
                if z.size > 0:
                    rs.extend(z[:, 0].tolist())
                    bs.extend(z[:, 1].tolist())

            zhat = rec.predicted_measurements
            if zhat is not None:
                zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)
                if zhat.size > 0:
                    rs.extend(zhat[:, 0].tolist())
                    bs.extend(zhat[:, 1].tolist())

        if len(rs) == 0:
            return (-1, 1), (-1, 1)

        rmin, rmax = float(np.min(rs)), float(np.max(rs))
        bmin, bmax = float(np.min(bs)), float(np.max(bs))

        rspan = max(1e-6, rmax - rmin)
        bspan = max(1e-6, bmax - bmin)

        r_margin = max(min_margin_r, margin_fraction * rspan)
        b_margin = max(min_margin_b, margin_fraction * bspan)

        return (rmin - r_margin, rmax + r_margin), (bmin - b_margin, bmax + b_margin)
