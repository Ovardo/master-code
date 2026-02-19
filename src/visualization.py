from typing import Iterable, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


from scipy.stats.distributions import chi2

from association import NIS
from config import VisualizationConfig
from result import SLAMHistory



class SLAMVisualizer:
    """Handle SLAM visualization with optimized numpy operations."""

    def __init__(self, config: VisualizationConfig, history: SLAMHistory):
        self.cfg = config
        self.history = history

    # =====================================
    # OPTIMIZED PLOTTING FUNCTIONS
    # =====================================
    
    def plot_estimates_np(
        self, 
        step: int,
        ax=None,
        dead_reckoning_poses: np.ndarray | None = None,  # (K, 3)
        ground_truth_poses: np.ndarray | None = None,  # (K, 3)
        ground_truth_landmarks: np.ndarray | None = None  # (L, 2)
    ):
        """Plot SLAM estimates at a given step (fully optimized)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 8))
            show_plot = True
        else:
            show_plot = False
        
        record = self.history.get_or_raise(step)
        poses_est = record.poses  # (K, 3)
        landmarks_est = record.landmarks  # (L, 2)

        # Plot trajectories and landmarks with single calls
        if len(poses_est) > 0:
            ax.plot(poses_est[:, 0], poses_est[:, 1], 'b-', alpha=0.7, 
                   label=r'$\hat{x}_{SLAM}$', linewidth=2)
        
        if len(landmarks_est) > 0:
            ax.scatter(landmarks_est[:, 0], landmarks_est[:, 1], 
                      c='r', marker='x', s=50, label=r'$\hat{m}_{SLAM}$')

        if dead_reckoning_poses is not None and len(dead_reckoning_poses) > 0:
            # Only plot up to current step
            dr_poses = dead_reckoning_poses # [:step+1] # TODO: dead_reckoning_poses is not same length as poses_est, need to handle this properly
            ax.plot(dr_poses[:, 0], dr_poses[:, 1], 'k-', alpha=0.7, 
                   label=r'$\hat{x}_{DR}$', linewidth=2)
        
        if ground_truth_poses is not None and len(ground_truth_poses) > 0:
            gt_poses = ground_truth_poses[:step+1]
            ax.plot(gt_poses[:, 0], gt_poses[:, 1], 'g--', alpha=0.7, 
                   label=r'$x_{GT}$', linewidth=2)

        if ground_truth_landmarks is not None and len(ground_truth_landmarks) > 0:
            ax.scatter(ground_truth_landmarks[:, 0], ground_truth_landmarks[:, 1], 
                      c='m', marker='D', s=50, label=r'$m_{GT}$')
        
        ax.legend()
        ax.set_title(f"Step: {step}, Num landmarks: {len(landmarks_est)}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        
        if show_plot:
            plt.show()
        
        return ax

    def plot_measurements_polar_np(self, step: int, ax=None):
        """Plot measured and predicted measurements with associations (optimized with separate masks)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_plot = True
        else:
            show_plot = False
        
        record = self.history.get_or_raise(step)
        
        meas = record.measurements  # (M, 2) [range, bearing]
        zbars = record.predicted_measurements  # (L', 2)
        zbars_ids = record.predicted_measurements_ids  # (L',)
        associations_idx = record.associations_idx  # (M,)

        # --- Predicted measurements (vectorized) ---
        if len(zbars) > 0:
            ax.scatter(zbars[:, 0], zbars[:, 1], marker='o', c='b', s=50, 
                    label="predicted", zorder=1)
            # Text labels still need loop
            for z, lm_id in zip(zbars, zbars_ids):
                ax.text(z[0], z[1], str(lm_id), fontsize=8, alpha=0.6)
        
        # --- Measured points (separate masks for control) ---
        if len(meas) > 0:
            # Create masks for each association type
            mask_unassoc = associations_idx == -1
            mask_ambig = associations_idx == -2
            mask_assoc = associations_idx >= 0
            
            # Plot in order: unassociated (bottom), ambiguous, associated (top)
            if mask_unassoc.any():
                ax.scatter(meas[mask_unassoc, 0], meas[mask_unassoc, 1], 
                        c='red', marker='x', s=50, zorder=3, 
                        label='unassociated (-1)', alpha=0.8)
            
            if mask_ambig.any():
                ax.scatter(meas[mask_ambig, 0], meas[mask_ambig, 1], 
                        c='yellow', marker='x', s=50, zorder=3, 
                        label='ambiguous (-2)', alpha=0.8)
            
            if mask_assoc.any():
                ax.scatter(meas[mask_assoc, 0], meas[mask_assoc, 1], 
                        c='orange', marker='x', s=50, zorder=3, 
                        label='associated', alpha=0.8)
                
                # --- Association lines (vectorized) ---
                meas_assoc = meas[mask_assoc]
                pred_assoc = zbars[associations_idx[mask_assoc]]
                segments = np.stack([meas_assoc, pred_assoc], axis=1)
                
                lc = LineCollection(segments, colors='k', alpha=0.6, linewidths=1, zorder=1)
                ax.add_collection(lc)
        
        ax.legend(loc='best')
        ax.set_title(f"Measurement space polar (step {step})")
        ax.set_xlabel("range [m]")
        ax.set_ylabel("bearing [rad]")
        ax.grid(True, alpha=0.3)
        
        if show_plot:
            plt.show()
        
        return ax

    def plot_measurements_cartesian_np(self, step: int, ax=None):
        """Plot measurements in cartesian space (fully optimized with separate masks)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            show_plot = True
        else:
            show_plot = False
        
        record = self.history.get_or_raise(step)
        
        measurements = record.measurements  # (M, 2) [range, bearing]
        predicted_measurements = record.predicted_measurements  # (L, 2)
        predicted_measurements_ids = record.predicted_measurements_ids  # (L,)
        associations_idx = record.associations_idx  # (M,)
        
        # --- Vectorized coordinate conversion ---
        r, b = measurements[:, 0], measurements[:, 1]
        meas_xy = np.column_stack([-r * np.sin(b), r * np.cos(b)])
        
        r_pred, b_pred = predicted_measurements[:, 0], predicted_measurements[:, 1]
        meas_pred_xy = np.column_stack([-r_pred * np.sin(b_pred), r_pred * np.cos(b_pred)])
        
        # --- Range circles (vectorized max) ---
        max_range = 0
        if len(r) > 0:
            max_range = max(max_range, np.max(r))
        if len(r_pred) > 0:
            max_range = max(max_range, np.max(r_pred))
        
        ring_limit = int(np.ceil(max_range / 10) + 1) * 10
        for radius in range(10, ring_limit + 1, 10):
            circle = plt.Circle((0, 0), radius, color='lightgray', fill=False, 
                            linewidth=0.8, linestyle='--')
            ax.add_patch(circle)
            ax.text(0, radius, f"{radius}m", fontsize=7, color='gray', 
                va='bottom', ha='center')
        
        # --- Predicted measurements ---
        if len(meas_pred_xy) > 0:
            ax.scatter(meas_pred_xy[:, 0], meas_pred_xy[:, 1], 
                    marker='o', c='b', s=50, zorder=1, label="predicted")
            for xy, lm_id in zip(meas_pred_xy, predicted_measurements_ids):
                ax.text(xy[0], xy[1], str(lm_id), fontsize=8, alpha=0.6)
        
        # --- Measured points (separate masks for control over plotting order) ---
        if len(meas_xy) > 0:
            # Create masks for each association type
            mask_unassoc = associations_idx == -1
            mask_ambig = associations_idx == -2
            mask_assoc = associations_idx >= 0
            
            # Plot in order: unassociated (bottom), ambiguous, associated (top)
            # Lower zorder = plotted first (behind)
            if mask_unassoc.any():
                ax.scatter(meas_xy[mask_unassoc, 0], meas_xy[mask_unassoc, 1], 
                        c='red', marker='x', s=50, zorder=3, 
                        label='unassociated (-1)', alpha=0.8)
            
            if mask_ambig.any():
                ax.scatter(meas_xy[mask_ambig, 0], meas_xy[mask_ambig, 1], 
                        c='yellow', marker='x', s=50, zorder=3, 
                        label='ambiguous (-2)', alpha=0.8)
            
            if mask_assoc.any():
                ax.scatter(meas_xy[mask_assoc, 0], meas_xy[mask_assoc, 1], 
                        c='orange', marker='x', s=50, zorder=3, 
                        label='associated', alpha=0.8)
                
                # --- Association lines (vectorized) ---
                meas_assoc_xy = meas_xy[mask_assoc]
                pred_assoc_xy = meas_pred_xy[associations_idx[mask_assoc]]
                segments = np.stack([meas_assoc_xy, pred_assoc_xy], axis=1)
                
                lc = LineCollection(segments, colors='k', alpha=0.6, linewidths=1, zorder=1)
                ax.add_collection(lc)
        
        # --- Formatting ---
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.legend(loc='best')
        ax.set_title(f"Measurement space cartesian (step {step})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        
        if show_plot:
            plt.show()
    
        return ax

    # =====================================
    # GENERIC VIDEO CREATION
    # =====================================
    
    def create_video_generic(
        self,
        plot_func: Callable[[int, plt.Axes], None],
        output_path: str,
        fps: int = 10,
        start_step: int = 0,
        end_step: Optional[int] = None,
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 100,
        **plot_kwargs
    ):
        """
        Generic video creator that takes any plotting function.
        
        Args:
            plot_func: Function with signature (step: int, ax: plt.Axes) -> None
            output_path: Path to save video (.mp4 or .gif)
            fps: Frames per second
            start_step: First step to include
            end_step: Last step to include (None = last available)
            figsize: Figure size
            dpi: DPI for output
            **plot_kwargs: Additional kwargs passed to plot_func
        """
        if end_step is None:
            end_step = len(self.history) - 1
        
        steps = range(start_step, end_step + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        def update(step):
            ax.clear()
            plot_func(step, ax, **plot_kwargs)
            return ax,
        
        anim = FuncAnimation(fig, update, frames=steps, interval=1000/fps, blit=False)
        
        # Save based on file extension
        if output_path.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_path, writer=writer, dpi=dpi)
        elif output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
        else:
            raise ValueError("output_path must end with .mp4 or .gif")
        
        plt.close(fig)
        print(f"Video saved to {output_path}")

    def create_measurement_video_polar(self, output_path: str, fps: int = 10, 
                                      start_step: int = 0, end_step: Optional[int] = None):
        """Create video of polar measurements."""
        self.create_video_generic(
            plot_func=self.plot_measurements_polar_np,
            output_path=output_path,
            fps=fps,
            start_step=start_step,
            end_step=end_step,
            figsize=(10, 8)
        )

    def create_measurement_video_cartesian(self, output_path: str, fps: int = 10,
                                          start_step: int = 0, end_step: Optional[int] = None):
        """Create video of cartesian measurements."""
        self.create_video_generic(
            plot_func=self.plot_measurements_cartesian_np,
            output_path=output_path,
            fps=fps,
            start_step=start_step,
            end_step=end_step,
            figsize=(10, 10)
        )

    def create_estimates_video(
        self,
        output_path: str,
        fps: int = 10,
        start_step: int = 0,
        end_step: Optional[int] = None,
        dead_reckoning_poses: Optional[np.ndarray] = None,
        ground_truth_poses: Optional[np.ndarray] = None,
        ground_truth_landmarks: Optional[np.ndarray] = None,
    ):
        """Create video of SLAM estimates evolution."""
        self.create_video_generic(
            plot_func=self.plot_estimates_np,
            output_path=output_path,
            fps=fps,
            start_step=start_step,
            end_step=end_step,
            figsize=(13, 8),
            dead_reckoning_poses=dead_reckoning_poses,
            ground_truth_poses=ground_truth_poses,
            ground_truth_landmarks=ground_truth_landmarks,
        )

    def create_combined_video(
        self,
        output_path: str,
        fps: int = 10,
        start_step: int = 0,
        end_step: Optional[int] = None,
        layout: str = 'horizontal'  # 'horizontal' or 'vertical'
    ):
        """
        Create video with multiple subplots (e.g., polar + cartesian measurements).
        
        Args:
            layout: 'horizontal' for side-by-side, 'vertical' for stacked
        """
        if end_step is None:
            end_step = len(self.history) - 1
        
        steps = range(start_step, end_step + 1)
        
        if layout == 'horizontal':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        else:  # vertical
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
        
        def update(step):
            ax1.clear()
            ax2.clear()
            self.plot_measurements_polar_np(step, ax=ax1)
            self.plot_measurements_cartesian_np(step, ax=ax2)
            fig.suptitle(f"Step {step}", fontsize=14)
            return ax1, ax2
        
        anim = FuncAnimation(fig, update, frames=steps, interval=1000/fps, blit=False)
        
        if output_path.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(output_path, writer=writer)
        elif output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
        
        plt.close(fig)
        print(f"Combined video saved to {output_path}")

    # =====================================
    # ADVANCED: MULTI-PANEL VIDEO
    # =====================================
    
    def create_dashboard_video(
        self,
        output_path: str,
        fps: int = 10,
        start_step: int = 0,
        end_step: Optional[int] = None,
        dead_reckoning_poses: Optional[np.ndarray] = None,
        ground_truth_poses: Optional[np.ndarray] = None,
    ):
        """
        Create a dashboard video with:
        - Top: Estimates (trajectory + landmarks)
        - Bottom left: Polar measurements
        - Bottom right: Cartesian measurements
        """
        if end_step is None:
            end_step = len(self.history) - 1
        
        steps = range(start_step, end_step + 1)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        
        ax_est = fig.add_subplot(gs[0, :])  # Top, spans both columns
        ax_polar = fig.add_subplot(gs[1, 0])  # Bottom left
        ax_cart = fig.add_subplot(gs[1, 1])  # Bottom right
        
        def update(step):
            ax_est.clear()
            ax_polar.clear()
            ax_cart.clear()
            
            self.plot_estimates_np(
                step, ax=ax_est,
                dead_reckoning_poses=dead_reckoning_poses,
                ground_truth_poses=ground_truth_poses
            )
            self.plot_measurements_polar_np(step, ax=ax_polar)
            self.plot_measurements_cartesian_np(step, ax=ax_cart)
            
            fig.suptitle(f"SLAM Dashboard - Step {step}", fontsize=16, fontweight='bold')
            return ax_est, ax_polar, ax_cart
        
        anim = FuncAnimation(fig, update, frames=steps, interval=1000/fps, blit=False)
        
        if output_path.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps, bitrate=3600)
            anim.save(output_path, writer=writer, dpi=120)
        elif output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=120)
        
        plt.close(fig)
        print(f"Dashboard video saved to {output_path}")







# class SLAMVisualizer:
#     """Handle SLAM visualization"""

#     def __init__(self, config: VisualizationConfig, history: SLAMHistory):
#         self.cfg = config
#         self.history = history

#     def plot_estimates(
#             self, 
#             step: int, 
#             # plot_measurements: bool = False,
#             # plot_predicted_measurements: bool = False,
#             # plot_associations: bool = False,
#             dead_reckoning_poses: Optional[list[gtsam.Pose2]] = None, 
#             ground_truth_poses: Optional[list[gtsam.Pose2]] = None, 
#             ground_truth_landmarks: Optional[list[gtsam.Point2]] = None
#         ):
#         """Plot SLAM estimates at a given step using history (StepRecord)."""
#         record = self.history.get_or_raise(step) 

#         poses_est = record.poses
#         landmarks_est = record.landmarks

#         plt.figure(figsize=(13, 8))
        
#         x = [p.x() for p in poses_est]
#         y = [p.y() for p in poses_est]
#         plt.plot(x, y, 'b-', alpha=0.7, label=r'$\hat{x}_{SLAM}$')
     
#         x = [lm[0] for lm in landmarks_est]
#         y = [lm[1] for lm in landmarks_est]
#         plt.scatter(x, y, c='r', marker='x', label=r'$\hat{m}_{SLAM}$')

#         if dead_reckoning_poses is not None:
#             x_dr = [p.x() for p in dead_reckoning_poses]
#             y_dr = [p.y() for p in dead_reckoning_poses]
#             plt.plot(x_dr, y_dr, 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$')
        
#         plt.legend()
#         plt.title("Step: {}, Num landmarks: {}".format(step, len(landmarks_est)))
#         plt.xlabel("x [m]")
#         plt.ylabel("y [m]")
#         plt.axis("equal")
#         plt.show()


#     def plot_estimates_np(
#             self, 
#             step: int, 
#             # plot_measurements: bool = False,
#             # plot_predicted_measurements: bool = False,
#             # plot_associations: bool = False,
#             dead_reckoning_poses: np.ndarray | None = None, # (K,3)
#             ground_truth_poses: np.ndarray | None = None, # (K,3)
#             ground_truth_landmarks: np.ndarray | None = None # (L,3)
#         ):
#         """Plot SLAM estimates at a given step using history (StepRecord)."""
#         record = self.history.get_or_raise(step) 

#         poses_est = record.poses # (K, 3)
#         landmarks_est = record.landmarks # (L, 2) 

#         plt.figure(figsize=(13, 8))
        
#         plt.plot(poses_est[:, 0], poses_est[:, 1], 'b-', alpha=0.7, label=r'$\hat{x}_{SLAM}$')
#         plt.scatter(landmarks_est[:, 0], landmarks_est[:, 1], c='r', marker='x', label=r'$\hat{m}_{SLAM}$')

#         if dead_reckoning_poses is not None:
#             plt.plot(dead_reckoning_poses[:, 0], dead_reckoning_poses[:, 1], 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$')
        
#         if ground_truth_poses is not None:
#             plt.plot(ground_truth_poses[:, 0], ground_truth_poses[:, 1], 'g-', alpha=0.7, label=r'$x_{GT}$')

#         if ground_truth_landmarks is not None:
#             plt.scatter(ground_truth_landmarks[:, 0], ground_truth_landmarks[:, 1], c='m', marker='D', label=r'$m_{GT}$')
        
#         plt.legend()
#         plt.title("Step: {}, Num landmarks: {}".format(step, len(landmarks_est)))
#         plt.xlabel("x [m]")
#         plt.ylabel("y [m]")
#         plt.axis("equal")
#         plt.show()
    

#     def plot_measurements_polar_np(self, step: int, ax = None):
#         """Plot measured and predicted measurements with associations at a given step."""
#         if ax is None:
#             ax = plt.gca()
        
#         record = self.history.get_or_raise(step)
        
#         meas = record.measurements # (M,2) np.ndarray
#         zbars = record.predicted_measurements # (L',2) np.ndarray
#         zbars_ids = [zbar.lm_id for zbar in zbars] # (L',) np.ndarray
#         associations_ids = record.associations_ids # (M,) np.ndarray
#         associations_idx = record.associations_idx # (M,) np.ndarray 

#         # --- Predicted measurements ---
#         if len(zbars) > 0:
#             ax.scatter(zbars[:, 0], zbars[:, 1], marker='o', c='b', s=50, label="predicted")
#             for z, lm_id in zip(zbars, zbars_ids):
#                 ax.text(z[0], z[1], str(lm_id), fontsize=8, alpha=0.6)
        
#         # --- Measured points ---
#         if len(meas) > 0:
#             # Create color array based on associations
#             colors = np.empty(len(associations_ids), dtype='U10')
#             colors[associations_ids == -1] = 'red'
#             colors[associations_ids == -2] = 'yellow'
#             colors[associations_ids >= 0] = 'orange'
            
#             # Single scatter call for all measurements
#             ax.scatter(meas[:, 0], meas[:, 1], c=colors, marker='x', s=50)
            
#             # Manual legend (since we used color array)
#             legend_elements = [
#                 Patch(facecolor='b', label='predicted'),
#                 Patch(facecolor='orange', label='associated'),
#                 Patch(facecolor='red', label='unassociated (-1)'),
#                 Patch(facecolor='yellow', label='ambiguous (-2)')
#             ]
#             ax.legend(handles=legend_elements)
        
#         # --- Association lines ---
#         mask_assoc = associations_idx >= 0
#         if mask_assoc.any():
            
#             assoc_indices = associations_idx[mask_assoc]
#             meas_assoc = meas[mask_assoc]
            
#             segments = []
#             for meas, idx in zip(meas_assoc, assoc_indices):
#                 segments.append([meas, zbars[idx]])
            
#             if segments:
#                 lc = LineCollection(segments, colors='k', alpha=0.6, linewidths=1)
#                 ax.add_collection(lc)
        
#         ax.set_title(f"Measurement space (step {step})")
#         ax.set_xlabel("range [m]")
#         ax.set_ylabel("bearing [rad]")



#     def plot_measurements_polar(self, step: int):
#         """Plot measured and predicted measurements with associations at a given step."""
#         record = self.history.get_or_raise(step)
#         measurements = record.measurements
#         predicted_measurements = record.predicted_measurements
#         associations = record.associations_ids

#         # --- Predicted measurements ---
#         for pred in predicted_measurements:
#             plt.scatter(pred.zbar[0], pred.zbar[1], marker='o', c='b', label="predicted" if pred is predicted_measurements[0] else "")
#             plt.text(pred.zbar[0], pred.zbar[1], str(pred.lm_id), fontsize=8, alpha=0.6)

#         pred_meas_by_id = {m.lm_id: m.zbar for m in predicted_measurements}

#         # Association style: (color, label)
#         ASSOC_STYLE = {
#             -1: ('r', "unassociated (-1)"),
#             -2: ('y', "ambiguous (-2)"),
#         }
#         DEFAULT_ASSOC_STYLE = ('orange', "associated")

#         seen_labels = set()

#         def scatter_once(r, b, color, label):
#             """Scatter with de-duplicated legend labels."""
#             unique_label = label if label not in seen_labels else ""
#             seen_labels.add(label)
#             plt.scatter(r, b, c=color, marker='x', label=unique_label)

#         # --- Measured points ---
#         for assoc, (r, b) in zip(associations, measurements):
#             bearing = b.theta()

#             if assoc in ASSOC_STYLE:
#                 color, label = ASSOC_STYLE[assoc]
#                 scatter_once(r, bearing, color, label)
#             elif assoc >= 0:
#                 color, label = DEFAULT_ASSOC_STYLE
#                 scatter_once(r, bearing, color, label)

#                 zbar = pred_meas_by_id.get(assoc)
#                 if zbar is not None:
#                     plt.plot([r, zbar[0]], [bearing, zbar[1]], 'k-', alpha=0.6)

#         plt.legend()
#         plt.title(f"Measurement space (step {step})")
#         plt.xlabel("range [m]")
#         plt.ylabel("bearing [rad]")
#         plt.show()

#     def plot_measurements_cartesian_fastest(self, step: int, ax=None):
#         """Ultra-optimized version with single scatter call."""
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(10, 10))
        
#         record = self.history.get_or_raise(step)
        
#         measurements = record.measurements
#         predicted_measurements = record.predicted_measurements
#         predicted_measurements_ids = record.predicted_measurement_ids
#         associations_idx = record.associations_idx
        
#         # --- Vectorized coordinate conversion ---
#         r = measurements[:, 0]
#         b = measurements[:, 1]
#         meas_xy = np.column_stack([-r * np.sin(b), r * np.cos(b)])
        
#         r_pred = predicted_measurements[:, 0]
#         b_pred = predicted_measurements[:, 1]
#         meas_pred_xy = np.column_stack([-r_pred * np.sin(b_pred), r_pred * np.cos(b_pred)])
        
#         # --- Range circles ---
#         max_range = 0
#         if len(r) > 0:
#             max_range = max(max_range, np.max(r))
#         if len(r_pred) > 0:
#             max_range = max(max_range, np.max(r_pred))
        
#         ring_limit = int(np.ceil(max_range / 10) + 1) * 10
#         for radius in range(10, ring_limit + 1, 10):
#             circle = plt.Circle((0, 0), radius, color='lightgray', fill=False, linewidth=0.8, linestyle='--')
#             ax.add_patch(circle)
#             ax.text(0, radius, f"{radius}m", fontsize=7, color='gray', va='bottom', ha='center')
        
#         # --- Predicted measurements ---
#         if len(meas_pred_xy) > 0:
#             ax.scatter(meas_pred_xy[:, 0], meas_pred_xy[:, 1], marker='o', c='b', s=50, zorder=3, label="predicted")
#             for xy, lm_id in zip(meas_pred_xy, predicted_measurements_ids):
#                 ax.text(xy[0], xy[1], str(lm_id), fontsize=8, alpha=0.6)
        
#         # --- Single scatter for all measurements with color array ---
#         if len(meas_xy) > 0:
#             # Create color array based on associations
#             colors = np.where(associations_idx == -1, 'red',
#                     np.where(associations_idx == -2, 'yellow', 'orange'))
            
#             ax.scatter(meas_xy[:, 0], meas_xy[:, 1], 
#                     c=colors, marker='x', s=50, zorder=2)
            
#             # Custom legend
#             from matplotlib.lines import Line2D
#             legend_elements = [
#                 Line2D([0], [0], marker='o', color='w', markerfacecolor='b', 
#                     markersize=8, label='predicted'),
#                 Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', 
#                     markersize=8, label='associated'),
#                 Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
#                     markersize=8, label='unassociated (-1)'),
#                 Line2D([0], [0], marker='x', color='w', markerfacecolor='yellow', 
#                     markersize=8, label='ambiguous (-2)')
#             ]
#             ax.legend(handles=legend_elements)
        
#         # --- Association lines ---
#         mask_assoc = associations_idx >= 0
#         if mask_assoc.any():
#             from matplotlib.collections import LineCollection
            
#             meas_assoc_xy = meas_xy[mask_assoc]
#             pred_assoc_xy = meas_pred_xy[associations_idx[mask_assoc]]
#             segments = np.stack([meas_assoc_xy, pred_assoc_xy], axis=1)
            
#             lc = LineCollection(segments, colors='k', alpha=0.6, linewidths=1, zorder=1)
#             ax.add_collection(lc)
        
#         # --- Formatting ---
#         ax.set_aspect('equal')
#         ax.axhline(0, color='gray', linewidth=0.5)
#         ax.axvline(0, color='gray', linewidth=0.5)
#         ax.set_title(f"Measurement space cartesian (step {step})")
#         ax.set_xlabel("x [m]")
#         ax.set_ylabel("y [m]")
        
#         return ax

    
#     def plot_measurements_cartesian(self, step: int):
#         """Plot measured and predicted measurements in cartesian space at a given step."""
#         record = self.history.get_or_raise(step)
        
#         measurements = record.measurements
#         predicted_measurements = record.predicted_measurements
#         associations = record.associations

#         pred_meas_by_id = {m.lm_id: m.zbar for m in predicted_measurements}

#         def to_cartesian(r: float, b: float) -> tuple[float, float]:
#             """Convert polar (range, bearing) to cartesian (x, y).
#             bearing=0 points along +y, bearing=90deg points along -x.
#             """
#             return -r * np.sin(b), r * np.cos(b)

#         fig, ax = plt.subplots()

#         # --- Range circles ---
#         max_range = max((r for r, _ in measurements), default=0)
#         max_range = max(max_range, max((np.hypot(*to_cartesian(pred.zbar[0], pred.zbar[1])) for pred in predicted_measurements), default=0))
#         ring_limit = int(np.ceil(max_range / 10) + 1) * 10

#         for radius in range(10, ring_limit + 1, 10):
#             circle = plt.Circle((0, 0), radius, color='lightgray', fill=False, linewidth=0.8, linestyle='--')
#             ax.add_patch(circle)
#             ax.text(0, radius, f"{radius}m", fontsize=7, color='gray', va='bottom', ha='center')

#         # --- Predicted measurements ---
#         for i, pred in enumerate(predicted_measurements):
#             x, y = to_cartesian(pred.zbar[0], pred.zbar[1])
#             ax.scatter(x, y, marker='o', c='b', label="predicted" if i == 0 else "")
#             ax.text(x, y, str(pred.lm_id), fontsize=8, alpha=0.6)

#         # Association style: (color, label)
#         ASSOC_STYLE = {
#             -1: ('r', "unassociated (-1)"),
#             -2: ('y', "ambiguous (-2)"),
#         }
#         DEFAULT_ASSOC_STYLE = ('orange', "associated")

#         seen_labels: set[str] = set()

#         def scatter_once(x, y, color, label):
#             unique_label = label if label not in seen_labels else ""
#             seen_labels.add(label)
#             ax.scatter(x, y, c=color, marker='x', label=unique_label)

#         # --- Measured points ---
#         for assoc, (r, b) in zip(associations, measurements):
#             x, y = to_cartesian(r, b.theta())

#             if assoc in ASSOC_STYLE:
#                 color, label = ASSOC_STYLE[assoc]
#                 scatter_once(x, y, color, label)
#             elif assoc >= 0:
#                 color, label = DEFAULT_ASSOC_STYLE
#                 scatter_once(x, y, color, label)

#                 zbar = pred_meas_by_id.get(assoc)
#                 if zbar is not None:
#                     x_pred, y_pred = to_cartesian(zbar[0], zbar[1])
#                     ax.plot([x, x_pred], [y, y_pred], 'k-', alpha=0.6)

#         # --- Formatting ---
#         ax.set_aspect('equal')
#         ax.axhline(0, color='gray', linewidth=0.5)
#         ax.axvline(0, color='gray', linewidth=0.5)
#         ax.legend()
#         ax.set_title(f"Measurement space cartesian (step {step})")
#         ax.set_xlabel("x [m]")
#         ax.set_ylabel("y [m]")
#         plt.tight_layout()
#         plt.show()

    


#     # ================
#     # Vidoe generation
#     # ================
    
#     def create_measurement_video_polar(self, output_path: str, fps: int = 10, 
#                                       start_step: int = 0, end_step: int = None):
#         """Create video of polar measurements over time."""
#         if end_step is None:
#             end_step = len(self.history) - 1
        
#         steps = range(start_step, end_step + 1)
        
#         fig, ax = plt.subplots(figsize=(10, 8))
        
#         def update(step):
#             ax.clear()
#             self._plot_measurements_polar_on_ax(ax, step)
#             return ax,
        
#         anim = FuncAnimation(fig, update, frames=steps, interval=1000/fps, blit=False)
        
#         # Save as mp4 (requires ffmpeg) or gif
#         if output_path.endswith('.mp4'):
#             writer = FFMpegWriter(fps=fps, bitrate=1800)
#             anim.save(output_path, writer=writer)
#         elif output_path.endswith('.gif'):
#             writer = PillowWriter(fps=fps)
#             anim.save(output_path, writer=writer)
        
#         plt.close(fig)
#         print(f"Video saved to {output_path}")
    
#     def _plot_measurements_polar_on_ax(self, ax, step: int):
#         """Modified version that plots on a given axis instead of creating new figure."""
#         record = self.history.get_or_raise(step)
#         measurements = record.measurements
#         predicted_measurements = record.predicted_measurements
#         associations = record.associations

#         # --- Predicted measurements ---
#         for i, pred in enumerate(predicted_measurements):
#             ax.scatter(pred.zbar[0], pred.zbar[1], marker='o', c='b', 
#                       label="predicted" if i == 0 else "")
#             ax.text(pred.zbar[0], pred.zbar[1], str(pred.lm_id), 
#                    fontsize=8, alpha=0.6)

#         pred_meas_by_id = {m.lm_id: m.zbar for m in predicted_measurements}

#         # Association style: (color, label)
#         ASSOC_STYLE = {
#             -1: ('r', "unassociated (-1)"),
#             -2: ('y', "ambiguous (-2)"),
#         }
#         DEFAULT_ASSOC_STYLE = ('orange', "associated")

#         seen_labels = set()

#         def scatter_once(r, b, color, label):
#             """Scatter with de-duplicated legend labels."""
#             unique_label = label if label not in seen_labels else ""
#             seen_labels.add(label)
#             ax.scatter(r, b, c=color, marker='x', label=unique_label)

#         # --- Measured points ---
#         for assoc, (r, b) in zip(associations, measurements):
#             bearing = b.theta()

#             if assoc in ASSOC_STYLE:
#                 color, label = ASSOC_STYLE[assoc]
#                 scatter_once(r, bearing, color, label)
#             elif assoc >= 0:
#                 color, label = DEFAULT_ASSOC_STYLE
#                 scatter_once(r, bearing, color, label)

#                 zbar = pred_meas_by_id.get(assoc)
#                 if zbar is not None:
#                     ax.plot([r, zbar[0]], [bearing, zbar[1]], 'k-', alpha=0.6)

#         ax.legend()
#         ax.set_title(f"Measurement space (step {step})")
#         ax.set_xlabel("range [m]")
#         ax.set_ylabel("bearing [rad]")
    
#     # TODO: generalize this to take a plotting function as argument? or just make a separate one for estimates?
#     def create_measurement_video_cartesian(self, output_path: str, fps: int = 10,
#                                           start_step: int = 1, end_step: int = None):
#         """Create video of cartesian measurements over time."""
#         if end_step is None:
#             end_step = len(self.history) - 1
        
#         steps = range(start_step, end_step + 1)
        
#         fig, ax = plt.subplots(figsize=(10, 10))
        
#         def update(step):
#             ax.clear()
#             self._plot_measurements_cartesian_on_ax(ax, step)
#             return ax,
        
#         anim = FuncAnimation(fig, update, frames=steps, interval=1000/fps, blit=False)
        
#         if output_path.endswith('.mp4'):
#             writer = FFMpegWriter(fps=fps, bitrate=1800)
#             anim.save(output_path, writer=writer)
#         elif output_path.endswith('.gif'):
#             writer = PillowWriter(fps=fps)
#             anim.save(output_path, writer=writer)
        
#         plt.close(fig)
#         print(f"Video saved to {output_path}")
    
#     def _plot_measurements_cartesian_on_ax(self, ax, step: int):
#         """Modified version that plots on a given axis."""
#         record = self.history.get_or_raise(step)
#         measurements = record.measurements
#         predicted_measurements = record.predicted_measurements
#         associations = record.associations

#         pred_meas_by_id = {m.lm_id: m.zbar for m in predicted_measurements}

#         def to_cartesian(r: float, b: float) -> tuple[float, float]:
#             return -r * np.sin(b), r * np.cos(b)

#         # --- Range circles ---
#         max_range = max((r for r, _ in measurements), default=0)
#         max_range = max(max_range, max((np.hypot(*to_cartesian(pred.zbar[0], pred.zbar[1])) 
#                                        for pred in predicted_measurements), default=0))
#         ring_limit = int(np.ceil(max_range / 10) + 1) * 10

#         for radius in range(10, ring_limit + 1, 10):
#             circle = plt.Circle((0, 0), radius, color='lightgray', fill=False, 
#                               linewidth=0.8, linestyle='--')
#             ax.add_patch(circle)
#             ax.text(0, radius, f"{radius}m", fontsize=7, color='gray', 
#                    va='bottom', ha='center')

#         # --- Predicted measurements ---
#         for i, pred in enumerate(predicted_measurements):
#             x, y = to_cartesian(pred.zbar[0], pred.zbar[1])
#             ax.scatter(x, y, marker='o', c='b', label="predicted" if i == 0 else "")
#             ax.text(x, y, str(pred.lm_id), fontsize=8, alpha=0.6)

#         ASSOC_STYLE = {
#             -1: ('r', "unassociated (-1)"),
#             -2: ('g', "ambiguous (-2)"),
#         }
#         DEFAULT_ASSOC_STYLE = ('orange', "associated")

#         seen_labels: set[str] = set()

#         def scatter_once(x, y, color, label):
#             unique_label = label if label not in seen_labels else ""
#             seen_labels.add(label)
#             ax.scatter(x, y, c=color, marker='x', label=unique_label)

#         # --- Measured points ---
#         for assoc, (r, b) in zip(associations, measurements):
#             x, y = to_cartesian(r, b.theta())

#             if assoc in ASSOC_STYLE:
#                 color, label = ASSOC_STYLE[assoc]
#                 scatter_once(x, y, color, label)
#             elif assoc >= 0:
#                 color, label = DEFAULT_ASSOC_STYLE
#                 scatter_once(x, y, color, label)

#                 zbar = pred_meas_by_id.get(assoc)
#                 if zbar is not None:
#                     x_pred, y_pred = to_cartesian(zbar[0], zbar[1])
#                     ax.plot([x, x_pred], [y, y_pred], 'k-', alpha=0.6)

#         # --- Formatting ---
#         ax.set_aspect('equal')
#         ax.axhline(0, color='gray', linewidth=0.5)
#         ax.axvline(0, color='gray', linewidth=0.5)
#         ax.legend()
#         ax.set_title(f"Measurement space cartesian (step {step})")
#         ax.set_xlabel("x [m]")
#         ax.set_ylabel("y [m]")





#     def _init_estimate_figure(self):
#         fig, ax = plt.subplots(figsize=(13, 8))

#         # Lines/markers that we update each frame
#         slam_line, = ax.plot([], [], 'b-', alpha=0.7, label=r'$\hat{x}_{SLAM}$')
#         lm_scatter = ax.scatter([], [], c='r', marker='x', label=r'$\hat{m}_{SLAM}$')

#         dr_line = None
#         if getattr(self.cfg, "show_dead_reckoning", True):
#             dr_line, = ax.plot([], [], 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$')

#         ax.legend()
#         ax.set_aspect("equal", adjustable="box")
#         ax.grid(True, alpha=0.2)

#         title = ax.set_title("")
#         return fig, ax, slam_line, lm_scatter, dr_line, title

#     def _update_estimate_frame(
#         self,
#         ax,
#         slam_line,
#         lm_scatter,
#         dr_line,
#         title_artist,
#         step: int,
#         dead_reckoning_poses: Optional[list] = None,
#         ground_truth_poses: Optional[list] = None,
#         ground_truth_landmarks: Optional[list] = None,
#         autoscale: bool = True,
#     ):
#         record = self.history.get_or_raise(step)

#         poses_est = record.poses      # expected shape (k+1, 3) OR list of Pose2-like
#         lms_est = record.landmarks    # expected shape (L, 2)

#         # --- SLAM poses ---
#         if poses_est is None or len(poses_est) == 0:
#             slam_x, slam_y = [], []
#         else:
#             slam_x = [p.x() for p in poses_est]
#             slam_y = [p.y() for p in poses_est]

#         slam_line.set_data(slam_x, slam_y)

#         # --- Landmarks ---
#         if lms_est is None or len(lms_est) == 0:
#             lm_xy = np.empty((0, 2))
#         else:
#             lm_xy = np.asarray(lms_est).reshape(-1, 2)
#         lm_scatter.set_offsets(lm_xy)

#         # --- Dead reckoning (optional) ---
#         if dr_line is not None:
#             if dead_reckoning_poses is None or len(dead_reckoning_poses) == 0:
#                 dr_line.set_data([], [])
#             else:
#                 # list of Pose2 expected
#                 dr_x = [p.x() for p in dead_reckoning_poses[: step + 1]]
#                 dr_y = [p.y() for p in dead_reckoning_poses[: step + 1]]
#                 dr_line.set_data(dr_x, dr_y)

#         title_artist.set_text(f"Step: {step}, Num landmarks: {0 if lms_est is None else len(lms_est)}")

#         # --- keep view stable or autoscale ---
#         if autoscale:
#             xs = np.array(slam_x) if len(slam_x) else np.array([])
#             ys = np.array(slam_y) if len(slam_y) else np.array([])
#             if lm_xy.size:
#                 xs = np.concatenate([xs, lm_xy[:, 0]]) if xs.size else lm_xy[:, 0]
#                 ys = np.concatenate([ys, lm_xy[:, 1]]) if ys.size else lm_xy[:, 1]
#             if xs.size and ys.size:
#                 pad = 1.0
#                 ax.set_xlim(xs.min() - pad, xs.max() + pad)
#                 ax.set_ylim(ys.min() - pad, ys.max() + pad)

#     def create_video(
#         self,
#         filename: str,
#         fps: int = 5,
#         steps: Optional[Iterable[int]] = None,
#         dead_reckoning_poses: Optional[list] = None,
#         autoscale_each_frame: bool = False,
#         dpi: int = 150,
#     ):
#         """
#         Create an mp4 where each frame is one SLAM step.
#         Requires ffmpeg installed on your system.
#         """
#         if steps is None:
#             steps = self.history.steps
#         steps = list(steps)
#         if not steps:
#             raise ValueError("No steps in history.")

#         fig, ax, slam_line, lm_scatter, dr_line, title_artist = self._init_estimate_figure()

#         # For a nicer “step through” video, keep axes fixed across frames.
#         # We compute global bounds once if autoscale_each_frame=False.
#         if not autoscale_each_frame:
#             all_x, all_y = [], []
#             for k in steps:
#                 rec = self.history.get_or_raise(k)
#                 if rec.poses is not None and len(rec.poses):
#                     if isinstance(rec.poses, np.ndarray):
#                         all_x.extend(rec.poses[:, 0].tolist())
#                         all_y.extend(rec.poses[:, 1].tolist())
#                     else:
#                         all_x.extend([p.x() for p in rec.poses])
#                         all_y.extend([p.y() for p in rec.poses])
#                 if rec.landmarks is not None and len(rec.landmarks):
#                     lm = np.asarray(rec.landmarks).reshape(-1, 2)
#                     all_x.extend(lm[:, 0].tolist())
#                     all_y.extend(lm[:, 1].tolist())

#             if all_x and all_y:
#                 pad = 1.0
#                 ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
#                 ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

#         writer = FFMpegWriter(fps=fps, metadata={"artist": "SLAMVisualizer"})
#         with writer.saving(fig, filename, dpi=dpi):
#             for k in steps:
#                 self._update_estimate_frame(
#                     ax=ax,
#                     slam_line=slam_line,
#                     lm_scatter=lm_scatter,
#                     dr_line=dr_line,
#                     title_artist=title_artist,
#                     step=k,
#                     dead_reckoning_poses=dead_reckoning_poses,
#                     autoscale=autoscale_each_frame,
#                 )
#                 fig.canvas.draw()
#                 writer.grab_frame()

#         plt.close(fig)







    # def plot_measurement_space(
    #     self,
    #     step: int,
    #     show_lines: bool = True,
    #     show_labels: bool = True,
    #     figsize=(7, 5),
    # ):

    #     rec = self.history.get_or_raise(step)

    #     if rec.measurements is None:
    #         raise ValueError(f"No measurements stored for step={step}")

    #     z = np.array(
    #         [[r, b.theta()] for (r, b) in rec.measurements], dtype=float
    #     )  # (M,2)

    #     zhat = rec.predicted_measurements
    #     if zhat is None:
    #         zhat = np.empty((0, 2), dtype=float)
    #     else:
    #         zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)

    #     assoc = rec.associations if rec.associations is not None else []
    #     local_ids = getattr(rec, "local_landmark_ids", None)

    #     fig, ax = plt.subplots(figsize=figsize)

    #     # predicted + measured
    #     if len(zhat) > 0:
    #         ax.scatter(zhat[:, 0], zhat[:, 1], marker="o", label="predicted")
    #     if len(z) > 0:
    #         ax.scatter(z[:, 0], z[:, 1], marker="x", label="measured")

    #     # association lines measured -> predicted
    #     if (
    #         show_lines
    #         and local_ids is not None
    #         and len(local_ids) == len(zhat)
    #         and len(assoc) == len(z)
    #     ):
    #         id_to_i = {lm_id: i for i, lm_id in enumerate(local_ids)}
    #         i_to_id = {i: lm_id for i, lm_id in enumerate(local_ids)}

    #         assoc_arr = np.asarray(assoc, dtype=int)
    #         new_mask = assoc_arr == -1

    #         for j, a_j in enumerate(assoc_arr):
    #             if a_j == -1:
    #                 continue
    #             i = id_to_i.get(int(a_j), None)
    #             if i is None:
    #                 continue

    #             ax.plot(
    #                 [z[j, 0], zhat[i, 0]], [z[j, 1], zhat[i, 1]], linewidth=1, alpha=0.6
    #             )

    #             if show_labels:
    #                 ax.text(z[j, 0], z[j, 1], f"{a_j}", fontsize=8, alpha=0.8)

    #         # mark unassociated
    #         if np.any(new_mask):
    #             ax.scatter(
    #                 z[new_mask, 0],
    #                 z[new_mask, 1],
    #                 marker="x",
    #                 label="unassociated (-1)",
    #             )

    #     ax.set_title(f"Measurement space (step {step})")
    #     ax.set_xlabel("range [m]")
    #     ax.set_ylabel("bearing [rad]")
    #     ax.grid(True, alpha=0.3)
    #     ax.legend()
    #     plt.tight_layout()
    #     return fig, ax

    # def plot_NIS(self, figsize=(13, 3), ax=None, show_expected=True):

    #     steps = list(self.history.steps)
    #     N = len(steps)

    #     nis_sequence = np.full(N, np.nan, dtype=float)
    #     dof_sequence = np.zeros(N, dtype=int)
    #     lower_bounds = np.full(N, np.nan, dtype=float)
    #     upper_bounds = np.full(N, np.nan, dtype=float)

    #     for k, step in enumerate(steps):
    #         if step == 0:
    #             continue  # skip first step (no measurements)
    #         rec = self.history.get_or_raise(step)

    #         if rec.innovation_covariance is None:
    #             raise ValueError(f"No innovation covariance stored for step={step}")

    #         S = rec.innovation_covariance
    #         z = rec.measurements
    #         zhat = rec.predicted_measurements

    #         assoc = np.array(rec.associations_local)
    #         z = np.array([[r, b.theta()] for (r, b) in z], dtype=float)  # (M,2)

    #         # number of associated landmark measurements (each landmark gives 2D measurement)
    #         num_assoc = np.sum(assoc > -1)
    #         dof = 2 * num_assoc

    #         dof_sequence[k] = dof

    #         # If no associations, NIS is not meaningful (0 dof -> chi2 not defined nicely)
    #         if dof <= 0:
    #             continue

    #         nis_sequence[k] = NIS(z, zhat, S, assoc)

    #         lower, upper = chi2.interval(0.999999, df=dof) # TODO fix hardcoding 
    #         lower_bounds[k] = lower
    #         upper_bounds[k] = upper

    #     # ---- plotting ----
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)

    #     x = np.arange(N)

    #     ax.plot(x, nis_sequence, label="NIS", linewidth=1.8)
    #     ax.plot(
    #         x,
    #         lower_bounds,
    #         "--",
    #         label=r"$\chi^2_{{dof},1-\alpha_{joint}}$",
    #         linewidth=1.2,
    #     )
    #     ax.plot(
    #         x,
    #         upper_bounds,
    #         "--",
    #         label=r"$\chi^2_{{dof},\alpha_{joint}}$",
    #         linewidth=1.2,
    #     )

    #     if show_expected:
    #         # E[chi2(dof)] = dof
    #         expected = np.where(dof_sequence > 0, dof_sequence.astype(float), np.nan)
    #         ax.plot(x, expected, ":", label="E[NIS] = dof", linewidth=1.2)

    #     ax.set_title("NIS consistency over time")
    #     ax.set_xlabel("Timestep index")
    #     ax.set_ylabel("NIS")
    #     ax.grid(True, alpha=0.3)
    #     ax.legend()

    #     return fig, ax

    # def plot_NEES(self, gt_poses):
    #     steps = list(self.history.steps)
    #     N = len(steps)

    #     nees_sequence = np.full(N, np.nan, dtype=float)
    #     lower_bounds = np.full(N, np.nan, dtype=float)
    #     upper_bounds = np.full(N, np.nan, dtype=float)

    #     dof = 3  # Pose2 minimal dimension
    #     alpha = 0.95

    #     for k, step in enumerate(steps):
    #         rec = self.history.get_or_raise(step)
    #         est = rec.estimate
    #         cov = rec.cov_last_pose

    #         if est is None:
    #             continue

    #         if step >= len(gt_poses):
    #             continue

    #         pose_est = est.atPose2(X(step))
    #         pose_gt = gt_poses[step]

    #         error = pose2_to_array(pose_est.between(pose_gt))  # in minimal coordinates

    #         nees_sequence[k] = error.T @ np.linalg.inv(cov) @ error

    #         lower, upper = chi2.interval(alpha, df=dof)
    #         lower_bounds[k] = lower
    #         upper_bounds[k] = upper

    #     # ---- plotting ----
    #     fig, ax = plt.subplots(figsize=(13, 3))
    #     x = np.arange(N)
    #     ax.plot(x, nees_sequence, label="NEES", linewidth=1.8)
    #     ax.plot(
    #         x, lower_bounds, "--", label=f"Lower bound (α={alpha:g})", linewidth=1.2
    #     )
    #     ax.plot(
    #         x, upper_bounds, "--", label=f"Upper bound (α={alpha:g})", linewidth=1.2
    #     )
    #     ax.set_title("NEES consistency over time")
    #     ax.set_xlabel("Timestep index")
    #     ax.set_ylabel("NEES")
    #     ax.grid(True, alpha=0.3)
    #     ax.legend()
    #     return fig, ax

    # def plot_error(self, gt_poses):

    #     steps = list(self.history.steps)
    #     N = len(steps)

    #     # errors (x, y, theta) and sigmas
    #     err = np.full((N, 3), np.nan, dtype=float)
    #     sig = np.full((N, 3), np.nan, dtype=float)

    #     for k, step in enumerate(steps):
    #         rec = self.history.get_or_raise(step)
    #         est = rec.estimate
    #         cov = rec.cov_last_pose  # expected 3x3 in (x,y,theta) minimal coords

    #         if est is None or cov is None:
    #             continue
    #         if step >= len(gt_poses):
    #             continue

    #         pose_est = est.atPose2(X(step))
    #         pose_gt = gt_poses[step]

    #         # Minimal error coordinates: Pose2 "between" -> (dx, dy, dtheta)
    #         e = pose2_to_array(pose_est.between(pose_gt))
    #         e[2] = ssa(e[2])

    #         err[k, :] = e
    #         sig[k, :] = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    #     # ---- plotting ----
    #     labels = ["x error [m]", "y error [m]", "yaw error [rad]"]
    #     fig, axs = plt.subplots(3, 1, figsize=(13, 5.5), sharex=True)

    #     x = np.arange(N)
    #     for i, ax in enumerate(axs):
    #         ax.plot(x, err[:, i], linewidth=1.6, label="Error")

    #         # envelopes
    #         ax.fill_between(x, -2 * sig[:, i], 2 * sig[:, i], alpha=0.25, label="±2σ")
    #         ax.fill_between(x, -3 * sig[:, i], 3 * sig[:, i], alpha=0.15, label="±3σ")

    #         ax.set_ylabel(labels[i])
    #         ax.grid(True, alpha=0.3)
    #         ax.legend(loc="upper right")

    #     axs[-1].set_xlabel("Timestep index")
    #     fig.suptitle("Pose estimation error with 2σ/3σ covariance envelopes", y=0.98)
    #     fig.tight_layout()
    #     return fig, axs

    # def plot_result_step(
    #     self,
    #     step: int,
    #     marginals: Optional[gtsam.Marginals] = None,
    #     poses_gt: Optional[list[gtsam.Pose2]] = None,
    #     landmarks_gt: Optional[list[gtsam.Point2]] = None,
    #     poses_dead_reckoning: Optional[list[gtsam.Pose2]] = None,
    #     show_covariances: bool = True,
    #     show_landmarks: bool = True,
    #     axis_length: float = 0.5,
    #     figsize=(22, 6),
    #     ax=None,
    #     title: Optional[str] = None,
    #     show_orientations: bool = True,
    # ):
    #     """
    #     Plot estimate at a given step using history (StepRecord).

    #     Notes on covariances:
    #       - `marginals` must correspond to the same (graph, values) solution.
    #       - If you pass the current slam.get_marginals(), it usually corresponds to the final step.
    #     """
    #     from gtsam.utils import plot as gtsam_plot

    #     from utils.utils_plot import (
    #         MultivariateNormalParameters,
    #         plot_ellipse,
    #         plot_se2_covariance_on_manifold_gtsam,
    #     )

    #     rec = self.history.get(step)
        
    #     if rec is None:
    #         raise ValueError(f"No record for step={step}")
        
    #     est = rec.estimates

    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     else:
    #         fig = ax.figure

    #     ax.set_aspect("equal")
    #     if title is None:
    #         title = f"SLAM result at step {step}"
    #         if show_covariances and marginals is not None:
    #             title += " (with marginals)"
    #     ax.set_title(title)

    #     # ----- Plot estimated poses up to step -----
    #     x_coords = []
    #     y_coords = []
    #     for k in range(step + 1):
    #         pose_key = X(k)
    #         if not est.exists(pose_key):
    #             continue
    #         pose = est.atPose2(pose_key)
    #         x_coords.append(pose.x())
    #         y_coords.append(pose.y())
    #     ax.plot(x_coords, y_coords, "-r", label=r"$\hat{x}$")

    #     for k in range(step + 1):
    #         pose_key = gtsam.symbol('x', k)
    #         if not est.exists(pose_key):
    #             continue

    #         pose = est.atPose2(pose_key)

    #         if show_covariances and (marginals is not None):
    #             try:
    #                 cov = marginals.marginalCovariance(pose_key)
    #                 plot_se2_covariance_on_manifold_gtsam(
    #                     ax,
    #                     dist=MultivariateNormalParameters(mean=pose, covariance=cov),
    #                     fill_alpha=0.2,
    #                     fill_color="red",
    #                     linestyle="none",
    #                 )
    #                 # plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, show_axis=show_orientations)
    #                 # gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length, covariance=cov)
    #             except Exception:
    #                 gtsam_plot.plot_pose2_on_axes(
    #                     ax, pose=pose, axis_length=axis_length
    #                 )
    #         else:
    #             gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)

    #     # ----- Plot estimated landmarks (that exist in this estimate) -----
    #     if show_landmarks:
    #         # Count how many landmarks exist in this estimate
    #         est_landmark_count = 0
    #         for lm_key in slam.landmark_keys:
    #             if not est.exists(lm_key):
    #                 continue

    #             lm_pos = est.atPoint2(lm_key)
    #             est_landmark_count += 1

    #             if show_covariances and (marginals is not None):
    #                 try:
    #                     cov = marginals.marginalCovariance(lm_key)
    #                     ax.plot(lm_pos[0], lm_pos[1], "ob")
    #                     plot_ellipse(
    #                         ax,
    #                         MultivariateNormalParameters(mean=lm_pos, covariance=cov),
    #                         fill_alpha=0.2,
    #                         fill_color="blue",
    #                         linestyle="",
    #                         linewidth=0.8,
    #                     )
    #                     # gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b", P=cov)
    #                 except Exception:
    #                     gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b")
    #             else:
    #                 gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b")

    #         # Add a legend entry indicating the number of estimated landmarks
    #         try:
    #             ax.plot([], [], "ob", label=f"$\\hat{{m}}$ (#{est_landmark_count})")
    #         except Exception:
    #             pass

    #     # ----- Optional: overlay GT on same axes -----
    #     if poses_gt is not None:
    #         for pose in poses_gt[: step + 1]:
    #             plot_pose2_on_axes(
    #                 ax, pose=pose, axis_length=axis_length, marker="x", color="green"
    #             )
    #             # gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=axis_length)
    #         ax.plot([], [], "gx", label="$x_{GT}$")

    #     if landmarks_gt is not None:
    #         for lm_pos in landmarks_gt:
    #             ax.plot(lm_pos[0], lm_pos[1], "x", color="orange")
    #             # gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="gx")
    #         ax.plot([], [], "x", color="orange", label=r"$m_{GT}$")

    #     ax.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.legend()
    #     return fig, ax

    # @staticmethod
    # def plot_final_result(
    #     slam,
    #     marginals: Optional[gtsam.Marginals] = None,
    #     poses_gt: Optional[list[gtsam.Pose2]] = None,
    #     landmarks_gt: Optional[list[gtsam.Point2]] = None,
    #     poses_dead_reckoning: Optional[list[gtsam.Pose2]] = None,
    #     **kwargs,
    # ):
    #     if len(slam.history) == 0:
    #         raise ValueError("No history in slam.history")
    #     last_step = slam.history.steps[-1]
    #     return SLAMVisualizer.plot_result_step(
    #         slam,
    #         step=last_step,
    #         marginals=marginals,
    #         poses_dead_reckoning=poses_dead_reckoning,
    #         poses_gt=poses_gt,
    #         landmarks_gt=landmarks_gt,
    #         **kwargs,
    #     )

    # # @staticmethod
    # # def plot_final_result(slam: FactorGraphSLAM,
    # #                      marginals: Optional[gtsam.Marginals] = None,
    # #                      figsize=(22, 6)):
    # #     """Plot final SLAM result with covariances"""
    # #     import matplotlib.pyplot as plt
    # #     from gtsam.utils import plot as gtsam_plot

    # #     if marginals is None:
    # #         marginals = slam.get_marginals()

    # #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    # #     ax.set_aspect('equal')
    # #     ax.set_title("Nonlinear 2D SLAM with Marginals")

    # #     # Plot poses
    # #     for k in range(slam.num_poses):
    # #         pose_key = X(k)
    # #         pose = slam.values.atPose2(pose_key)
    # #         cov = marginals.marginalCovariance(pose_key)
    # #         gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5, covariance=cov)

    # #     # Plot landmarks
    # #     for lm_key in slam.landmark_keys:
    # #         lm_pos = slam.values.atPoint2(lm_key)
    # #         cov = marginals.marginalCovariance(lm_key)
    # #         gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='b', P=cov)

    # #     plt.tight_layout()
    # #     return fig, ax

    # # @staticmethod
    # # def plot_ground_truth(poses_gt: list[gtsam.Pose2],
    # #                       landmarks_gt: list[gtsam.Point2],
    # #                       figsize=(22, 6)):
    # #     """Plot ground truth trajectory and landmarks"""
    # #     import matplotlib.pyplot as plt
    # #     from gtsam.utils import plot as gtsam_plot
    # #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    # #     ax.set_aspect('equal')
    # #     ax.set_title("Ground Truth Trajectory and Landmarks")
    # #     # Plot ground truth poses
    # #     for k, pose in enumerate(poses_gt):
    # #         gtsam_plot.plot_pose2_on_axes(ax, pose=pose, axis_length=0.5)
    # #     # Plot ground truth landmarks
    # #     for lm_pos in landmarks_gt:
    # #         gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec='go')
    # #     plt.tight_layout()
    # #     return fig, ax

    # @staticmethod
    # def plot_step_by_step(
    #     slam,
    #     subplot_size: float = 4.0,
    #     axis_length: float = 0.5,
    #     margin_fraction: float = 0.2,
    #     min_margin: float = 0.5,
    # ):
    #     """
    #     Plot SLAM evolution step-by-step in a grid of subplots using StepRecords.
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     from gtsam.utils import plot as gtsam_plot

    #     plt.ioff()

    #     steps = slam.history.steps
    #     K = len(steps)
    #     if K == 0:
    #         print("No estimates to plot!")
    #         return None, None

    #     # Compute grid layout
    #     cols = int(np.ceil(np.sqrt(K)))
    #     rows = int(np.ceil(K / cols))

    #     # Compute global axis limits across all stored estimates
    #     xlim, ylim = SLAMVisualizer._compute_global_limits_from_history(
    #         slam, steps, margin_fraction, min_margin
    #     )

    #     # Create subplots
    #     fig, axes = plt.subplots(
    #         rows, cols, figsize=(subplot_size * cols, subplot_size * rows)
    #     )
    #     axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    #     # Plot each record
    #     for plot_idx, step in enumerate(steps):
    #         ax = axes_flat[plot_idx]
    #         rec = slam.history.get_or_raise(step)
    #         est = rec.estimate

    #         ax.set_aspect("equal")
    #         ax.set_title(f"Step {step} ({plot_idx}/{K - 1})")
    #         ax.set_xlabel("x [m]")
    #         ax.set_ylabel("y [m]")

    #         if est is None:
    #             ax.text(
    #                 0.5,
    #                 0.5,
    #                 "No estimate",
    #                 transform=ax.transAxes,
    #                 ha="center",
    #                 va="center",
    #             )
    #             ax.set_xlim(xlim)
    #             ax.set_ylim(ylim)
    #             ax.grid(True, alpha=0.3)
    #             continue

    #         # Plot poses up to current step
    #         for k in range(step + 1):
    #             pose_key = X(k)
    #             if est.exists(pose_key):
    #                 pose = est.atPose2(pose_key)
    #                 gtsam_plot.plot_pose2_on_axes(ax, pose, axis_length=axis_length)

    #         # Plot observed landmarks (that exist in this estimate)
    #         for lm_key in slam.landmark_keys:
    #             if est.exists(lm_key):
    #                 lm_pos = est.atPoint2(lm_key)
    #                 gtsam_plot.plot_point2_on_axes(ax, lm_pos, linespec="b")

    #         # Apply global limits
    #         ax.set_xlim(xlim)
    #         ax.set_ylim(ylim)
    #         ax.grid(True, alpha=0.3)

    #     # Hide unused axes
    #     for i in range(K, len(axes_flat)):
    #         fig.delaxes(axes_flat[i])

    #     plt.tight_layout()
    #     return fig, axes


    # @staticmethod
    # def _compute_global_limits_from_history(
    #     slam,
    #     steps,
    #     margin_fraction: float = 0.2,
    #     min_margin: float = 0.5,
    # ):
    #     xs, ys = [], []

    #     for step in steps:
    #         rec = slam.history.get(step)
    #         if rec is None or rec.estimate is None:
    #             continue
    #         est = rec.estimate

    #         # poses up to this step
    #         for k in range(step + 1):
    #             pose_key = X(k)
    #             if est.exists(pose_key):
    #                 pose = est.atPose2(pose_key)
    #                 xs.append(pose.x())
    #                 ys.append(pose.y())

    #         # landmarks present in this estimate
    #         for lm_key in slam.landmark_keys:
    #             if est.exists(lm_key):
    #                 lm = est.atPoint2(lm_key)
    #                 xs.append(lm[0])
    #                 ys.append(lm[1])

    #     if len(xs) == 0:
    #         return (-1, 1), (-1, 1)

    #     xmin, xmax = min(xs), max(xs)
    #     ymin, ymax = min(ys), max(ys)

    #     xspan = max(1e-3, xmax - xmin)
    #     yspan = max(1e-3, ymax - ymin)

    #     x_margin = max(min_margin, margin_fraction * xspan)
    #     y_margin = max(min_margin, margin_fraction * yspan)

    #     return (xmin - x_margin, xmax + x_margin), (ymin - y_margin, ymax + y_margin)

    # @staticmethod
    # def plot_trajectory_with_uncertainty(
    #     slam: FactorGraphSLAM,
    #     marginals: Optional[gtsam.Marginals] = None,
    #     show_landmarks: bool = True,
    #     figsize=(12, 8),
    # ):
    #     """
    #     Plot robot trajectory with uncertainty ellipses

    #     Args:
    #         slam: FactorGraphSLAM object
    #         marginals: Pre-computed marginals (computed if None)
    #         show_landmarks: Whether to show landmarks
    #         figsize: Figure size
    #     """
    #     import matplotlib.pyplot as plt
    #     from gtsam.utils import plot as gtsam_plot

    #     if marginals is None:
    #         marginals = slam.get_marginals()

    #     fig, ax = plt.subplots(figsize=figsize)
    #     ax.set_aspect("equal")
    #     ax.set_title("Robot Trajectory with Uncertainty")
    #     ax.set_xlabel("x [m]")
    #     ax.set_ylabel("y [m]")

    #     # Plot trajectory line
    #     trajectory_x = []
    #     trajectory_y = []
    #     for k in range(slam.num_poses):
    #         pose = slam.values.atPose2(X(k))
    #         trajectory_x.append(pose.x())
    #         trajectory_y.append(pose.y())

    #     ax.plot(
    #         trajectory_x,
    #         trajectory_y,
    #         "r--",
    #         alpha=0.5,
    #         linewidth=1,
    #         label="Trajectory",
    #     )

    #     # Plot poses with covariance
    #     for k in range(slam.num_poses):
    #         pose_key = X(k)
    #         pose = slam.values.atPose2(pose_key)
    #         cov = marginals.marginalCovariance(pose_key)
    #         gtsam_plot.plot_pose2_on_axes(
    #             ax, pose=pose, axis_length=0.5, covariance=cov
    #         )

    #     # Plot landmarks if requested
    #     if show_landmarks:
    #         for lm_key in slam.landmark_keys:
    #             lm_pos = slam.values.atPoint2(lm_key)
    #             cov = marginals.marginalCovariance(lm_key)
    #             gtsam_plot.plot_point2_on_axes(ax, point=lm_pos, linespec="b", P=cov)

    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #     plt.tight_layout()

    #     return fig, ax

    # @staticmethod
    # def plot_measurement_space_step_by_step(
    #     slam,
    #     subplot_size: float = 4.0,
    #     show_lines: bool = True,
    #     show_labels: bool = False,
    #     margin_fraction: float = 0.1,
    #     min_margin_r: float = 0.5,
    #     min_margin_b: float = 0.1,
    # ):
    #     """
    #     Plot measurement-space evolution step-by-step (range vs bearing) in a grid of subplots.

    #     Uses StepRecords in slam.history:
    #       - rec.measurements: list of (range, bearing_obj) where bearing_obj.theta() is used
    #       - rec.predicted_measurements: (N,2) array-like of [range, bearing]
    #       - rec.associations: list length M, with landmark ids or -1 for new/unassociated
    #       - rec.local_landmark_ids: list length N matching predicted_measurements
    #     """

    #     plt.ioff()

    #     steps = slam.history.steps
    #     K = len(steps)
    #     if K == 0:
    #         print("No history to plot!")
    #         return None, None

    #     # ---- global limits across all steps ----
    #     xlim, ylim = SLAMVisualizer._compute_global_meas_limits_from_history(
    #         slam,
    #         steps,
    #         margin_fraction=margin_fraction,
    #         min_margin_r=min_margin_r,
    #         min_margin_b=min_margin_b,
    #     )

    #     # ---- grid layout ----
    #     cols = int(np.ceil(np.sqrt(K)))
    #     rows = int(np.ceil(K / cols))

    #     fig, axes = plt.subplots(
    #         rows, cols, figsize=(subplot_size * cols, subplot_size * rows)
    #     )
    #     axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    #     for plot_idx, step in enumerate(steps):
    #         ax = axes_flat[plot_idx]
    #         rec = slam.history.get_or_raise(step)

    #         ax.set_title(f"Step {step} ({plot_idx}/{K - 1})")
    #         ax.set_xlabel("range [m]")
    #         ax.set_ylabel("bearing [rad]")

    #         if rec.measurements is None:
    #             ax.text(
    #                 0.5,
    #                 0.5,
    #                 "No measurements",
    #                 transform=ax.transAxes,
    #                 ha="center",
    #                 va="center",
    #             )
    #             ax.set_xlim(xlim)
    #             ax.set_ylim(ylim)
    #             ax.grid(True, alpha=0.3)
    #             continue

    #         z = np.array(
    #             [[r, b.theta()] for (r, b) in rec.measurements], dtype=float
    #         )  # (M,2)

    #         zhat = rec.predicted_measurements
    #         if zhat is None:
    #             zhat = np.empty((0, 2), dtype=float)
    #         else:
    #             zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)

    #         # predicted + measured
    #         if len(zhat) > 0:
    #             ax.scatter(zhat[:, 0], zhat[:, 1], marker="o", label="predicted")
    #         if len(z) > 0:
    #             ax.scatter(z[:, 0], z[:, 1], marker="x", label="measured")

    #         # association lines measured -> predicted
    #         assoc = rec.associations if rec.associations is not None else []
    #         local_ids = getattr(rec, "local_landmark_ids", None)

    #         if (
    #             show_lines
    #             and local_ids is not None
    #             and len(local_ids) == len(zhat)
    #             and len(assoc) == len(z)
    #         ):
    #             id_to_i = {int(lm_id): i for i, lm_id in enumerate(local_ids)}
    #             assoc_arr = np.asarray(assoc, dtype=int)
    #             new_mask = assoc_arr == -1

    #             for j, a_j in enumerate(assoc_arr):
    #                 if a_j == -1:
    #                     continue
    #                 i = id_to_i.get(int(a_j), None)
    #                 if i is None:
    #                     continue

    #                 ax.plot(
    #                     [z[j, 0], zhat[i, 0]],
    #                     [z[j, 1], zhat[i, 1]],
    #                     linewidth=1,
    #                     alpha=0.6,
    #                 )
    #                 if show_labels:
    #                     ax.text(z[j, 0], z[j, 1], f"{a_j}", fontsize=8, alpha=0.8)

    #             if np.any(new_mask):
    #                 ax.scatter(
    #                     z[new_mask, 0],
    #                     z[new_mask, 1],
    #                     marker="x",
    #                     label="unassociated (-1)",
    #                 )

    #         ax.set_xlim(xlim)
    #         ax.set_ylim(ylim)
    #         ax.grid(True, alpha=0.3)

    #         # only show legend if something was plotted
    #         handles, labels = ax.get_legend_handles_labels()
    #         if len(handles) > 0:
    #             ax.legend(fontsize=8)

    #     # Hide unused axes
    #     for i in range(K, len(axes_flat)):
    #         fig.delaxes(axes_flat[i])

    #     plt.tight_layout()
    #     return fig, axes

    # @staticmethod
    # def _compute_global_meas_limits_from_history(
    #     slam,
    #     steps,
    #     margin_fraction: float = 0.1,
    #     min_margin_r: float = 0.5,
    #     min_margin_b: float = 0.1,
    # ):
    #     import numpy as np

    #     rs, bs = [], []

    #     for step in steps:
    #         rec = slam.history.get(step)
    #         if rec is None:
    #             continue

    #         if rec.measurements is not None:
    #             z = np.array(
    #                 [[r, b.theta()] for (r, b) in rec.measurements], dtype=float
    #             )
    #             if z.size > 0:
    #                 rs.extend(z[:, 0].tolist())
    #                 bs.extend(z[:, 1].tolist())

    #         zhat = rec.predicted_measurements
    #         if zhat is not None:
    #             zhat = np.asarray(zhat, dtype=float).reshape(-1, 2)
    #             if zhat.size > 0:
    #                 rs.extend(zhat[:, 0].tolist())
    #                 bs.extend(zhat[:, 1].tolist())

    #     if len(rs) == 0:
    #         return (-1, 1), (-1, 1)

    #     rmin, rmax = float(np.min(rs)), float(np.max(rs))
    #     bmin, bmax = float(np.min(bs)), float(np.max(bs))

    #     rspan = max(1e-6, rmax - rmin)
    #     bspan = max(1e-6, bmax - bmin)

    #     r_margin = max(min_margin_r, margin_fraction * rspan)
    #     b_margin = max(min_margin_b, margin_fraction * bspan)

    #     return (rmin - r_margin, rmax + r_margin), (bmin - b_margin, bmax + b_margin)
