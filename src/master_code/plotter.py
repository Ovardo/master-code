from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from master_code.loaders.victoria_park import VictoriaParkLoader
from master_code.loaders.simulated import SimulatedDataLoader
from master_code.config import SlamConfig
from master_code.logger import SlamLogger
from master_code.plotting.plotting_funcs import (
    plot_estimate,
    plot_pose_diagnostics,
    plot_pose_nees,
    plot_position_nis,
    plot_landmark_growth,
    plot_landmark_nis,
    plot_landmarks_and_timing,
    plot_timing_vs_landmarks,
    plot_cumulative_timing,
    plot_per_step_timing,
)
from master_code.plotting.thesis_style import apply_thesis_style


@dataclass
class SlamRunPlotter:
    """Utility class for plotting SLAM runs from saved logs and snapshots."""
    run_dir: Path
    steps: dict[str, np.ndarray]
    snapshots: list[dict[str, np.ndarray]]
    association: list[dict[str, np.ndarray]]
    config: SlamConfig
    gnss: np.ndarray | None = None # Inject if real-run
    gt_poses: np.ndarray | None = None # Inject if sim-run
    gt_landmarks: np.ndarray | None = None # Inject if sim-run

    
    @classmethod
    def from_run(cls, run_dir: Path | str) -> SlamRunPlotter:
        
        run_path = Path(run_dir)
        run_type = run_path.parent.name
        
        gnss = None
        gt_poses = None
        gt_landmarks = None

        if run_type == "sim":
            loader = SimulatedDataLoader()
            gt_poses = loader.poses_gt
            gt_landmarks = loader.landmarks_gt
        elif run_type == "real":
            loader = VictoriaParkLoader()
            gnss = loader.gnss_filtered
       
        return cls(
            run_dir      = run_path,
            steps        = SlamLogger.load_steps(run_path),
            snapshots    = SlamLogger.load_all_snapshots(run_path),
            association  = SlamLogger.load_all_association_diagnostics(run_path),
            config       = SlamConfig.load(run_path / "config.yaml"),
            gnss         = gnss,
            gt_poses     = gt_poses,
            gt_landmarks = gt_landmarks,
        )
    
    @property
    def figure_dir(self) -> Path:
        path = self.run_dir / "figures"
        path.mkdir(exist_ok=True)
        return path
    
    def _finish_figure(self, result, name: str, save: bool, show: bool = True, fmt: str = "pdf") -> None:
        if result is None:
            return
        
        fig = result[0] if isinstance(result, tuple) else result
        if save:
            path = self.figure_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {path}")
        if not show:
            plt.close(fig)

    def plot_final_snapshot(self, ax = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_estimate(
            ax            = ax,
            poses         = self.snapshots[-1].get("poses"),
            poses_cov     = self.snapshots[-1].get("poses_covariance"),
            landmarks     = self.snapshots[-1].get("landmarks"),
            landmarks_cov = self.snapshots[-1].get("landmarks_covariance"), 
            gnss          = self.gnss,
            poses_gt      = self.gt_poses,
            landmarks_gt  = self.gt_landmarks,
            poses_cov_stride = 20,
            **kwargs
        )
        return fig, ax
        
    def plot_cumulative_timing(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, axes = plot_cumulative_timing(
            ax      = ax,
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            t_assoc = self.steps.get("duration_association"),
            t_opt   = self.steps.get("duration_optimization"),
            t_lmap  = self.steps.get("duration_local_landmark_extraction"),
            t_total = self.steps.get("duration_update"),         
        )
        return fig, axes
    
    def plot_per_step_timing(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, axes = plot_per_step_timing(
            ax      = ax,
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            t_assoc = self.steps.get("duration_association"),
            t_opt   = self.steps.get("duration_optimization"),
            t_lmap  = self.steps.get("duration_local_landmark_extraction"),
            t_total = self.steps.get("duration_update"),         
        )
        return fig, axes
    
    def plot_landmarks_and_timing(self, axes = None) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        fig, axes = plot_landmarks_and_timing(
            axes    = axes,
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            n_local = self.steps.get("num_local_landmarks"),
        )
        return fig, axes

    def plot_timing_vs_landmarks(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_timing_vs_landmarks(
            ax      = ax,
            n_local = self.steps.get("num_local_landmarks"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
        )
        return fig, ax

    def plot_landmark_growth(self, ax = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_landmark_growth(
            ax          = ax,
            steps       = self.steps.get("scan_step"),
            n_landmarks = self.steps.get("num_landmarks"),
            **kwargs
        )
        return fig, ax
    
    def plot_pose_diagnostics(self, axes = None) -> tuple[plt.Figure, np.ndarray]:
        fig, axes = plot_pose_diagnostics(
            axes = axes,
            poses = self.snapshots[-1].get("poses"),
            poses_covs = self.snapshots[-1].get("poses_covariance"),
            poses_gt = self.gt_poses,
            cov_frame = 'world',
        )
        return fig, axes
    
    
    def plot_pose_nees(self, axes = None) -> tuple[plt.Figure, plt.Axes] | None:
        if self.gt_poses is None:
            print("No ground truth poses found, skipping NEES plot.")
            return None
        
        fig, axes = plot_pose_nees(
            axes = axes,
            poses = self.snapshots[-1].get("poses"),
            poses_covs = self.snapshots[-1].get("poses_covariance"),
            poses_gt = self.gt_poses,
        )
        return fig, axes


    def plot_position_nis(self, ax = None) -> tuple[plt.Figure, plt.Axes] | None:
        if self.gnss is None:
            print("No GNSS reference found, skipping position NIS plot.")
            return None

        fig, ax = plot_position_nis(
            ax          = ax,
            gnss        = self.gnss,
            poses       = self.snapshots[-1].get("poses"),
            poses_covs  = self.snapshots[-1].get("poses_covariance"),
            poses_times = self.steps.get("scan_time"),
        )

        return fig, ax

    def plot_landmark_nis(self, ax = None) -> tuple[plt.Figure, plt.Axes] | None:
        if not self.association:
            print("No association diagnostics found, skipping NIS plot.")
            return None
        
        fig, ax = plot_landmark_nis(
            diagnostics=self.association,
        )
        return fig, ax
    
    
    # def plot_association(self, save: bool = True, fmt: str = "pdf") -> None:
    #     if not self.association:
    #         print("No association diagnostics found, skipping association plot.")
    #         return
        
    #     for diag in self.association:
    #         scan_step = int(diag["scan_step"][0])
    #         fig = plot_association(
    #             diagnostics=diag,
    #             show_covariances=True,
    #             alpha_individual=self.config.association.alpha_individual,
    #             alpha_joint=self.config.association.alpha_joint,
    #         )
    #         self._finish_figure(fig, f"association_scan_{scan_step:03d}", save, show=False)

    
    def plot_all(self, save: bool = True, show: bool = True):
        self._finish_figure( self.plot_final_snapshot(), name="final_snapshot", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_pose_diagnostics(), name="pose_diagnostics", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_pose_nees(), name="pose_nees", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_landmark_nis(), name="landmark_nis", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_position_nis(), name="position_nis", save=save, show=show, fmt="pdf" )
        # self._finish_figure( self.plot_cumulative_timing(), name="timing_breakdown", save=save, show=show, fmt="pdf" )
        # self._finish_figure( self.plot_per_step_timing(), name="timing_over_time", save=save, show=show, fmt="pdf" )
        # self._finish_figure( self.plot_timing_vs_landmarks(), name="timing_vs_landmarks", save=save, show=show, fmt="pdf" )
        # self._finish_figure( self.plot_landmark_growth(), name="landmark_growth", save=save, show=show, fmt="pdf" )
        plt.show()


def main() -> None:
    apply_thesis_style()
    
    parser = argparse.ArgumentParser(description="Plot a saved SLAM run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    plotter = SlamRunPlotter.from_run(run_dir)
    plotter.plot_all(save=True, show=args.show)
    

if __name__ == "__main__":
    # plotter = SlamRunPlotter.from_run("runs/real/20260527_224117")
    # plotter.plot_all(save=True, show=True)
    
    main()
    
