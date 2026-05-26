from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from master_code.config import SlamConfig
from master_code.data_loader import VictoriaParkLoader
from master_code.logger import SlamLogger
from master_code.plotting.plotting_funcs import (
    plot_error,
    plot_estimate,
    plot_landmark_growth,
    plot_landmark_nis,
    plot_pose_covariance_evolution,
    plot_position_nis,
    plot_timing_breakdown,
    plot_timing_over_time,
    plot_timing_vs_landmarks,
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
    
    @classmethod
    def from_run(cls, run_dir: Path | str) -> SlamRunPlotter:
        run_path = Path(run_dir)
        steps       = SlamLogger.load_steps(run_path)
        snapshots   = SlamLogger.load_all_snapshots(run_path)
        association = SlamLogger.load_all_association_diagnostics(run_path)
        config      = SlamConfig.load(run_path / "config.yaml")
        
        return cls(run_dir=run_dir, steps=steps, snapshots=snapshots, association=association, config=config)
    
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
            gnss = VictoriaParkLoader().gnss_filtered,
            poses_cov_stride = 20,
            **kwargs
        )
        return fig, ax
        
    def plot_timing_breakdown(self, axes = None) -> tuple[plt.Figure, plt.Axes]:
        fig, axes = plot_timing_breakdown(
            axes    = axes,
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            t_assoc = self.steps.get("duration_association"),
            t_opt   = self.steps.get("duration_optimization"),
            t_lmap  = self.steps.get("duration_local_landmark_extraction"),
            t_total = self.steps.get("duration_update"),
            # t_tent  = self.steps.get("duration_tentative_processing"),
            # t_inno  = self.steps.get("duration_innovation_covariance"),
            # t_scan  = self.steps.get("duration_scan_processing"),
            
        )
        return fig, axes
    
    def plot_timing_over_time(self, axes = None) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        fig, axes = plot_timing_over_time(
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
    
    def plot_error_over_time(self, ax = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_error(
            ax         = ax,
            scan_steps = self.steps.get("scan_step"),
            error      = self.steps.get("factor_graph_error"),
            n_factors  = self.steps.get("num_factors"),
            **kwargs,
        )
        return fig, ax

    def plot_pose_covariance_evolution(self, axes = None) -> tuple[plt.Figure, np.ndarray]:
        fig, axes = plot_pose_covariance_evolution(
            axes = axes,
            steps = self.steps.get("scan_step"),
            covs = self.snapshots[-1].get("poses_covariance"),
            poses = self.snapshots[-1].get("poses"),
        )
        return fig, axes

    def plot_position_nis(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_position_nis(
            ax = ax,
            gnss=VictoriaParkLoader().gnss_filtered,
            poses=self.snapshots[-1].get("poses"),
            poses_covs=self.snapshots[-1].get("poses_covariance"),
            poses_times=self.steps.get("scan_time"),
            
        )
        return fig, ax
    
    def plot_landmark_nis(self, ax = None) -> tuple[plt.Figure, plt.Axes] | None:
        if not self.association:
            print("No association diagnostics found, skipping NIS plot.")
            return None
        
        fig, ax = plot_landmark_nis(
            diagnostics=self.association,
            alpha_individual=self.config.association.alpha_individual,
            alpha_joint=self.config.association.alpha_joint,
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
        self._finish_figure( self.plot_timing_breakdown(), name="timing_breakdown", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_timing_over_time(), name="timing_over_time", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_timing_vs_landmarks(), name="timing_vs_landmarks", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_landmark_growth(), name="landmark_growth", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_error_over_time(), name="error_over_time", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_pose_covariance_evolution(), name="pose_covariance_evolution", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_landmark_nis(), name="landmark_nis", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_position_nis(), name="position_nis", save=save, show=show, fmt="pdf" )
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved SLAM run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    plotter = SlamRunPlotter.from_run(run_dir)
    plotter.plot_all(save=True, show=args.show)
    

if __name__ == "__main__":
    apply_thesis_style()

    # plotter = SlamRunPlotter.from_run(Path('runs/20260523_213633_normal'))
    # plotter.plot_all(save=True, show=True)

    main()
    
