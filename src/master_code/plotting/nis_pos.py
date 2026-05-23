import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from master_code.plotting.thesis_style import thesis_figsize, save_figure, apply_thesis_style
from master_code.data_loader import VictoriaParkLoader 

def plot_position_nis(
    gnss: np.ndarray,
    poses: np.ndarray,
    poses_covs: np.ndarray,
    poses_times: np.ndarray,
    figsize = (8, 4)
) -> plt.Figure:
    """Plot NIS between GNSS samples and nearest-in-time pose estimates."""
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=figsize)
 
    pose_xy = poses[:, :2]
    covs_xy = poses_covs[:, :2, :2]

    if len(poses_times) != len(pose_xy):
        raise ValueError("pose_times must have the same length as poses.")
    if len(poses_times) == 0:
        raise ValueError("At least one pose is required.")
    if len(gnss) == 0:
        raise ValueError("At least one GNSS measurement is required.")

    nearest_pose_indices = _nearest_indices(gnss[:, 0], poses_times)
    matched_pose_xy = pose_xy[nearest_pose_indices]
    matched_covs_xy = covs_xy[nearest_pose_indices] + np.eye(2) * (1.0**2)    # Adding 1m GNSS covariance for NIS calculation

    innovation_xy = matched_pose_xy - gnss[:, 1:3]
    nis_xy = np.einsum("ij,ijk,ik->i", innovation_xy, np.linalg.inv(matched_covs_xy), innovation_xy)
    anis_xy = np.sum(nis_xy) / len(nis_xy)
   
    ax.scatter(nearest_pose_indices, nis_xy, s=5, color="steelblue", label=r"$\mathrm{NIS}_{xy}$")
    ax.axhline(chi2.isf(1-0.95, 2), ls="--", c="tomato", lw=1, label=r"$\chi^2_{2,0.95}$")
    ax.axhline(chi2.isf(1-0.05, 2), ls="--", c="orange", lw=1, label=r"$\chi^2_{2,0.05}$")
    ax.set_xlabel("Scan step")
    ax.set_ylabel("NIS")
    ax.grid(True, lw=0.4)
    ax.legend()
    ax.set_title(f"Position NIS (ANIS = {anis_xy:.2f})")
    return fig


def main():
    loader = VictoriaParkLoader()
    gnss = loader.gnss_filtered
    
    run_dirs = [
        'runs/20260521_191137_backward', 
        'runs/20260521_191710_forward',
        'runs/20260521_192630_diverging_1000'
    ]
    save_name = [
        "nis/position/nis_pos_backward", 
        "nis/position/nis_pos_forward",
        "nis/position/nis_pos_diverging"
    ]

    apply_thesis_style()

    for run_dir, save_name in zip(run_dirs, save_name):
        snapshot = np.load(f'{run_dir}/snapshots/snap_final.npz')
        steps = np.load(f'{run_dir}/steps.npz')
        
        poses = snapshot['poses']  # shape (K, 3) 
        poses_covs = snapshot['poses_covariance']  # shape (K, 3, 3)
        poses_times = steps['scan_time']  # shape (K,)

        fig = plot_position_nis(
            gnss=gnss,  
            poses=poses, 
            poses_covs=poses_covs,
            poses_times=poses_times,
            figsize=thesis_figsize("text", ratio=0.3)
        )
        save_figure(fig, name=save_name)
    
    plt.show()


if __name__ == "__main__":
    main()