import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from master_code.plotting.thesis_style import thesis_figsize, save_figure, apply_thesis_style
from master_code.data_loader import VictoriaParkLoader 
from master_code.plotting.plotting_funcs import plot_position_nis




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