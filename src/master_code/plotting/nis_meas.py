import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from master_code.plotting.thesis_style import thesis_figsize, save_figure, apply_thesis_style
from master_code.logger import SlamLogger
from master_code.data_association import NIS



def main():
    run_dirs = [
        # 'runs/20260521_191137_backward', 
        # 'runs/20260521_191710_forward',
        # 'runs/20260521_192630_diverging_1000',
        'runs/20260521_203732_all'
    ]
    save_name = [
        # "nis/landmark/nis_meas_backward", 
        # "nis/landmark/nis_meas_forward",
        # "nis/landmark/nis_meas_diverging"
        "nis/landmark/nis_meas_all"
    ]

    apply_thesis_style()

    for run_dir, save_name in zip(run_dirs, save_name):

        diagnostics = SlamLogger.load_all_association_diagnostics(run_dir)

        nis_values = np.empty(len(diagnostics), dtype=float)
        dofs       = np.empty(len(diagnostics), dtype=int)
        scan_steps = np.empty(len(diagnostics), dtype=int)

        for i, diag in enumerate(diagnostics):
            nis = NIS(
                z=diag["measurements"],
                zbar=diag["predicted_measurements"],
                S=diag["innovation_covariance"],
                a=diag["association"]
            )
            nis_values[i] = nis
            dofs[i] = 2 * np.sum(diag["association"] >= 0)
            scan_steps[i] = diag["scan_step"][0]

        alpha_joint = 0.99999
        upper_bound = np.nan_to_num(chi2.isf(1-alpha_joint, dofs))
        lower_bound = np.nan_to_num(chi2.isf(alpha_joint, dofs))

        fig, ax = plt.subplots(figsize=thesis_figsize())
        ax.plot(scan_steps, nis_values, color="steelblue", label=r"$\mathrm{NIS}_{\mathrm{joint}}$")
        ax.fill_between(range(len(upper_bound)), lower_bound, upper_bound, alpha=0.2, color="gray", label=r"$\chi^2_{\mathrm{joint}}$ bounds")
        # ax.plot(upper_bound, ls="--", c="tomato", lw=1, label=r"$\chi^2_{\mathrm{joint},\alpha}$")
        # ax.plot(lower_bound, ls="--", c="orange", lw=1, label=r"$\chi^2_{\mathrm{joint},1-\alpha}$")
    
        save_figure(fig, name=save_name)
    
    plt.show()


if __name__ == "__main__":
    main()