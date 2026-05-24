import matplotlib.pyplot as plt
import numpy as np

from master_code.plotter import SlamRunPlotter
from master_code.plotting.thesis_style import apply_thesis_style, save_figure


def main():
    apply_thesis_style()

    f_run = SlamRunPlotter.from_run('runs/20260521_191710_forward')
    b_run = SlamRunPlotter.from_run('runs/20260521_191137_backward')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)

    f_run.plot_final_snapshot(ax=axes[0], title="Forward",  show_legend=False, equal_aspect=False, show_grid=True, x_label=None, y_label=None)
    b_run.plot_final_snapshot(ax=axes[1], title="Backward", show_legend=False, equal_aspect=False, show_grid=True, x_label=None, y_label=None)
    save_figure(fig, "forward_backward_estimate")
    plt.show()


if __name__ == "__main__":
    main()



