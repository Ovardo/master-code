from pathlib import Path

import matplotlib.pyplot as plt

from master_code.plotter import SlamRunPlotter
from master_code.plotting.thesis_style import apply_thesis_style
from master_code.paths import FIGURES_ROOT


def inset_sim():
    apply_thesis_style()

    run_dir = Path("runs/sim/20260603_214149")

    plotter = SlamRunPlotter.from_run(run_dir)

    # Full MAP estimate (keeps legend/title/labels).
    fig, ax = plotter.plot_final_snapshot()
    # Ensure legend is placed in the top-right corner on the axes
    ax.legend(loc="upper right")
    

    # Zoomed-in inset in the lower-right corner, re-using the same data.
    axins = ax.inset_axes([0.65, 0.01, 0.34, 0.34])
    plotter.plot_final_snapshot(
        ax=axins,
        show_legend=False,
        show_grid=False,
        title="",
        x_label="",
        y_label="",
    )
    axins.set_xlim(-81, -41)
    axins.set_ylim(41, 81)
    axins.set_xticks([])
    axins.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="gray")

    fig.savefig(FIGURES_ROOT / "sim_final_snapshot_zoom.pdf", dpi=200, bbox_inches="tight")
    plt.show()

def inset_real():
    apply_thesis_style()

    run_dir = Path('runs/real/20260603_235251')

    plotter = SlamRunPlotter.from_run(run_dir)

    # Full MAP estimate (keeps legend/title/labels).
    fig, ax = plotter.plot_final_snapshot(legend_loc="upper right")
    ax.legend(loc="upper right")

    # Zoomed-in inset in the lower-right corner, re-using the same data.
    axins = ax.inset_axes([0.65, 0.01, 0.34, 0.34])
    plotter.plot_final_snapshot(
        ax=axins,
        show_legend=False,
        show_grid=False,
        title="",
        x_label="",
        y_label="",
    )
    axins.set_xlim(-81, -41)
    axins.set_ylim(41, 81)
    axins.set_xticks([])
    axins.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="gray")

    fig.savefig(FIGURES_ROOT / "real_final_snapshot_zoom.pdf", dpi=200, bbox_inches="tight")
    plt.show()

def main() -> None:
    # inset_sim()
    inset_real()


if __name__ == "__main__":
    main()
