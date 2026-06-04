from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from master_code.plotter import SlamRunPlotter
from master_code.plotting.thesis_style import apply_thesis_style
from master_code.paths import FIGURES_ROOT


def add_basemap(
    ax,
    image_path,
    extent=None,       # [x_min, x_max, y_min, y_max] in TRAJECTORY (world) coords
    center=None,       # (x, y) world centre; use WITH `scale` instead of `extent`
    scale=None,        # metres per pixel; one knob to size the map (keeps aspect)
    rotation_deg=0.0,  # rotate about the image centre if north isn't axis-aligned
    alpha=1.0,
    zorder=-10,
):
    """Underlay a georeferenced raster (satellite tile) behind a plot.

    Size/position the image one of two ways:
      * `extent=[x_min, x_max, y_min, y_max]` -- world coords of the edges, or
      * `center=(x, y)` + `scale` (metres per pixel) -- the extent is derived
        from the image's pixel size so the aspect ratio is always preserved;
        shrink `scale` to make the map smaller.

    Calibrate against the GNSS path: overlay with alpha<1 and tweak the numbers
    until the GNSS track follows the roads, then restore alpha=1.0.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()  # imshow would clobber these

    img = plt.imread(image_path)

    if extent is None:
        if center is None or scale is None:
            raise ValueError("pass either `extent`, or both `center` and `scale`")
        h, w = img.shape[:2]
        half_w = 0.5 * w * scale
        half_h = 0.5 * h * scale
        extent = [
            center[0] - half_w, center[0] + half_w,
            center[1] - half_h, center[1] + half_h,
        ]

    im = ax.imshow(
        img,
        extent=extent,
        origin="upper",   # row 0 of the JPG is the top (max-y) edge
        alpha=alpha,
        zorder=zorder,    # negative -> behind trajectory/landmarks
        interpolation="bilinear",
    )

    if rotation_deg:
        cx = 0.5 * (extent[0] + extent[1])
        cy = 0.5 * (extent[2] + extent[3])
        im.set_transform(
            Affine2D().rotate_deg_around(cx, cy, rotation_deg) + ax.transData
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    return im


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
    axins = ax.inset_axes([0.68, 0.01, 0.36, 0.36])
    plotter.plot_final_snapshot(
        ax=axins,
        show_legend=False,
        show_grid=False,
        title="",
        x_label="",
        y_label="",
    )
    axins.set_xlim(-165, -125)
    axins.set_ylim(120, 180)
    axins.set_xticks([])
    axins.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Georeferenced satellite underlay. CALIBRATE these four numbers (and the
    # rotation, if the image isn't north-up in your frame) against the GNSS path:
    # set alpha=0.5 and adjust until the GNSS track follows the roads, then
    # restore alpha=1.0. A good starting guess is the trajectory's x/y bounds.
    add_basemap(
        ax,
        "data/victoria_park/victoria_park.jpg",
        extent=[-250, 115, -150, 245],  # [x_min, x_max, y_min, y_max] -> TUNE ME
        rotation_deg=0.0,               # -> TUNE ME if roads look rotated
        alpha=0.4,                      # -> set to 1.0 once aligned
    )

    fig.savefig(FIGURES_ROOT / "real_final_snapshot_zoom.pdf", dpi=200, bbox_inches="tight")
    plt.show()

def main() -> None:
    # inset_sim()
    inset_real()


if __name__ == "__main__":
    main()
