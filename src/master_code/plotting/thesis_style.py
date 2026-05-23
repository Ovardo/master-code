import matplotlib.pyplot as plt

from master_code.paths import FIGURES_ROOT

TEXT_WIDTH = 8
SMALL_WIDTH = 0.75 * TEXT_WIDTH
HALF_WIDTH = 0.5 * TEXT_WIDTH

DISPLAY_DPI = 120
SAVE_DPI = 300

def thesis_figsize(width="text", ratio=0.62):
    widths = {
        "text": TEXT_WIDTH,
        "full": TEXT_WIDTH,
        "single": TEXT_WIDTH,
        "small": SMALL_WIDTH,
        "half": HALF_WIDTH,
    }
    w = widths[width] if isinstance(width, str) else float(width)
    return (w, w * ratio)

def apply_thesis_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1,
        "lines.markersize": 4,
        "figure.dpi": DISPLAY_DPI,
        "savefig.dpi": SAVE_DPI,
        # "savefig.bbox": "tight",
    })

def save_figure(fig, name):
    path = FIGURES_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=SAVE_DPI)
