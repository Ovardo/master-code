from pathlib import Path
from datetime import datetime

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs"
FIGURES_ROOT = PROJECT_ROOT / "figures"
VIDEOS_ROOT = PROJECT_ROOT / "videos"


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    path = Path(output_dir)
    if path.is_absolute():
        return path
    return RUNS_ROOT / path