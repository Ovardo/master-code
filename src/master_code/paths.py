from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs"
FIGURES_ROOT = PROJECT_ROOT / "figures"
VIDEOS_ROOT = PROJECT_ROOT / "videos"