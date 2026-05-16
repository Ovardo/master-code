# Master's Thesis

This repository contains the code developed for **TTK4900 – Master’s Thesis in Engineering Cybernetics** at **NTNU**.
 

## Setup

The project now uses `pyproject.toml` as the main source of dependency metadata.

### Requirements

- Python `3.10+`
- `pip` and `venv`
- `ffmpeg` if you want to export videos from the visualization tools
- `gtsam` for the main SLAM pipelines and examples

For a student-facing assignment release, the recommended support matrix should be narrower than the development target:

- Python `3.11`
- macOS and Linux
- Windows through WSL2 unless native Windows wheels are explicitly validated

### Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install `gtsam` first. This dependency is the main non-trivial requirement in the project:

- To install the stable 4.2 releas of gtsam you can run:

```bash
python -m pip install gtsam-ttk4250
```

- If you are using a patched GTSAM fork with custom Python wrappers, build and install that fork first. After installing this project, verify the wrapper with:

```bash
check-gtsam-install --expect gtsam.ISAM2.jointMarginalCovariance
```

That check confirms that the patched `ISAM2.jointMarginalCovariance(...)` Python wrapper is available.

Install the project in editable mode:

```bash
python -m pip install -e .
```

Optional extras:
```bash
python -m pip install -e .[dev]
```

If this repository is turned into a student assignment, the intended end-state is to ship prebuilt wheels for the patched GTSAM fork instead of asking students to build GTSAM from source on their own machines. The release plan and CI template are documented in [docs/gtsam_distribution_plan.md](docs/gtsam_distribution_plan.md).

## Entry Points

The main runnable scripts are:

- `run-real-slam`
  Runs the Victoria Park SLAM pipeline defined in `src/run_real_SLAM.py`. Supports `--config` and defaults to `src/conf/victoria_park_config.yaml`.
- `toy-example`
  Runs the synthetic comparison example in `src/toy_example.py`.

You can also run the same entry points directly with Python:

```bash
python src/run_real_SLAM.py
python src/toy_example.py
```

Select a different OmegaConf YAML file for the real-data pipeline with:

```bash
run-real-slam --config src/conf/victoria_park_config.yaml
run-real-slam --config src/conf/sandbox/default_config.yaml
python src/run_real_SLAM.py --config src/conf/victoria_park_config.yaml
```

Notes:

- `run-real-slam` expects the Victoria Park dataset already present under `data/victoria_park/`.
- Video export paths under `videos/` require `ffmpeg` because the visualization code uses Matplotlib's `FFMpegWriter`.

## Structure
```text
.
├── data/             # Victoria Park dataset files and small data utilities
├── figures/          # Exported figures used for analysis and thesis material
├── src/              # Core SLAM implementation
│   ├── conf/         # YAML configuration files
│   ├── div/          # Scratch scripts and notebooks
│   ├── models/       # Motion and measurement models
│   ├── plotting/     # Plot helpers
│   ├── simulation/   # Synthetic data generation
│   └── utils/        # Shared utilities
├── videos/           # Generated animations
├── README.md
└── TODO.md           # Active backlog and follow-up tasks
```

Active follow-up work is tracked in [TODO.md](TODO.md).
