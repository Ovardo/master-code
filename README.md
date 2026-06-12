# Factor Graph SLAM

This repository contains the code developed for the TTK4900 Master's Thesis in
Engineering Cybernetics at NTNU.

The project implements landmark SLAM for the Victoria Park dataset using GTSAM
iSAM2 as the factor graph backend. The main contribution is the use of Joint Compatibility Branch
and Bound (JCBB) data association in a factor-graph-based framework, where joint marginal covariances are not as readily available as in traditional EKF-SLAM.

![Trajectory](figures/master/trajectory.png)

## Setup

The project requires Python 3.11 or newer and uses `uv` for dependency
management.

```sh
uv sync
```

This creates `.venv` and installs the project in editable mode. More information
about GTSAM is available in [docs/INSTALL.md](docs/INSTALL.md).

## Running SLAM

Commands should be run from the repository root.

```sh
uv run run_real
uv run run_sim
```

Both commands accept explicit paths and run options:

```sh
uv run run_real \
  --config configs/default_real.yaml \
  --steps 7300 \
  --output-dir runs/real/example

uv run run_sim \
  --config configs/default_sim.yaml \
  --steps 1000 \
  --output-dir runs/sim/example
```

Plots are displayed and saved by default. Use `--no-show-plots` to suppress the
interactive plot windows, or `--no-save-plots` to skip figure generation.

Plot an existing run with:

```sh
uv run plot_run runs/sim/example
```

## Project Structure

```text
.
├── configs/                 User-editable YAML run configurations
├── data/                    Victoria Park and simulated datasets
├── docs/                    Installation and supporting documentation
├── figures/                 Curated figures used in the thesis
├── notebooks/               Exploratory notebooks
├── scripts/
│   ├── experiments/         Benchmark and parameter-sweep programs
│   └── plots/               Standalone thesis figure and video programs
├── src/master_code/
│   ├── config.py            Typed configuration and YAML loading
│   ├── data_association.py  JCBB and compatibility tests
│   ├── landmark_manager.py  Tentative landmark management
│   ├── loaders/             Dataset adapters
│   ├── plotting/            Reusable plotting helpers
│   ├── preprocessing.py     Odometry and lidar preprocessing
│   ├── slam.py              Main factor graph SLAM loop
│   ├── run_real.py          Victoria Park entry point
│   └── run_sim.py           Simulated-data entry point
├── tests/                   Focused unit and loader tests
├── archive/legacy/          Unmaintained exploratory Python code
├── runs/                    Generated run outputs, ignored by Git
└── videos/                  Generated videos, ignored by Git
```

Only reusable runtime code belongs under `src/master_code`. Research programs
consume that package from `scripts/`, while old exploratory code is retained
under `archive/legacy` for reference.

## Configuration

Configuration schemas and validation live in
[`src/master_code/config.py`](src/master_code/config.py). YAML files in
[`configs/`](configs/) override the dataclass defaults.

Saved runs contain a copy of their resolved configuration as `config.yaml` for
reproducibility.

## Development

Run the tests using the project environment:

```sh
uv run pytest
```

After changing dependencies:

```sh
uv lock
uv sync
```
