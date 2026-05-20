# Master's Thesis

This repository contains the code developed for **TTK4900 – Master’s Thesis in Engineering Cybernetics** at **NTNU**. 

This project implements landmark-SLAM on the Victoria Park dataset using a
factor graph formulation in GTSAM as depicted in the figure below. The main focus is running Joint Compatibility Branch and Bound (JCBB) data association in a graph-based SLAM pipeline.

![Trajectory](figures/master/trajectory.png)

## Setup

This project uses `uv` for dependency management. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`.

### Requirements

- Python 3.11 or newer
- `uv`
- Victoria Park dataset files available under `data/`

### Install

From the repository root, install the locked dependencies:

```sh
uv sync
```

This creates a local `.venv` and installs the project dependencies, including
`gtsam-develop`, `numpy`, `scipy`, `matplotlib`, `omegaconf`, and `tqdm`.

Run the main SLAM pipeline with:

```sh
uv run python src/run.py
```

After changing dependencies in `pyproject.toml`, update the lockfile with:

```sh
uv lock
uv sync
```

## Structure of project

TODO
