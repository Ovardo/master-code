# Master's Thesis

This repository contains the code developed for **TTK4900 – Master’s Thesis in Engineering Cybernetics** at **NTNU**. 

This project implements landmark-SLAM on the Victoria Park dataset using a iSAM2 from GTSAM as backend. The main focus is on running Joint Compatibility Branch and Bound (JCBB) data association in a graph-based SLAM pipeline.

![Trajectory](figures/master/trajectory.png)

## Setup

This project uses `uv` for dependency management. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`.

### Requirements

- Python 3.11 or newer
- `uv`

### Install

From the repository root, install the locked dependencies:

```sh
uv sync
```

This creates a local `.venv` and installs the project dependencies, including
`gtsam-develop`, `numpy`, `scipy`, `matplotlib`, `omegaconf`, and `tqdm`.

Run the main SLAM pipeline on real data with:
```sh
run_real
```
For simulated data, run:
```sh
run_sim
```

After changing dependencies in `pyproject.toml`, update the lockfile with:

```sh
uv lock
uv sync
```


## Structure of project

See [`src/master_code/run_real.py`](src/master_code/run_real.py) for entry point ...  
TODO








