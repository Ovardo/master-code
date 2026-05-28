# AGENTS.md

## Python environment
Use the python enviroment at ".venv" for running code

## Project overview
This repository contains the codebase for my master thesis project on SLAM. The main focus is on implementing a factor graph-based SLAM system using GTSAM, with a custom data pipeline and association logic. The code is organized into several modules:
- `src/slam.py`: Contains the main SLAM implementation, including the `GraphSLAM` class which manages the factor graph and the main SLAM loop.
- `src/run.py`: The entry point for running the SLAM system. It sets up the configuration, loads the data, and executes the main loop while logging results.
- `src/data_loader.py`: Contains the `VictoriaParkLoader` class which handles loading and synchronization of the Victoria Park dataset.
- `src/logger.py`: Contains the `SlamLogger` class which manages loading and saving of SLAM results and snapshots to disk.
- `src/preprocessing.py`: Contains utilities for preprocessing the raw dataset, ie. converting wheel odometry to relative poses and lidar scans to tree detections.
- `src/sensor.py`: Contains classes for range-bearing sensor, handles creation of jacboains and joint innovation covariances.
- `src/association.py`: Contains the JCBB association logic, including gating.
- `src/tentative.py`: Contains logic for managing tentative landmarks and their promotion to confirmed landmarks.
- `src/plotting.py`: Contains functionality for visualization of results from the SLAM runs.
- `src/utils.py`: Contains various utility functions used across the codebase.
- `src/config/`: Contains the configuration management for the SLAM system.
- `src/div/`: Contains exploratory scripts and toy examples for testing and development, migth contain deprecated parts.
- `src/plot/`: Contains div plotting scripts for producing plots relevant to the thesis
- `data/`: Contains the datasets used for testing and validation, currently only the Victoria Park dataset.
- `runs/`: This is where the results of SLAM runs are saved, including logs, snapshots, and figures. Each run gets its own timestamped subdirectory.


## Design principles and architecture
Adhere to the KISS principle. Do not make things more complicated than they need to be. Focus on a clean that is easy to understand and maintain. Avoid over-engineering or premature optimization. Keep in mind that this codebase might be used as a baseline for factor graph based SLAM assigments at universtiy level. The pedagogical value of the codebase is therefore important.
