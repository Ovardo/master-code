# TODO

Active backlog for the thesis codebase. Deprecated or already completed items from `README.md` have been removed.

## Simulation And Data Pipeline

- [ ] Add panel holoviz visualization.
- [ ] Replace the remaining dict-based simulator output with typed step/event objects, similar to `SLAMStep`.
- [ ] Introduce a timestamp-sorted event stream for simulated odometry and landmark measurements so the main loop can process them causally.
- [ ] Implement `DynamicRobotSimulatorSE2` so simulations can be driven by control inputs instead of pre-defined poses.
- [ ] Move the hard-coded lidar range filter in `VictoriaParkLoader` into configuration.
- [ ] Save result as pickles dict
- [ ] Figure out if i should save the whole pose history at each iteration or just the current pose and reconstruct the history from that for visualization and analysis. When doing smoothing the current pose at one iteration may change in later iterations, so saving the whole pose history at each iteration may be more accurate for analysis and visualization. However, it may also take up more storage space. I will need to experiment with both approaches and see which one works better for my use case.
  

## Association And Landmark Management

- [ ] Add measurements function jacobian caluclation to gtsam python wrapper
- [ ] Use `sensor_offset` consistently during prediction, association, and landmark initialization to avoid duplicate landmarks from frame mismatch.
- [ ] Decide how ambiguous associations (`-2`) should be handled in `FactorGraphSLAM` instead of silently ignoring them.
- [ ] Make the ambiguous-association threshold configurable instead of hard-coded.
- [ ] Add a cleaner fallback path when no predicted landmarks survive gating.


## Numerics And Cleanup

- [ ] Replace `reorder_covariance_naive` with a safer key-aware covariance reordering utility.
- [ ] Remove unused legacy pieces in `FactorGraphSLAM`, such as the unused `motion_model`.
- [ ] Handle near-singular range/bearing Jacobians more gracefully when landmarks are very close to the robot.
- [ ] Audit bearing/range ordering across the simulator, config, and GTSAM interfaces, then lock it down with tests.

## Plotting
- [ ] NIS?
- [ ] NEES?
- [ ] Cumulative time and time per step
- [ ] Per step #local_landmarks
- [ ] Cuimulative number of local landmarks per step
- [ ] Number of JCBB associations/iteration per step
- [ ] Number of tentative landmarks per step
- [ ] Confirmed landmark metadata:
    - First seen step
    - Confirmed step
    - Seen from these poses / seen at these steps
    - local_landmark(step) # might be a bit overkill?
- [ ] 


## Validation And Documentation

- [ ] Add regression tests for JCBB/ML association, tentative landmark promotion, and `isam2` vs `batch` consistency.
- [ ] Document setup requirements more clearly, especially the `gtsam` dependency and the main entry-point scripts.
- [ ] Replace the temporary patched-GTSAM setup with prebuilt `gtsam-ttk4250` wheels once the custom wrapper API is stable.
- [ ] Decide whether assignment support on Windows means native wheels or WSL2 only.
- [ ] Refresh `src/toy_example.py` and the exploratory scripts in `src/div/` so they match the current config and data model.
- [ ] pip install gtsam when bayesTree jointMarginalCovariance feature is in stable release, and remove the manual installation instructions.
- [ ] Add better timing and performance benchmarks/profiling for the main SLAM loop, especially the association step and `isam2` updates.
