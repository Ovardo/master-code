from collections.abc import Iterator

import numpy as np
from scipy.io import loadmat

from master_code.paths import DATA_ROOT


class SimulatedDataLoader:
    """Loader for the simulated SLAM dataset with processed measurements."""

    def __init__(self):
        data_file = DATA_ROOT / "simulatedSLAM.mat"
        raw_data = loadmat(str(data_file))
        
        self.measurements = [zk.T for zk in raw_data["z"].ravel()]
        self.landmarks_gt = np.asarray(raw_data["landmarks"].T, dtype=float)
        self.odometry = np.asarray(raw_data["odometry"].T, dtype=float)
        self.poses_gt = np.asarray(raw_data["poseGT"].T, dtype=float)

    @property
    def max_steps(self) -> int:
        return len(self.odometry)

    @property
    def initial_pose(self) -> np.ndarray:
        return self.poses_gt[0]


    def iterate(self, max_steps: int | None = None) -> Iterator[dict]:

        stop = len(self.odometry) 
        if max_steps is not None:
            stop = min(stop, max_steps)

        for scan_step in range(stop-1):
            dx, dy, dtheta = self.odometry[scan_step]
            measurement_index = scan_step + 1
            measurements = self.measurements[measurement_index]

            yield {
                'relative_pose': np.array([dx, dy, dtheta], dtype=float),
                'measurements': measurements,
                'scan_step': scan_step,
                'scan_time': float(measurement_index),
            }