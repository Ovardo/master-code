from collections.abc import Iterator

import gtsam
import numpy as np
from scipy.io import loadmat

from master_code.config import SlamConfig
from master_code.paths import DATA_ROOT
from master_code.slam import SlamStepInput


class SimulatedDataLoader:
    """Loader for the simulated SLAM dataset with processed measurements."""

    name = "sim"

    def __init__(self):
        data_file = DATA_ROOT / "simulated" / "simulatedSLAM.mat"
        raw_data = loadmat(str(data_file))
        
        self.measurements = [zk.T for zk in raw_data["z"].ravel()]
        self.landmarks_gt = np.asarray(raw_data["landmarks"].T, dtype=float)
        self.odometry = np.asarray(raw_data["odometry"].T, dtype=float)
        self.poses_gt = np.asarray(raw_data["poseGT"].T, dtype=float)

    @property
    def max_steps(self) -> int:
        return len(self.odometry) - 1

    @property
    def initial_pose(self) -> np.ndarray:
        return self.poses_gt[0]

    def iterate_slam(
        self,
        config: SlamConfig,
        max_steps: int | None = None,
    ) -> Iterator[SlamStepInput]:
        for meas in self.iterate(max_steps):
            yield SlamStepInput(
                relative_pose=gtsam.Pose2(*meas["relative_pose"]),
                relative_pose_cov=np.asarray(config.noise.odom_cov_matrix, dtype=float),
                measurements=np.asarray(meas["measurements"], dtype=float).reshape(-1, 2),
                scan_time=float(meas["scan_time"]),
            )

    def iterate(self, max_steps: int | None = None) -> Iterator[dict]:
        if max_steps is not None:
            max_steps = min(self.max_steps, max_steps)
        else:
            max_steps = self.max_steps

        for scan_step in range(max_steps):
            dx, dy, dtheta = self.odometry[scan_step]
            measurement_index = scan_step + 1
            measurements = self.measurements[measurement_index]

            yield {
                'relative_pose': np.array([dx, dy, dtheta], dtype=float),
                'measurements': measurements,
                'scan_step': scan_step,
                'scan_time': float(measurement_index),
            }
