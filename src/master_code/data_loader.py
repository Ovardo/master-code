"""
Dataset loader module. Handles loading from mat-files, unit conversion,
and dataset-specific synchronization.
"""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import gtsam
import numpy as np
from scipy.io import loadmat

from master_code.measurements import RelativePoseMeasurement, SlamStepInput
from master_code.paths import DATA_ROOT



@dataclass(slots=True)
class WheelOdometry:
    velocity: float
    steering: float
    dt: float


@dataclass(slots=True)
class RawLidarStepInput:
    odometry: list[WheelOdometry]
    scan: np.ndarray
    scan_time: float
    scan_step: int


LidarStepInput = RawLidarStepInput


class VictoriaParkLoader:
    def __init__(self, data_folder: Path | None = None):
        if data_folder is None:
            data_folder = DATA_ROOT / "victoria_park" / "raw"

        self._load_data(data_folder)
        self._prepare_odometry_intervals()

    def _load_data(self, data_folder: Path):
        raw_data = {
            **loadmat(str(data_folder / "aa3_dr")),
            **loadmat(str(data_folder / "aa3_lsr2")),
            **loadmat(str(data_folder / "aa3_gpsx")),
        }

        # Laser scans: cm -> m, ms -> s
        self.lsr_scans = raw_data["LASER"] / 100.0
        self.lsr_timestamps = raw_data["TLsr"].ravel() / 1000.0

        # Wheel odometry: ms -> s
        self.odo_steering = raw_data["steering"].ravel()
        self.odo_velocity = raw_data["speed"].ravel()
        self.odo_timestamps = raw_data["time"].ravel() / 1000.0

        # gnss: ms -> s
        self.gnss_latitude = raw_data["La_m"].ravel()
        self.gnss_longitude = raw_data["Lo_m"].ravel()
        self.gnss_timestamps = raw_data["timeGps"].ravel() / 1000.0

    def _prepare_odometry_intervals(self):
        """
        Forward odometry convention:

            u_k = (velocity[k], steering[k])

        describes motion over:

            [odo_timestamps[k], odo_timestamps[k + 1]]
        """
        self.odo_interval_t0 = self.odo_timestamps[:-1]
        self.odo_interval_t1 = self.odo_timestamps[1:]

        # Change below to [1:] for backward odometry convention
        self.odo_interval_velocity = self.odo_velocity[:-1]
        self.odo_interval_steering = self.odo_steering[:-1]

        if np.any(self.odo_interval_t1 <= self.odo_interval_t0):
            raise ValueError("Odometry timestamps must be strictly increasing.")

    def _odometry_between(self, t0: float, t1: float) -> list[WheelOdometry]:
        """
        Return odometry inputs clipped to [t0, t1].

        Each returned input represents motion over a clipped interval:

            [start_time, end_time]

        with duration:

            dt = end_time - start_time

        using the forward convention:

            u_k applies over [T_odo[k], T_odo[k+1]]
        """
        if t1 <= t0:
            return []

        # Find odometry intervals that overlap [t0, t1].
        first = np.searchsorted(self.odo_interval_t1, t0, side="right")
        last = np.searchsorted(self.odo_interval_t0, t1, side="left")

        if first >= last:
            return []

        # Clip intervals to [t0, t1] and compute dt for each interval.
        starts = np.maximum(self.odo_interval_t0[first:last], t0)
        ends = np.minimum(self.odo_interval_t1[first:last], t1)
        dts = ends - starts

        return [
            WheelOdometry(
                velocity=float(v),
                steering=float(a),
                dt=float(dt),
            )
            for v, a, dt in zip(
                self.odo_interval_velocity[first:last],
                self.odo_interval_steering[first:last],
                dts,
            )
        ]

    def iterate(self, max_steps: int | None = None) -> Iterator[RawLidarStepInput]:
        """
        Iterate over LiDAR-to-LiDAR steps.
        """
        start_lidar_idx = 2 # as first lidar measurement (idx 0) is before odometry starts
        stop_lidar_idx = self.lsr_timestamps.size
        scan_step = 0

        if max_steps is not None:
            stop_lidar_idx = min(stop_lidar_idx, start_lidar_idx + max_steps)

        for lidar_idx in range(start_lidar_idx, stop_lidar_idx):
            t0 = float(self.lsr_timestamps[lidar_idx - 1])
            t1 = float(self.lsr_timestamps[lidar_idx])

            odometry = self._odometry_between(t0, t1)

            yield RawLidarStepInput(
                odometry=odometry,
                scan=self.lsr_scans[lidar_idx],
                scan_time=self.lsr_timestamps[lidar_idx],
                scan_step=scan_step,
            )
            scan_step += 1

    def iterate_synced(self, max_steps: int | None = None) -> Iterator[RawLidarStepInput]:
        """
        Iterate over LiDAR steps snapped to the preceding odometry sample.

        This simpler synchronization uses complete odometry intervals only:
        every LiDAR timestamp is matched to the latest odometry timestamp before
        it, and the returned odometry spans between two consecutive matched
        odometry timestamps. No partial odometry intervals are clipped.
        """
        preceding_odo_idx = np.searchsorted(
            self.odo_timestamps,
            self.lsr_timestamps,
            side="right",
        ) - 1

        valid_lidar_indices = np.flatnonzero(preceding_odo_idx >= 0)
        if valid_lidar_indices.size < 2:
            return

        first_valid_lidar_idx = int(valid_lidar_indices[0])
        start_lidar_idx = first_valid_lidar_idx + 1
        stop_lidar_idx = self.lsr_timestamps.size
        scan_step = 0

        if max_steps is not None:
            stop_lidar_idx = min(stop_lidar_idx, start_lidar_idx + max_steps)

        for lidar_idx in range(start_lidar_idx, stop_lidar_idx):
            first_odo_idx = preceding_odo_idx[lidar_idx - 1]
            last_odo_idx = preceding_odo_idx[lidar_idx]

            odometry = [
                WheelOdometry(
                    velocity=float(v),
                    steering=float(a),
                    dt=float(t1 - t0),
                )
                for v, a, t0, t1 in zip(
                    self.odo_interval_velocity[first_odo_idx:last_odo_idx],
                    self.odo_interval_steering[first_odo_idx:last_odo_idx],
                    self.odo_interval_t0[first_odo_idx:last_odo_idx],
                    self.odo_interval_t1[first_odo_idx:last_odo_idx],
                )
            ]

            yield RawLidarStepInput(
                odometry=odometry,
                scan=self.lsr_scans[lidar_idx],
                scan_time=self.lsr_timestamps[lidar_idx],
                scan_step=scan_step,
            )
            scan_step += 1

    @property
    def lidar(self) -> np.ndarray:
        return np.column_stack(
            [self.lsr_timestamps, self.lsr_scans]
        )

    @property
    def odometry(self) -> np.ndarray:
        return np.column_stack(
            [self.odo_timestamps, self.odo_velocity, self.odo_steering]
        )

    @property
    def gnss(self) -> np.ndarray:
        return np.column_stack(
            [self.gnss_timestamps, self.gnss_longitude, self.gnss_latitude]
        )

    @property
    def gnss_filtered(self) -> np.ndarray:
        gnss = self.gnss
        return np.delete(gnss, find_gnss_outliers(gnss), axis=0)

    @property 
    def initial_pose(self) -> np.ndarray:
        return np.array(
            [self.gnss_longitude[0], self.gnss_latitude[0], np.deg2rad(36)]
        ) 


class SimulatedDataLoader:
    """Loader for the simulated SLAM dataset with processed measurements."""

    def __init__(self, data_file: Path | None = None):
        if data_file is None:
            data_file = DATA_ROOT / "simulatedSLAM.mat"

        raw_data = loadmat(str(data_file))
        self.measurements = [
            np.asarray(zk.T, dtype=float).reshape(-1, 2)
            for zk in raw_data["z"].ravel()
        ]
        self.landmarks_gt = np.asarray(raw_data["landmarks"].T, dtype=float)
        self.odometry = np.asarray(raw_data["odometry"].T, dtype=float)
        self.poses_gt = np.asarray(raw_data["poseGT"].T, dtype=float)

        if self.odometry.shape[1] != 3:
            raise ValueError("Simulated odometry must have columns [dx, dy, dtheta].")
        if len(self.measurements) != len(self.odometry):
            raise ValueError("Simulated z and odometry must have the same length.")
        if len(self.poses_gt) != len(self.odometry) + 1:
            raise ValueError("Simulated poseGT must contain one more pose than odometry.")

    @property
    def initial_pose(self) -> np.ndarray:
        return self.poses_gt[0]

    @property
    def reference(self) -> dict[str, np.ndarray]:
        return {
            "poses_gt": self.poses_gt,
            "landmarks_gt": self.landmarks_gt,
        }

    def iterate(
        self,
        max_steps: int | None = None,
        *,
        odometry_covariance: np.ndarray,
        max_range: float | None = None,
    ) -> Iterator[SlamStepInput]:
        # measurements[i] belongs to poses_gt[i]. Since FactorGraphSLAM.update()
        # first applies odometry[i] to create poses_gt[i + 1], then registers
        # measurements, odometry[i] must be paired with measurements[i + 1].
        stop = min(len(self.odometry), len(self.measurements) - 1)
        if max_steps is not None:
            stop = min(stop, max_steps)

        covariance = np.asarray(odometry_covariance, dtype=float).reshape(3, 3)

        for scan_step in range(stop):
            dx, dy, dtheta = self.odometry[scan_step]
            measurement_index = scan_step + 1
            measurements = self.measurements[measurement_index]

            if max_range is not None:
                measurements = measurements[measurements[:, 0] < max_range]

            yield SlamStepInput(
                odometry=RelativePoseMeasurement(
                    pose=gtsam.Pose2(float(dx), float(dy), float(dtheta)),
                    covariance=covariance.copy(),
                ),
                measurements=measurements,
                scan_step=scan_step,
                scan_time=float(measurement_index),
            )





GNSS_MAX_SPEED_M_S = 1
GNSS_OUTLIER_MARGIN_M = 1.0

# Helper function to identify large outliers in GNSS data based on speed and distance criteria.
def find_gnss_outliers(
    gnss: np.ndarray,
    max_speed_m_s: float = GNSS_MAX_SPEED_M_S,
    distance_margin_m: float = GNSS_OUTLIER_MARGIN_M,
) -> np.ndarray:
    """
    Return indices of isolated GNSS samples that violate a speed bound.

    A sample is marked as an outlier when it is too far from both immediate
    neighbours to be reachable at ``max_speed_m_s``, while those neighbours are
    mutually plausible over their combined time interval. The margin absorbs
    normal GNSS noise so short sampling intervals do not become too strict.
    """
    gnss = np.asarray(gnss, dtype=float)
    if gnss.ndim != 2 or gnss.shape[1] < 3:
        raise ValueError("gnss must be an array with columns [timestamp, x, y].")

    if len(gnss) < 3:
        return np.array([], dtype=int)

    times = gnss[:, 0]
    positions = gnss[:, 1:3]

    outlier_mask = ~np.isfinite(times) | ~np.all(np.isfinite(positions), axis=1)

    dt_prev = times[1:-1] - times[:-2]
    dt_next = times[2:] - times[1:-1]

    dist_prev = np.linalg.norm(positions[1:-1] - positions[:-2], axis=1)
    dist_next = np.linalg.norm(positions[2:] - positions[1:-1], axis=1)

    valid_triplet = (
        np.isfinite(dt_prev)
        & np.isfinite(dt_next)
        & np.isfinite(dist_prev)
        & np.isfinite(dist_next)
        & (dt_prev > 0.0)
        & (dt_next > 0.0)
    )

    too_far_from_prev = dist_prev > max_speed_m_s * dt_prev + distance_margin_m
    too_far_from_next = dist_next > max_speed_m_s * dt_next + distance_margin_m

    outlier_mask[1:-1] |= (
        valid_triplet
        & too_far_from_prev
        & too_far_from_next
    )

    return np.flatnonzero(outlier_mask)


    
    
   
