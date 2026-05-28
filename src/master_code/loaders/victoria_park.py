from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from master_code.paths import DATA_ROOT


@dataclass(slots=True)
class WheelOdometry:
    velocity: float
    steering: float
    dt: float


@dataclass(slots=True)
class VictoriaParkStep:
    odometry: list[WheelOdometry]
    scan: np.ndarray
    scan_time: float
    scan_step: int


# class VictoriaParkLoader:
#     def __init__(self, data_folder: Path | None = None):
#         if data_folder is None:
#             data_folder = DATA_ROOT / "victoria_park" / "raw"

#         dr = loadmat(str(data_folder / "aa3_dr"))
#         lsr = loadmat(str(data_folder / "aa3_lsr2"))
#         gps = loadmat(str(data_folder / "aa3_gpsx"))

#         # LiDAR: cm -> m, ms -> s
#         self.scans = lsr["LASER"] / 100.0
#         self.scan_times = lsr["TLsr"].ravel() / 1000.0

#         # Odometry: ms -> s
#         self.odo_times = dr["time"].ravel() / 1000.0
#         self.velocity = dr["speed"].ravel()
#         self.steering = dr["steering"].ravel()

#         # Optional, only for initial pose / plotting
#         self.gps_times = gps["timeGps"].ravel() / 1000.0
#         self.gps_x = gps["Lo_m"].ravel()
#         self.gps_y = gps["La_m"].ravel()

#         self.odo_dt = np.diff(self.odo_times)

#     def iterate(self, num_steps: int | None = None) -> Iterator[VictoriaParkStep]:
#         """
#         Yield one SLAM step per LiDAR scan.

#         Odometry is grouped between consecutive LiDAR scans by snapping each
#         LiDAR timestamp to the latest preceding odometry timestamp.
#         """

#         # For each scan, find the latest odometry sample before the scan.
#         scan_to_odo = np.searchsorted(self.odo_times, self.scan_times, side="right") - 1

#         # Keep scans that occur after odometry has started.
#         valid_scans = np.flatnonzero(scan_to_odo >= 0)

#         if len(valid_scans) < 2:
#             return

#         start = valid_scans[1]
#         stop = len(self.scan_times)

#         if num_steps is not None:
#             stop = min(stop, start + num_steps)

#         scan_step = 0

#         for scan_idx in range(start, stop):
#             prev_scan_idx = scan_idx - 1

#             i0 = scan_to_odo[prev_scan_idx]
#             i1 = scan_to_odo[scan_idx]

#             odometry = []
#             for i in range(i0, i1):
#                 odometry.append(
#                     WheelOdometry(
#                         velocity=float(self.velocity[i]),
#                         steering=float(self.steering[i]),
#                         dt=float(self.odo_dt[i]),
#                     )
#                 )

#             yield VictoriaParkStep(
#                 odometry=odometry,
#                 scan=self.scans[scan_idx],
#                 scan_time=float(self.scan_times[scan_idx]),
#                 scan_step=scan_step,
#             )

#             scan_step += 1
    
#     @property
#     def max_steps(self) -> int:
#         return len(self.scan_times) - 1

#     @property
#     def lidar(self) -> np.ndarray:
#         return np.column_stack([self.scan_times, self.scans])

#     @property
#     def odometry(self) -> np.ndarray:
#         return np.column_stack([self.odo_times, self.velocity, self.steering])

#     @property
#     def gnss(self) -> np.ndarray:
#         return np.column_stack([self.gps_times, self.gps_x, self.gps_y])

#     @property
#     def gnss_filtered(self) -> np.ndarray:
#         gnss = self.gnss
#         return np.delete(gnss, find_gnss_outliers(gnss), axis=0)

#     @property
#     def initial_pose(self) -> np.ndarray:
#         return np.array([self.gps_x[0], self.gps_y[0], np.deg2rad(36.0)])




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
    times = gnss[:, 0]
    positions = gnss[:, 1:3]

    dt_prev = times[1:-1] - times[:-2]
    dt_next = times[2:] - times[1:-1]

    dist_prev = np.linalg.norm(positions[1:-1] - positions[:-2], axis=1)
    dist_next = np.linalg.norm(positions[2:] - positions[1:-1], axis=1)

    too_far_from_prev = dist_prev > max_speed_m_s * dt_prev + distance_margin_m
    too_far_from_next = dist_next > max_speed_m_s * dt_next + distance_margin_m
    
    outlier_mask = ~np.isfinite(times) | ~np.all(np.isfinite(positions), axis=1)
    outlier_mask[1:-1] |= (too_far_from_prev & too_far_from_next)

    return np.flatnonzero(outlier_mask)









# """
# Dataset loader module. Handles loading from mat-files, unit conversion,
# and dataset-specific synchronization.
# """
# from collections.abc import Iterator
# from dataclasses import dataclass
# from pathlib import Path

# import gtsam
# import numpy as np
# from scipy.io import loadmat

# from master_code.measurements import RelativePoseMeasurement, SlamStepInput
# from master_code.paths import DATA_ROOT



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
    def max_steps(self) -> int:
        return len(self.lsr_timestamps) - 1

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


# GNSS_MAX_SPEED_M_S = 1
# GNSS_OUTLIER_MARGIN_M = 1.0

# # Helper function to identify large outliers in GNSS data based on speed and distance criteria.
# def find_gnss_outliers(
#     gnss: np.ndarray,
#     max_speed_m_s: float = GNSS_MAX_SPEED_M_S,
#     distance_margin_m: float = GNSS_OUTLIER_MARGIN_M,
# ) -> np.ndarray:
#     """
#     Return indices of isolated GNSS samples that violate a speed bound.

#     A sample is marked as an outlier when it is too far from both immediate
#     neighbours to be reachable at ``max_speed_m_s``, while those neighbours are
#     mutually plausible over their combined time interval. The margin absorbs
#     normal GNSS noise so short sampling intervals do not become too strict.
#     """
#     gnss = np.asarray(gnss, dtype=float)
#     if gnss.ndim != 2 or gnss.shape[1] < 3:
#         raise ValueError("gnss must be an array with columns [timestamp, x, y].")

#     if len(gnss) < 3:
#         return np.array([], dtype=int)

#     times = gnss[:, 0]
#     positions = gnss[:, 1:3]

#     outlier_mask = ~np.isfinite(times) | ~np.all(np.isfinite(positions), axis=1)

#     dt_prev = times[1:-1] - times[:-2]
#     dt_next = times[2:] - times[1:-1]

#     dist_prev = np.linalg.norm(positions[1:-1] - positions[:-2], axis=1)
#     dist_next = np.linalg.norm(positions[2:] - positions[1:-1], axis=1)

#     valid_triplet = (
#         np.isfinite(dt_prev)
#         & np.isfinite(dt_next)
#         & np.isfinite(dist_prev)
#         & np.isfinite(dist_next)
#         & (dt_prev > 0.0)
#         & (dt_next > 0.0)
#     )

#     too_far_from_prev = dist_prev > max_speed_m_s * dt_prev + distance_margin_m
#     too_far_from_next = dist_next > max_speed_m_s * dt_next + distance_margin_m

#     outlier_mask[1:-1] |= (
#         valid_triplet
#         & too_far_from_prev
#         & too_far_from_next
#     )

#     return np.flatnonzero(outlier_mask)


    
    
   
