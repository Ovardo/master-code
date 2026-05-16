# """
# Victoria Park dataset loader module.
# Handles synching of Lidar and odometry data
# """


from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(slots=True)
class WheelOdometry:
    velocity: float
    steering: float
    dt: float


@dataclass(slots=True)
class LidarStepInput:
    odometry: list[WheelOdometry]
    scan: np.ndarray



class VictoriaParkLoader1:
    """
    Iterator-based loader for Victoria Park SLAM dataset.  
    Handles timing synchronization between dead reckoning and laser measurements,
    """
    def __init__(self, data_folder: Path | None = None):
        if data_folder is None:
            data_folder = Path(__file__).parents[1].joinpath("data/victoria_park/matlab")
        self._load_data(data_folder)
        
    def _load_data(self, data_folder: Path):
        """Load all raw data from .mat files."""
        raw_data = {
            **loadmat(str(data_folder.joinpath("aa3_dr"))),
            **loadmat(str(data_folder.joinpath("aa3_lsr2"))),
            **loadmat(str(data_folder.joinpath("aa3_gpsx"))),
        }
        
        # Laser data (lsr)
        self.Z_lsr = raw_data["LASER"] / 100 # (K_lsr, 361) cm -> m 
        self.T_lsr = raw_data["TLsr"].ravel() / 1000 # (K_lsr,) ms -> s 
        
        # Odometry data (odo)
        self.Alpha_odo = raw_data["steering"].ravel() # (K_odo,)
        self.Ve_odo = raw_data["speed"].ravel() # (K_odo,)
        self.T_odo = raw_data["time"].ravel() / 1000 # (K_odo,) ms -> s 
        
        # GPS data (gps)
        self.La_gps = raw_data["La_m"].ravel() # (K_gps,)
        self.Lo_gps = raw_data["Lo_m"].ravel() # (K_gps,)
        self.T_gps = raw_data["timeGps"].ravel() / 1000 #  (K_gps,) ms -> s 
        
        # Data sizes
        self.K_odo = self.T_odo.size
        self.K_lsr = self.T_lsr.size
        self.K_gps = self.T_gps.size
    

    def iter_lidar_steps(self, max_steps: int | None = None):
        first_lidar_idx = 1  # idx 0 is a bit off in timing, so start from 1

        n_lidar = self.K_lsr - first_lidar_idx
        if max_steps is not None:
            n_lidar = min(n_lidar, max_steps)
        if n_lidar <= 0:
            return

        k_lsr = first_lidar_idx
        k_odo = 1  # odometry sample at k_odo describes interval [k_odo-1, k_odo]

        # Start integration from the first odometry timestamp; carry this forward
        # even when lidar cuts an odometry interval in two pieces.
        interval_start_time = self.T_odo[0]
        odometry: list[WheelOdometry] = []

        for i in range(n_lidar):
            t_scan = self.T_lsr[k_lsr]

            # Consume complete or partial odometry intervals until we reach t_scan.
            while k_odo < self.K_odo and interval_start_time < t_scan:
                t_odo_k = self.T_odo[k_odo]

                if t_odo_k <= interval_start_time:
                    raise ValueError(f"Non-increasing odometry timestamp at index {k_odo}")

                segment_end = min(t_odo_k, t_scan)
                dt = segment_end - interval_start_time
                if dt < 0:
                    raise ValueError(
                        f"Negative dt while syncing lidar index {k_lsr} and odometry index {k_odo}"
                    )

                if dt > 0:
                    odometry.append(
                        WheelOdometry(
                            velocity=self.Ve_odo[k_odo],
                            steering=self.Alpha_odo[k_odo],
                            dt=dt,
                        )
                    )

                interval_start_time = segment_end
                if np.isclose(interval_start_time, t_odo_k) or interval_start_time >= t_odo_k:
                    k_odo += 1

            yield LidarStepInput(
                odometry=odometry,
                scan=self.Z_lsr[k_lsr],
            )

            # Start collecting odometry for the next lidar step.
            odometry = []
            k_lsr += 1


    @property
    def lidar(self) -> np.ndarray:
        """Return raw (unsynced) lidar measurements as (K_lsr, 361+1) array where first column is timestamps and rest are ranges."""
        return np.column_stack([self.T_lsr, self.Z_lsr])
    
    @property
    def odometry(self) -> np.ndarray:
        """Return raw (unsynced) dead reckoning measurement as (K_odo, 3) array of [t_odo, ve, alpha]."""
        return np.column_stack([self.T_odo, self.Ve_odo, self.Alpha_odo])

    @property
    def gps(self) -> np.ndarray:
        """Return raw (unsynced) GPS data as (K_gps, 3) array of [t_gps, longitude, latitude]."""
        return np.column_stack([self.T_gps, self.Lo_gps, self.La_gps])
    
    @property
    def initial_pose(self) -> np.ndarray:
        """Return initial position (x, y, theta) from GPS data in ENU frame."""
        return np.array([self.Lo_gps[1], self.La_gps[1], np.deg2rad(36)]) 
    

# class VictoriaParkLoader2:
#     def __init__(self, data_folder: Path | None = None):
#         if data_folder is None:
#             data_folder = Path(__file__).parents[1] / "data/victoria_park/matlab"

#         self._load_data(data_folder)
#         self._prepare_odometry_intervals()

#     def _load_data(self, data_folder: Path):
#         raw_data = {
#             **loadmat(str(data_folder / "aa3_dr")),
#             **loadmat(str(data_folder / "aa3_lsr2")),
#             **loadmat(str(data_folder / "aa3_gpsx")),
#         }

#         # Laser scans: cm -> m, ms -> s
#         self.lsr_scans = raw_data["LASER"] / 100.0
#         self.lsr_timestamps = raw_data["TLsr"].ravel() / 1000.0

#         # Wheel odometry: ms -> s
#         self.odo_steering = raw_data["steering"].ravel()
#         self.odo_velocity = raw_data["speed"].ravel()
#         self.odo_timestamps = raw_data["time"].ravel() / 1000.0

#         # GPS: ms -> s
#         self.gps_latitude = raw_data["La_m"].ravel()
#         self.gps_longitude = raw_data["Lo_m"].ravel()
#         self.gps_timestamps = raw_data["timeGps"].ravel() / 1000.0

#     def _prepare_odometry_intervals(self):
#         """
#         Odometry sample k describes the interval [T_odo[k], T_odo[k+1]].
#         Therefore the usable interval arrays have length K_odo - 1.
#         """
#         self.odo_t0 = self.odo_timestamps[:-1]
#         self.odo_t1 = self.odo_timestamps[1:]

#         if np.any(self.odo_t1 <= self.odo_t0):
#             raise ValueError("Odometry timestamps must be strictly increasing.")

#     def _odometry_between(self, t0: float, t1: float) -> list[WheelOdometry]:
#         """
#         Return odometry inputs clipped to [t0, t1].

#         Each returned input represents motion over:

#             [start_time, end_time]

#         using the forward convention:

#             u_k applies over [T_odo[k], T_odo[k+1]]
#         """
#         if t1 <= t0:
#             return []

#         # Find odometry intervals overlapping [t0, t1].
#         first = np.searchsorted(self.odo_t1, t0, side="right")
#         last = np.searchsorted(self.odo_t0, t1, side="left")

#         if first >= last:
#             return []

#         # Clip intervals to [t0, t1] and compute dt for each interval.
#         starts = np.maximum(self.odo_t0[first:last], t0)
#         ends = np.minimum(self.odo_t1[first:last], t1)
#         dts = ends - starts

#         valid = dts > 0.0

#         return [
#             WheelOdometry(
#                 velocity=float(v),
#                 steering=float(a),
#                 dt=float(dt),
#             )
#             for v, a, dt in zip(
#                 self.odo_velocity[first:last][valid],
#                 self.odo_steering[first:last][valid],
#                 dts[valid],
#             )
#     ]    

#     def iter_lidar_steps(self, max_steps: int | None = None):
#         """
#         Iterate over LiDAR-to-LiDAR steps.

#         The first graph pose corresponds to:

#             lsr_scans[first_lidar_index]

#         This iterator starts from the next scan and yields odometry from:

#             lidar[k - 1] -> lidar[k]
#         """

#         # Start from second scan (first scan is a bit off in timing, so skip it)
#         start_lidar_idx = 1  
#         yield LidarStepInput(
#             odometry=[],
#             scan=self.lsr_scans[start_lidar_idx],
#         )
        
#         start_lidar_idx += 1
#         stop_lidar_idx = self.lsr_timestamps.size

#         if max_steps is not None:
#             stop_lidar_idx = min(stop_lidar_idx, start_lidar_idx + max_steps)

#         for lidar_idx in range(start_lidar_idx, stop_lidar_idx):
#             t0 = float(self.lsr_timestamps[lidar_idx - 1])
#             t1 = float(self.lsr_timestamps[lidar_idx])

#             odometry = self._odometry_between(t0, t1)

#             yield LidarStepInput(
#                 odometry=odometry,
#                 scan=self.lsr_scans[lidar_idx]
#             )


#     @property
#     def lidar(self) -> np.ndarray:
#         return np.column_stack([self.lsr_timestamps, self.lsr_scans])

#     @property
#     def odometry(self) -> np.ndarray:
#         return np.column_stack([self.odo_timestamps, self.odo_velocity, self.odo_steering])

#     @property
#     def gps(self) -> np.ndarray:
#         return np.column_stack([self.gps_timestamps, self.gps_longitude, self.gps_latitude])

#     @property
#     def initial_pose(self) -> np.ndarray:
#         n_gps_avg = 5 
#         # We know the car is stationary for the first 2 seconds
#         x0 = np.mean(self.gps_longitude[:n_gps_avg])
#         y0 = np.mean(self.gps_latitude[:n_gps_avg])
#         theta0 = np.deg2rad(36.0)

#         return np.array([x0, y0, theta0])


class VictoriaParkLoader:
    def __init__(self, data_folder: Path | None = None):
        if data_folder is None:
            data_folder = Path(__file__).parents[1] / "data/victoria_park/matlab"

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

        # GPS: ms -> s
        self.gps_latitude = raw_data["La_m"].ravel()
        self.gps_longitude = raw_data["Lo_m"].ravel()
        self.gps_timestamps = raw_data["timeGps"].ravel() / 1000.0

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
        self.odo_interval_velocity = self.odo_velocity[1:]
        self.odo_interval_steering = self.odo_steering[1:]

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

            u_k applies over [T_odo[k], T_odo[k + 1]]
        """
        if t1 <= t0:
            return []

        # Find odometry intervals that overlap [t0, t1].
        #
        # An interval overlaps if:
        #
        #     odo_interval_t1 > t0
        #     odo_interval_t0 < t1
        #
        first = np.searchsorted(self.odo_interval_t1, t0, side="right")
        last = np.searchsorted(self.odo_interval_t0, t1, side="left")

        if first >= last:
            return []

        starts = np.maximum(self.odo_interval_t0[first:last], t0)
        ends = np.minimum(self.odo_interval_t1[first:last], t1)
        dts = ends - starts

        valid = dts > 0.0

        return [
            WheelOdometry(
                velocity=float(v),
                steering=float(a),
                dt=float(dt),
            )
            for v, a, dt in zip(
                self.odo_interval_velocity[first:last][valid],
                self.odo_interval_steering[first:last][valid],
                dts[valid],
            )
        ]

    def iter_lidar_steps(self, max_steps: int | None = None):
        """
        Iterate over LiDAR-to-LiDAR steps.

        The first graph pose corresponds to:

            lsr_scans[first_lidar_index]

        This iterator starts from the next scan and yields odometry from:

            lidar[k - 1] -> lidar[k]
        """
        start_lidar_idx = 2 # as first lidar measurement (idx 0) is before odometry starts
        stop_lidar_idx = self.lsr_timestamps.size

        if max_steps is not None:
            stop_lidar_idx = min(stop_lidar_idx, start_lidar_idx + max_steps)

        for lidar_idx in range(start_lidar_idx, stop_lidar_idx):
            t0 = float(self.lsr_timestamps[lidar_idx - 1])
            t1 = float(self.lsr_timestamps[lidar_idx])

            odometry = self._odometry_between(t0, t1)

            yield LidarStepInput(
                odometry=odometry,
                scan=self.lsr_scans[lidar_idx],
            )

    @property
    def lidar(self) -> np.ndarray:
        """
        Raw LiDAR data as:

            [timestamp, range_0, range_1, ..., range_360]
        """
        return np.column_stack([self.lsr_timestamps, self.lsr_scans])

    @property
    def odometry(self) -> np.ndarray:
        """
        Raw odometry data as:

            [timestamp, velocity, steering]
        """
        return np.column_stack(
            [
                self.odo_timestamps,
                self.odo_velocity,
                self.odo_steering,
            ]
        )

    @property
    def gps(self) -> np.ndarray:
        """
        Raw GPS data as:

            [timestamp, longitude, latitude]
        """
        return np.column_stack(
            [
                self.gps_timestamps,
                self.gps_longitude,
                self.gps_latitude,
            ]
        )

    @property
    def initial_pose(self) -> np.ndarray:
        """
        Prior pose for the first SLAM pose.

        Since the car is stationary during the first seconds, use the average
        GPS position during this stationary period.

        The yaw is hardcoded from prior knowledge.
        """
        stationary_duration = 2.0

        stationary_gps = np.where(
            self.gps_timestamps <= self.gps_timestamps[0] + stationary_duration
        )[0]

        if stationary_gps.size == 0:
            stationary_gps = np.arange(min(5, self.gps_timestamps.size))

        x0 = np.mean(self.gps_longitude[stationary_gps])
        y0 = np.mean(self.gps_latitude[stationary_gps])
        theta0 = np.deg2rad(36.0)

        return np.array([x0, y0, theta0])




    
    
   
