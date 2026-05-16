"""
Victoria Park dataset loader module. Handles loading from mat-files, 
unit conversion and synching of Lidar and odometry data
"""

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


class VictoriaParkLoader:
    def __init__(self, data_folder: Path | None = None):
        if data_folder is None:
            data_folder = Path(__file__).parents[1] / "data/victoria_park/raw"

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

    def iterate(self, max_steps: int | None = None) -> LidarStepInput:
        """
        Iterate over LiDAR-to-LiDAR steps.
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
        Initial pose in ENU frame, derived from first GPS measurement.
        
            [longitude, latitude, yaw]
        """
        return np.array(
            [
                self.gps_longitude[0], 
                self.gps_latitude[0], 
                np.deg2rad(36)
            ]
        ) 



    
    
   
