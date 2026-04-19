"""
Victoria Park dataset loader module.
Handles loading and preprocessing of Victoria Park SLAM dataset.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

@dataclass(slots=True)
class OdometryInput:
    timestamp: float
    ve: float
    alpha: float
    dt: float

@dataclass(slots=True)
class LidarStepInput:
    step_index: int
    timestamp: float
    odometry: list[OdometryInput]
    z_lsr: np.ndarray


class VictoriaParkLoader:
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
        last_time = self.T_odo[0]

        n_lidar = self.K_lsr - first_lidar_idx
        if max_steps is not None:
            n_lidar = min(n_lidar, max_steps)

        k_odo = 1

        for i in range(n_lidar):
            k_lsr = first_lidar_idx + i
            t_scan = self.T_lsr[k_lsr]

            odometry = []

            while k_odo < self.K_odo and self.T_odo[k_odo] <= t_scan:
                dt = self.T_odo[k_odo] - last_time
                if dt < 0:
                    raise ValueError(f"Negative dt at odometry index {k_odo}")

                if dt > 0:
                    odometry.append(
                        OdometryInput(
                            timestamp=self.T_odo[k_odo],
                            ve=self.Ve_odo[k_odo],
                            alpha=self.Alpha_odo[k_odo],
                            dt=dt,
                        )
                    )
                    last_time = self.T_odo[k_odo]

                k_odo += 1

            # handle partial interval if scan falls between odometry timestamps
            if last_time < t_scan and k_odo < self.K_odo:
                dt = t_scan - last_time
                if dt < 0:
                    raise ValueError(f"Negative partial dt before lidar index {k_lsr}")

                odometry.append(
                    OdometryInput(
                        timestamp=t_scan,
                        ve=self.Ve_odo[k_odo-1],
                        alpha=self.Alpha_odo[k_odo-1],
                        dt=dt,
                    )
                )
                last_time = t_scan

            yield LidarStepInput(
                step_index=i,
                timestamp=t_scan,
                odometry=odometry,
                z_lsr=self.Z_lsr[k_lsr],
            )


    @property
    def lidar(self) -> np.ndarray:
        """Return raw (unsynced) lidar measurements as (K_lsr, 361+1) array where first column is timestamps and rest are ranges."""
        return np.column_stack([self.T_lsr, self.Z_lsr])
    
    @property
    def odometry(self) -> np.ndarray:
        """Return raw (unsynced) dead reckoning measurement as (K_odo, 3) array of [ve, alpha, t_odo]."""
        return np.column_stack([self.Ve_odo, self.Alpha_odo, self.T_odo])

    @property
    def gps(self) -> np.ndarray:
        """Return raw (unsynced) GPS data as (K_gps, 3) array of [longitud, latitude, t_gps]."""
        return np.column_stack([self.Lo_gps, self.La_gps, self.T_gps])
    
    @property
    def initial_pose(self) -> np.ndarray:
        """Return initial position (x, y, theta) from GPS data in ENU frame."""
        return np.array([self.Lo_gps[1], self.La_gps[1], np.deg2rad(36)])
    
    
   

