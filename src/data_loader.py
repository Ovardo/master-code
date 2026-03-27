"""
Victoria Park dataset loader module.
Handles loading and preprocessing of Victoria Park SLAM dataset.
"""

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.io import loadmat

from slam_types import SLAMStepInput
from utils.utils_math import ssa
from utils.utils_victoria_park import detectTrees, odom_increment_and_jac_from_ve_alpha


class VictoriaParkLoader:
    """
    Iterator-based loader for Victoria Park SLAM dataset.
    
    Handles all timing synchronization between dead reckoning and laser measurements,
    and provides clean step-by-step data access.
    """
    
    def __init__(self, data_folder: Path):
        """
        Initialize the Victoria Park data loader.
        
        Parameters
        ----------
        data_folder : Path, optional
            Path to the Victoria Park data folder
        """        
        # Load raw data
        self._load_data(data_folder)
        
        # Initialize iteration state
        self._reset_state()
    

    def _load_data(self, data_folder: Path):
        """Load all data from .mat files."""
        if data_folder is None:
            data_folder = Path(__file__).parents[1].joinpath("data/victoria_park")
        
        # Load .mat files
        realSLAM_ws = {
            **loadmat(str(data_folder.joinpath("aa3_dr"))),
            **loadmat(str(data_folder.joinpath("aa3_lsr2"))),
            **loadmat(str(data_folder.joinpath("aa3_gpsx"))),
        }
        
        # Laser data (lsr)
        self.Z_lsr = realSLAM_ws["LASER"] / 100 # (K_lsr, 361) convert from cm to m 
        self.T_lsr = (realSLAM_ws["TLsr"] / 1000).ravel() # (K_lsr,) convert ms to s 
        
        # Dead reckoning data (dr)
        self.Alpha_dr = realSLAM_ws["steering"].ravel() # (K_dr,)
        self.Ve_dr = realSLAM_ws["speed"].ravel() # (K_dr,)
        self.T_dr = (realSLAM_ws["time"] / 1000).ravel() # (K_dr,) convert ms to s 
        
        # GPS data (gps)
        self.La_gps = realSLAM_ws["La_m"].ravel() # (K_gps,)
        self.Lo_gps = realSLAM_ws["Lo_m"].ravel() # (K_gps,)
        self.T_gps = (realSLAM_ws["timeGps"] / 1000).ravel() #  (K_gps,) convert ms to s 
        
        # Data sizes
        self.K_dr = self.T_dr.size
        self.K_lsr = self.T_lsr.size
        self.K_gps = self.T_gps.size
    
    def _reset_state(self):
        """Reset iteration state to beginning."""
        self.k_lsr = 1  # First laser measurement (0 seems to be off in timing)
        self.t = self.T_dr[0] # time of last processed step (start with first dead reckoning timestamp)
    

    def get_step(self, k_dr: int) -> SLAMStepInput:
        """
        Get processed data for dead step k.
        
        Parameters
        ----------
        k_dr : int
            Odometry step index (1-based)
        
        Returns
        -------
        SLAMStepInput
            Processed step data including odometry and any measurements
        """
        if k_dr >= self.K_dr- 1:
            raise StopIteration(f"Reached end of data at step {k_dr}")
        
        # Check if we have a laser measurement at this step
        has_laser = (self.k_lsr < self.K_lsr - 1 and 
                     self.T_lsr[self.k_lsr] <= self.T_dr[k_dr + 1])
        
        if has_laser:
            # Compute odometry up to laser measurement time
            dt_dr = self.T_lsr[self.k_lsr] - self.t
            if dt_dr < 0:
                raise ValueError(f"Negative time increment at step {k_dr}")
            
            self.t = self.T_lsr[self.k_lsr]  
            ve_dr = self.Ve_dr[k_dr + 1] # we do not really have this value yet in 
            alpha_dr = self.Alpha_dr[k_dr + 1] #

            # odo = odometry_func(vel, steer, dt)
            odo, J_odo = odom_increment_and_jac_from_ve_alpha(ve_dr, alpha_dr, dt_dr)
        
            # Process laser measurements
            z_lsr = self.Z_lsr[self.k_lsr]  # (361,) raw lidar scan
            meas = detectTrees(z_lsr)
            meas[:, 1] = ssa(meas[:, 1])
            
            # filter measurements with range > 10m 
            meas = meas[meas[:, 0] <= 40] # TODO! 
    
            # Create step with accumulated odometry
            step = SLAMStepInput(
                step_index=k_dr,
                timestamp=self.t,
                ve_dr=ve_dr,
                alpha_dr=alpha_dr,
                dt_dr=dt_dr,
                z_lsr=z_lsr,
                odometry=odo,
                measurements=meas,
            )
   
            self.k_lsr += 1
            
        else:
            # No laser measurement - just odometry
            dt_dr = self.T_dr[k_dr + 1] - self.t
            self.t = self.T_dr[k_dr + 1]
            ve_dr = self.Ve_dr[k_dr + 1]
            alpha_dr = self.Alpha_dr[k_dr + 1]
            
            # odo = odometry_func(vel, steer, dt)
            odo, J_odo = odom_increment_and_jac_from_ve_alpha(ve_dr, alpha_dr, dt_dr)

            step = SLAMStepInput(
                step_index=k_dr,
                timestamp=self.t,
                ve_dr=ve_dr,
                alpha_dr=alpha_dr,
                dt_dr=dt_dr,
                z_lsr=None,
                odometry=odo,
                measurements=np.empty((0, 2)),  
            )
        
        return step
    
    def iterate_steps(self, max_steps: Optional[int] = None) -> Iterator[SLAMStepInput]:
        """
        Iterate through all SLAM steps.
        
        Parameters
        ----------
        max_steps : int, optional
            Maximum number of steps to process
        
        Yields
        ------
        SLAMStepInput
            Data for each step
        """
        self._reset_state()
        n_steps = min(max_steps, self.K_dr - 1) if max_steps else self.K_dr - 1
        
        for k in range(1, n_steps):
            try:
                yield self.get_step(k)
            except StopIteration:
                break

    @property
    def lidar(self) -> np.ndarray:
        """Return raw (unsynced) lidar measurements as (K_lsr, 361+1) array where first column is timestamps and rest are ranges."""
        return np.column_stack([self.T_lsr, self.Z_lsr])
    
    @property
    def dead_reckoning(self) -> np.ndarray:
        """Return raw (unsynced) dead reckoning measurement as (K_dr, 3) array of [ve, alpha, t_dr]."""
        return np.column_stack([self.Ve_dr, self.Alpha_dr, self.T_dr])

    @property
    def gps(self) -> np.ndarray:
        """Return raw (unsynced) GPS data as (K_gps, 3) array of [longitud, latitude, t_gps]."""
        return np.column_stack([self.Lo_gps, self.La_gps, self.T_gps])
    
    @property
    def initial_position(self) -> np.ndarray:
        """Return initial position (x, y, theta) from GPS data in local coordinate frame."""
        return np.array([self.Lo_gps[0], self.La_gps[0], 36 * np.pi / 180])
    
    
   

