"""
Victoria Park dataset loader module.
Handles loading and preprocessing of Victoria Park SLAM dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.io import loadmat

from utils.utils_math import ssa
from utils.utils_victoria_park import Car, detectTrees, odometry


@dataclass
class SLAMStep:
    """Data for a single SLAM processing step."""
    k_odo: int  # Odometry step index
    odometry: np.ndarray # (x, y, theta) odometry since last step
    measurements: np.ndarray  # l
    has_laser: bool  # Whether this step includes laser measurements
    timestamp: float  # Current timestamp




class VictoriaParkLoader:
    """
    Iterator-based loader for Victoria Park SLAM dataset.
    
    Handles all timing synchronization between odometry and laser measurements,
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
        self.car = Car()  # default parameters for Victoria Park dataset
        
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
        
        # Extract and preprocess time data
        self.time_odo = (realSLAM_ws["time"] / 1000).ravel() # Convert ms to s
        self.time_lsr = (realSLAM_ws["TLsr"] / 1000).ravel() # Convert ms to s
        self.time_gps = (realSLAM_ws["timeGps"] / 1000).ravel() # Convert ms to s
        
        # Extract sensor data
        self.steering = realSLAM_ws["steering"].ravel()
        self.speed = realSLAM_ws["speed"].ravel()
        self.laser = realSLAM_ws["LASER"] / 100 # convert from cm to m
        
        # GPS data
        self.La_m = realSLAM_ws["La_m"].ravel()
        self.Lo_m = realSLAM_ws["Lo_m"].ravel()
        
        # Data sizes
        self.K_odo= self.time_odo.size
        self.K_lsr = self.time_lsr.size
        self.K_gps = self.time_gps.size
    
    def _reset_state(self):
        """Reset iteration state to beginning."""
        self.k_lsr = 1  # First laser measurement (0 seems to be off in timing)
        self.t = self.time_odo[0] # time of last processed step (start with first odometry timestamp)
    
    @property
    def initial_position(self) -> np.ndarray:
        """Return initial position (x, y, theta) from GPS data."""
        return np.array([self.Lo_m[0], self.La_m[0], 36 * np.pi / 180])
    

    def get_step(self, k_odo: int) -> SLAMStep:
        """
        Get processed data for odometry step k.
        
        Parameters
        ----------
        k_odo : int
            Odometry step index (1-based)
        
        Returns
        -------
        SLAMStep
            Processed step data including odometry and any measurements
        """
        if k_odo >= self.K_odo- 1:
            raise StopIteration(f"Reached end of data at step {k_odo}")
        
        # Check if we have a laser measurement at this step
        has_laser = (self.k_lsr < self.K_lsr - 1 and 
                     self.time_lsr[self.k_lsr] <= self.time_odo[k_odo + 1])
        
        if has_laser:
            # Compute odometry up to laser measurement time
            dt = self.time_lsr[self.k_lsr] - self.t
            if dt < 0:
                raise ValueError(f"Negative time increment at step {k_odo}")
            
            self.t = self.time_lsr[self.k_lsr]  
            odo = odometry(self.speed[k_odo + 1], self.steering[k_odo + 1], dt, self.car)
        
            # Process laser measurements
            meas = detectTrees(self.laser[self.k_lsr])
            meas[:, 1] = ssa(meas[:, 1])
    
            # Create step with accumulated odometry
            step = SLAMStep(
                k_odo=k_odo,
                odometry=odo,
                measurements=meas,
                has_laser=True,
                timestamp=self.t
            )
   
            self.k_lsr += 1
            
        else:
            # No laser measurement - just odometry
            dt = self.time_odo[k_odo + 1] - self.t
            self.t = self.time_odo[k_odo + 1]
            odo = odometry(self.speed[k_odo + 1], self.steering[k_odo + 1], dt, self.car)

            step = SLAMStep(
                k_odo=k_odo,
                odometry=odo,  
                measurements=np.empty((0, 2)),  
                has_laser=False,
                timestamp=self.t
            )
        
        return step
    
  
    def iterate_steps(self, max_steps: Optional[int] = None) -> Iterator[SLAMStep]:
        """
        Iterate through all SLAM steps.
        
        Parameters
        ----------
        max_steps : int, optional
            Maximum number of steps to process
        
        Yields
        ------
        SLAMStep
            Data for each step
        """
        self._reset_state()
        n_steps = min(max_steps, self.K_odo - 1) if max_steps else self.K_odo - 1
        
        for k in range(1, n_steps):
            try:
                yield self.get_step(k)
            except StopIteration:
                break


