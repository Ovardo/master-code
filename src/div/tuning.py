import gtsam
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# -------------------------------
# Nonlinear SLAM params
# -------------------------------

@dataclass
class BaseNoiseParams:
    """Base class for process, measurement, and prior noise parameters."""

    # --- Option 1: scalar sigmas ---
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    sigma_theta: Optional[float] = None

    sigma_range: Optional[float] = None
    sigma_bearing: Optional[float] = None

    sigma_x0: Optional[float] = None
    sigma_y0: Optional[float] = None
    sigma_theta0: Optional[float] = None

    # --- Option 2: pre-packed vectors ---
    Q_vec: Optional[np.ndarray] = None
    R_vec: Optional[np.ndarray] = None
    P_x0_vec: Optional[np.ndarray] = None

    def __post_init__(self): 
        """Populate Q_vec, R_vec, and P_x0_vec (and mirror scalars)."""

        # --- Process noise ---
        if self.Q_vec is None: # TODO ensure only one option provided
            if None in (self.sigma_x, self.sigma_y, self.sigma_theta):
                raise ValueError("Provide either Q_vec or sigma_x/y/theta.")
            self.Q_vec = np.array([self.sigma_x, self.sigma_y, self.sigma_theta])
        else:
            self.sigma_x, self.sigma_y, self.sigma_theta = self.Q_vec

        # --- Measurement noise ---
        if self.R_vec is None: # TODO: gtsam expext (bearing, range) order, be aware, need to fiz!
            if None in (self.sigma_range, self.sigma_bearing):
                raise ValueError("Provide either R_vec or sigma_range/bearing.")
            self.R_vec = np.array([self.sigma_range, self.sigma_bearing])
        else:
            self.sigma_range, self.sigma_bearing = self.R_vec

        # --- Prior noise ---
        if self.P_x0_vec is None:
            if None in (self.sigma_x0, self.sigma_y0, self.sigma_theta0):
                raise ValueError("Provide either P_x0_vec or sigma_x0/y0/theta0.")
            self.P_x0_vec = np.array([self.sigma_x0, self.sigma_y0, self.sigma_theta0])
        else:
            self.sigma_x0, self.sigma_y0, self.sigma_theta0 = self.P_x0_vec
    
    def __repr__(self):
        return (f"Q_vec: {self.Q_vec},\nR_vec: {self.R_vec},\nP_x0_vec: {self.P_x0_vec}")
    

@dataclass
class NonlinearSimParams(BaseNoiseParams):
    """Parameters for nonlinear simulation (ground truth generation)."""
    poses: Optional[List[np.ndarray]] = None
    landmarks: Optional[List[np.ndarray]] = None  
    observations: Optional[Dict[int, List[int]]] = None # pose index -> list of landmark indices
    odom_seed: int = 42
    meas_seed: int = 42

    def __repr__(self):
        return super().__repr__() 

@dataclass
class NonlinearFactorGraphParams(BaseNoiseParams):
    """Parameters for the nonlinear factor graph (estimator inference)."""
    init_state: Optional[np.ndarray] = None
    dead_reckoning: bool = False
    association_type : str = "known"  # "known", "jcbb",
    alpha_individual: float = 0.9999   # confidence levels for individual compatibility test
    alpha_joint: float = 0.9999        # confidence levels for joint compatibility test
    r_local: float = 20.0              # local feature filtering radius for JCBB TODO: should be adjusted to sensor range + margin
    use_isam: bool = False             # whether to use iSAM2 incremental solver or full batch optimization
    optimizer_params: gtsam.LevenbergMarquardtParams = field(default_factory=gtsam.LevenbergMarquardtParams)
    sensor_offset: Optional[np.ndarray] = None  # np.array([dx, dy]) offset of sensor wrt robot body frame

    def __repr__(self):
        return super().__repr__()


### -------------------------------
### Linear SLAM params
### -------------------------------

@dataclass
class LinearSimParams:
    # Process noise
    sigma_x: float = 0.25 
    sigma_y: float = 0.25
    
    # Measurement noise
    sigma_zx: float = 0.1
    sigma_zy: float = 0.1

    # Prior noise
    sigma_x0: float = 0.1
    sigma_y0: float = 0.1

    poses: List[np.ndarray] = None  # list of np.array([x, y, theta])
    landmarks: List[np.ndarray] = None  # list of np.array([x, y])
    observations: Dict[int, List[int]] = None  # dict mapping pose index to list of landmark indices

    odom_seed: int = 42  # random seed for data generation
    meas_seed: int = 42  # random seed for data generation

    def __post_init__(self):
        self.Q_vec = np.array([self.sigma_x**2, self.sigma_y**2])
        self.R_vec = np.array([self.sigma_zx**2, self.sigma_zy**2])


@dataclass
class LinearFactorGraphParams:
    # Process noise (Q)
    sigma_x: float = 0.25  
    sigma_y: float = 0.25  
    
    # Measurement noise covariance (R)
    sigma_zx: float = 5 
    sigma_zy: float = 3.1 

    #init_state: np.ndarray = None  

    def __post_init__(self):
        self.Q_vec = np.array([self.sigma_x**2, self.sigma_y**2])
        self.R_vec = np.array([self.sigma_zx**2, self.sigma_zy**2])

# -------------------------------

if __name__ == "__main__":
    
    # Example usage
    sim_params = NonlinearSimParams(
        poses=[
            np.array([0.0, 0.0, 0.0]),  # X0
            np.array([2.0, 0.0, 0.0]),  # X1
            np.array([4.0, 0.0, 0.0])   # X2
        ],
        landmarks=[
            np.array([2.0, 2.0]),  # L0
            np.array([4.0, 2.0])   # L1
        ],
        observations={
            0: [0],     # X0 sees L0
            1: [0],     # X1 sees L0
            2: [1]      # X2 sees L1
        },
        # Remark: elements are stddev, not variance
        Q_vec=np.array([0.1, 0.1, 0.05]),     # Process noise
        R_vec=np.array([0.1, 0.05]),          # Measurement
        P_x0_vec=np.array([0.05, 0.05, 0.05]) # Prior noise
    )

    # Can also declare using scalars
    fg_params = NonlinearFactorGraphParams(
        sigma_x=0.1,
        sigma_y=0.1,
        sigma_theta=0.05,
        sigma_range=0.1,
        sigma_bearing=0.05,
        sigma_x0=0.05,
        sigma_y0=0.05,
        sigma_theta0=0.05,
        init_state=np.array([0.0, 0.0, 0.0])
    )



# @dataclass
# class NonlinearSimParams:
#     # Process noise
#     sigma_x: float = 0.1 
#     sigma_y: float = 0.1
#     sigma_theta: float = 0.05
    
#     # Measurement noise
#     sigma_range: float = 0.1
#     sigma_bearing: float = 0.05

#     # Prior noise
#     sigma_x0: float = 0.01
#     sigma_y0: float = 0.01
#     sigma_theta0: float = 0.01

#     poses: List[np.ndarray] = None  # list of np.array([x, y, theta])
#     landmarks: List[np.ndarray] = None  # list of np.array([x, y])
#     observations: Dict[int, List[int]] = None  # dict mapping pose index to list of landmark indices

#     odom_seed: int = 42  # random seed for data generation
#     meas_seed: int = 42  # random seed for data generation

#     def __post_init__(self):
#         self.Q_vec = np.array([self.sigma_x, self.sigma_y, self.sigma_theta])
#         self.R_vec = np.array([self.sigma_range, self.sigma_bearing])
#         self.P_x0_vec = np.array([self.sigma_x0, self.sigma_y0, self.sigma_theta0])


# @dataclass
# class NonlinearFactorGraphParams:
#     # Process noise (Q)
#     sigma_x: float = 0.1 
#     sigma_y: float = 0.1  
#     sigma_theta: float = 0.05  
    
#     # Measurement noise covariance (R)
#     sigma_range: float = 0.1 
#     sigma_bearing: float = 0.05 

#     # Prior noise on pose (P_x0)
#     sigma_x0: float = 0.05
#     sigma_y0: float = 0.05
#     sigma_theta0: float = 0.05

#     init_state: np.ndarray = None  

#     def __post_init__(self):
#         self.Q_vec = np.array([self.sigma_x, self.sigma_y, self.sigma_theta])
#         self.R_vec = np.array([self.sigma_range, self.sigma_bearing])
#         self.P_x0_vec = np.array([self.sigma_x0, self.sigma_y0, self.sigma_theta0])