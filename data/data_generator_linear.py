import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for the simulation"""

    # Poses (ground truth)
    poses: List[np.ndarray] = None
    
    # Landmark positions (ground truth)
    landmarks: List[np.ndarray] = None

    # Observation pattern: which poses see which landmarks
    # Format: {pose_idx: [landmark_indices]}
    observations: Dict[int, List[int]] = None
    
    # Noise parameters for simulation (actual noise added to measurements)
    prior_noise_sim: np.ndarray = np.array([0.0, 0.0])  # Usually no noise on prior
    odometry_noise_sim: np.ndarray = np.array([0.05, 0.05])
    measurement_noise_sim: np.ndarray = np.array([0.08, 0.08])

    # Seeds for reproducibility
    odom_seed: int = 42
    meas_seed: int = 42
    
    # Default test case
    def __post_init__(self): 
        if self.poses is None:
            self.poses = [
                np.array([0.0, 0.0]),  # X1
                np.array([2.0, 0.0]),  # X2
                np.array([4.0, 0.0])   # X3
            ]
        if self.landmarks is None:
            self.landmarks = [
                np.array([2.0, 2.0]),  # L1
                np.array([4.0, 2.0])   # L2
            ]
        
        if self.observations is None:
            # Default: X1 and X2 see L1, X3 sees L2 (could potentially use max distance instead of hardcoding)
            self.observations = {
                0: [0],  # X1 sees L1
                1: [0],  # X2 sees L1
                2: [1]   # X3 sees L2
            }


class RobotSimulatorR2:
    """
    Simulates a robot (pose in R^2) moving with odometry and observing landmarks.
    Generates both ground truth and noisy measurements.
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.ground_truth_poses = self.config.poses
        self.ground_truth_landmarks = self.config.landmarks

        self.odometry_noise_sampler = gtsam.Sampler(
            self.config.odometry_noise_sim, seed=self.config.odom_seed)
        self.measurement_noise_sampler = gtsam.Sampler(
            self.config.measurement_noise_sim, seed=self.config.meas_seed)

        # Store generated measurements
        self.prior_measurement = None
        self.odometry_measurements = []
        self.landmark_measurements = []
    
    def generate_prior_measurement(self) -> Tuple[np.ndarray]:
        """Generate prior measurement (usually just the initial pose)"""
        prior_mean = self.ground_truth_poses[0]  # Could add noise if desired
        self.prior_measurement = prior_mean
        return prior_mean
    
    def generate_odometry_measurements(self) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate odometry measurements between consecutive poses.
        Returns list of (measurement, from_idx, to_idx)
        """
        measurements = []
        
        for i in range(len(self.ground_truth_poses) - 1):
            # Ground truth odometry
            true_odom = self.ground_truth_poses[i+1] - self.ground_truth_poses[i]
            
            # Add noise
            odom_noise_vec = self.odometry_noise_sampler.sample()
            noisy_odom = true_odom + odom_noise_vec

            measurements.append((noisy_odom, i, i+1))
        
        self.odometry_measurements = measurements
        return measurements
    
    def generate_landmark_measurements(self) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate landmark measurements.
        Returns list of (measurement, pose_idx, landmark_idx)
        """
        measurements = []
        
        for pose_idx, landmark_indices in self.config.observations.items():
            for landmark_idx in landmark_indices:
                # Ground truth relative position
                true_measurement = (
                    self.ground_truth_landmarks[landmark_idx] - 
                    self.ground_truth_poses[pose_idx]
                )
                
                # Add noise
                meas_noise_vec = self.measurement_noise_sampler.sample()
                noisy_measurement = true_measurement + meas_noise_vec

                measurements.append((noisy_measurement, pose_idx, landmark_idx))
        
        self.landmark_measurements = measurements
        return measurements
    
    def simulate(self) -> Dict:
        """Run complete simulation and return all data"""

        # Generate measurements
        prior = self.generate_prior_measurement()
        odometry = self.generate_odometry_measurements()
        landmarks = self.generate_landmark_measurements()
        
        return {
            'ground_truth': {
                'poses': self.ground_truth_poses,
                'landmarks': self.ground_truth_landmarks
            },
            'measurements': {
                'prior': prior,
                'odometry': odometry,
                'landmarks': landmarks
            },
            'noise_params': {
                'prior_sim': self.config.prior_noise_sim,
                'odometry_sim': self.config.odometry_noise_sim,
                'measurement_sim': self.config.measurement_noise_sim
            }
        }


def build_linear_factor_graph(sim_data: Dict, 
                      prior_noise_fg: Optional[np.ndarray] = None,
                      odometry_noise_fg: Optional[np.ndarray] = None,
                      measurement_noise_fg: Optional[np.ndarray] = None,
                      dead_reckoning: bool = False) -> gtsam.GaussianFactorGraph:
    """
    Build a Gaussian factor graph from simulated data.
    
    Args:
        sim_data: Dictionary containing simulation results
        prior_noise_fg: Noise model for prior in factor graph (if None, uses simulation noise)
        odometry_noise_fg: Noise model for odometry in factor graph (if None, uses simulation noise)
        measurement_noise_fg: Noise model for measurements in factor graph (if None, uses simulation noise)
    
    Returns:
        Configured GaussianFactorGraph
    """
    gfg = gtsam.GaussianFactorGraph()
    
    # Use provided noise models or default to simulation noise
    if prior_noise_fg is None:
        prior_noise_fg = sim_data['noise_params']['prior_sim']  # Default uncertainty
    if odometry_noise_fg is None:
        odometry_noise_fg = sim_data['noise_params']['odometry_sim']
    if measurement_noise_fg is None:
        measurement_noise_fg = sim_data['noise_params']['measurement_sim']
    
    # Create noise models
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(prior_noise_fg)
    odometry_noise_model = gtsam.noiseModel.Diagonal.Sigmas(odometry_noise_fg)
    measurement_noise_model = gtsam.noiseModel.Diagonal.Sigmas(measurement_noise_fg)
    
    # Add prior factor
    prior_mean = sim_data['measurements']['prior']
    gfg.add(gtsam.JacobianFactor(
        X(1), np.eye(2), 
        gtsam.Point2(prior_mean[0], prior_mean[1]), 
        prior_noise_model
    ))
    
    # Add odometry factors
    for odom, from_idx, to_idx in sim_data['measurements']['odometry']:
        gfg.add(gtsam.JacobianFactor(
            X(from_idx + 1), -np.eye(2), 
            X(to_idx + 1), np.eye(2),
            gtsam.Point2(odom[0], odom[1]),
            odometry_noise_model
        ))
    
    # Add landmark measurement factors
    if dead_reckoning:
        pass
    else:
        for meas, pose_idx, landmark_idx in sim_data['measurements']['landmarks']:
            gfg.add(gtsam.JacobianFactor(
                X(pose_idx + 1), -np.eye(2),
                L(landmark_idx + 1), np.eye(2),
                gtsam.Point2(meas[0], meas[1]),
                measurement_noise_model
            ))
    
    return gfg


def compute_error(estimated: gtsam.VectorValues, ground_truth: Dict) -> Dict[str, float]:
    """
    Compute error between estimated values and ground truth.
    
    Args:
        estimated: GTSAM solution
        ground_truth: Dictionary with 'poses' and 'landmarks' lists
    
    Returns:
        Dictionary with RMS errors for poses and landmarks
    """
    pose_errors = []
    landmark_errors = []
    
    # Compute pose errors
    for i, true_pose in enumerate(ground_truth['poses']):
        est_pose = estimated.at(X(i + 1))
        error = np.linalg.norm(est_pose - true_pose)
        pose_errors.append(error)
    
    # Compute landmark errors
    for i, true_landmark in enumerate(ground_truth['landmarks']):
        est_landmark = estimated.at(L(i + 1))
        error = np.linalg.norm(est_landmark - true_landmark)
        landmark_errors.append(error)
    
    return {
        'pose_rmse': np.sqrt(np.mean(np.square(pose_errors))),
        'landmark_rmse': np.sqrt(np.mean(np.square(landmark_errors))),
        'pose_errors': pose_errors,
        'landmark_errors': landmark_errors
    }


# Example usage
if __name__ == "__main__":
    
    # Create simulation configuration
    config = SimulationConfig(
        poses=[
            np.array([0.0, 0.0]),  # X1
            np.array([2.0, 0.0]),  # X2
            np.array([4.0, 0.0])   # X3
        ],
        landmarks=[
            np.array([2.0, 2.0]),  # L1
            np.array([4.0, 2.0])   # L2
        ],
        observations={ # could potentially use max distance to determine this
            0: [0],     # X1 sees L1
            1: [0],     # X2 sees L1
            2: [1]      # X3 sees L2
        },
        prior_noise_sim=np.array([0.0, 0.0]),
        odometry_noise_sim=np.array([0.05, 0.05]),
        measurement_noise_sim=np.array([0.08, 0.08])
    )
    
    # Run simulation
    simulator = RobotSimulatorR2(config)
    sim_data = simulator.simulate()
    
    # Print ground truth
    print("Ground Truth Poses:")
    for i, pose in enumerate(sim_data['ground_truth']['poses']):
        print(f"  X{i+1}: {pose}")
    
    print("\nGround Truth Landmarks:")
    for i, landmark in enumerate(sim_data['ground_truth']['landmarks']):
        print(f"  L{i+1}: {landmark}")
    
    # Build factor graph with different noise models than simulation
    # This represents our (potentially incorrect) belief about the noise
    gfg = build_linear_factor_graph(
        sim_data,
        prior_noise_fg=np.array([0.0, 0.0]),      # Our prior uncertainty
        odometry_noise_fg=np.array([0.1, 0.1]),   # Our odometry noise model (overconfident)
        measurement_noise_fg=np.array([0.1, 0.1]) # Our measurement noise model
    )
    
    # Solve the factor graph
    result = gfg.optimize()
    
    print("\nEstimated Values:")
    for i in range(len(config.poses)):
        print(f"  X{i+1}: {result.at(X(i+1))}")
    
    for i in range(len(config.landmarks)):
        print(f"  L{i+1}: {result.at(L(i+1))}")
    
    # Compute and print errors
    errors = compute_error(result, sim_data['ground_truth'])
    print(f"\nErrors:")
    print(f"  Pose RMSE: {errors['pose_rmse']:.4f}")
    print(f"  Landmark RMSE: {errors['landmark_rmse']:.4f}")
    
    # Example of running with perfect noise knowledge
    print("\n" + "="*50)
    print("Running with perfect noise knowledge:")
    
    gfg_perfect = build_linear_factor_graph(
        sim_data,
        prior_noise_fg=config.prior_noise_sim,      # True noise
        odometry_noise_fg=config.odometry_noise_sim,    # True noise
        measurement_noise_fg=config.measurement_noise_sim # True noise
    )
    
    result_perfect = gfg_perfect.optimize()
    errors_perfect = compute_error(result_perfect, sim_data['ground_truth'])
    print(f"  Pose RMSE: {errors_perfect['pose_rmse']:.4f}")
    print(f"  Landmark RMSE: {errors_perfect['landmark_rmse']:.4f}")

