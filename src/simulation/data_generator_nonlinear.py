import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from div.tuning import NonlinearSimParams, NonlinearFactorGraphParams


class RobotSimulatorSE2:
    """
    Simulates a robot (pose in SE(2)) moving with odometry and observing landmarks.
    Generates both ground truth and noisy measurements.
    """
    
    def __init__(self, simParams: NonlinearSimParams):
        self.simParams = simParams
        self.ground_truth_poses = [gtsam.Pose2(*pose) for pose in self.simParams.poses]
        self.ground_truth_landmarks = self.simParams.landmarks

        self.odometry_noise_sampler = gtsam.Sampler(self.simParams.Q_vec, seed=self.simParams.odom_seed)
        self.measurement_noise_sampler = gtsam.Sampler(self.simParams.R_vec, seed=self.simParams.meas_seed)

        # Store generated measurements
        self.prior_measurement = None
        self.odometry_measurements = []
        self.landmark_measurements = []
    
    def generate_prior_measurement(self) -> Tuple[gtsam.Pose2]:
        """Generate prior measurement (usually just the initial pose)"""
        prior_mean = self.ground_truth_poses[0] # Could add noise if desired
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
            pose_i = self.ground_truth_poses[i]
            pose_i1 = self.ground_truth_poses[i+1]
            true_odom = pose_i.between(pose_i1)
            
            # Add noise
            odom_noise_vec = self.odometry_noise_sampler.sample()
            #noisy_odom = true_odom.compose(gtsam.Pose2(odom_noise_vec)) # gtsam Pose2(np.array) uses exact Exp-map as input is intepreted as canonical coordinates \f$ [T_x,T_y,\theta] \f$ (Lie algebra
            noisy_odom = true_odom.compose(gtsam.Pose2(*odom_noise_vec))  # gtsam Pose2(x,y,theta) deos not use Exp-map, just direct construction (same result as below)
            #noisy_odom = true_odom.retract(odom_noise_vec) # gtsam Pose2.retract(np.array) uses approximate Exp-map, as GTSAM_SLOW_BUT_CORRECT_EXPMAP-flag is set to False in gtsam source
            
            measurements.append((noisy_odom, i, i+1))
        
        self.odometry_measurements = measurements
        return measurements
    
    def generate_landmark_measurements(self) -> Dict[int, List[Tuple[float, gtsam.Rot2]]]:
        """
        Generate landmark measurements.
        Returns:
            Dict[int, List[Tuple[float, gtsam.Rot2]]]
            Mapping from pose index -> list of (range, bearing) measurements.

        """
        measurements = defaultdict(list)
        
        for pose_idx, landmark_indices in self.simParams.observations.items():
            pose_i = self.ground_truth_poses[pose_idx]
            for landmark_idx in landmark_indices:
                landmark_j = self.ground_truth_landmarks[landmark_idx]

                true_range = pose_i.range(landmark_j)
                true_bearing = pose_i.bearing(landmark_j)

                meas_noise_vec = self.measurement_noise_sampler.sample()
                noisy_range = true_range + meas_noise_vec[0]
                noisy_bearing = true_bearing.retract(np.array([meas_noise_vec[1]])) # retract expects np.array

                measurements[pose_idx].append((noisy_range, noisy_bearing))

        self.landmark_measurements = dict(measurements)
        return self.landmark_measurements

    
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
                'prior_sim': self.simParams.P_x0_vec,
                'odometry_sim': self.simParams.Q_vec,
                'measurement_sim': self.simParams.R_vec
            },
            'associations': self.simParams.observations
        }
    
def DynamicRobotSimulatorSE2():
    # TODO: Implement a dynamic simulator where the robot moves based on velocity commands
    pass


def build_nonlinear_factor_graph(sim_data: Dict, fgParams: NonlinearFactorGraphParams = None) -> gtsam.NonlinearFactorGraph:
    """
    Build a Nonlinear factor graph from simulated data.
    
    Args:
        sim_data: Dictionary containing simulation results
        prior_noise_fg: Noise model for prior in factor graph (if None, uses simulation noise)
        odometry_noise_fg: Noise model for odometry in factor graph (if None, uses simulation noise)
        measurement_noise_fg: Noise model for measurements in factor graph (if None, uses simulation noise)

    Returns:
        Configured nfg: NonlinearFactorGraph, initial_estimate: Values
    """
    nfg = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # If fgParams is None, default to simulation noise
    if fgParams is None:
        prior_noise_fg = sim_data['noise_params']['prior_sim']
        odometry_noise_fg = sim_data['noise_params']['odometry_sim']
        measurement_noise_fg = sim_data['noise_params']['measurement_sim']
    else:
        prior_noise_fg = fgParams.P_x0_vec
        odometry_noise_fg = fgParams.Q_vec
        measurement_noise_fg = fgParams.R_vec

    # Create noise models
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(prior_noise_fg)
    odometry_noise_model = gtsam.noiseModel.Diagonal.Sigmas(odometry_noise_fg)
    measurement_noise_model = gtsam.noiseModel.Diagonal.Sigmas(measurement_noise_fg)

    # Add prior factor
    prior_mean = sim_data['measurements']['prior']
    nfg.add(gtsam.PriorFactorPose2(X(0), prior_mean, prior_noise_model))
    initial_estimate.insert(X(0), prior_mean)

    # Add odometry factors
    for odom, from_idx, to_idx in sim_data['measurements']['odometry']:
        nfg.add(gtsam.BetweenFactorPose2(X(from_idx), X(to_idx), odom, odometry_noise_model))
        previous_pose = initial_estimate.atPose2(X(from_idx))
        predicted_pose = previous_pose.compose(odom)
        initial_estimate.insert(X(to_idx), predicted_pose)

    # If dead reckoning is enabled, do not add landmark factors
    if fgParams.dead_reckoning:
        return nfg, initial_estimate
    
    # Add landmark measurement factors
    for i, meas_list in sim_data['measurements']['landmarks'].items():
        ids = sim_data['associations'][i] # !ground-truth associations from simulator!
        for j, (z_range, z_bearing) in zip(ids, meas_list):
            nfg.add(gtsam.BearingRangeFactor2D(X(i), L(j), z_bearing, z_range, measurement_noise_model))  
            if not initial_estimate.exists(L(j)):
                delta_x = z_range * np.cos(z_bearing.theta())
                delta_y = z_range * np.sin(z_bearing.theta())
                predicted_pose = initial_estimate.atPose2(X(i))
                landmark_initial = predicted_pose.transformFrom(gtsam.Point2(delta_x, delta_y))
                initial_estimate.insert(L(j), landmark_initial)

    return nfg, initial_estimate



def compute_error(estimated: gtsam.Values, ground_truth: Dict) -> Dict[str, float]:
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
        est_pose = estimated.atPose2(X(i))
        error = np.linalg.norm(gtsam.Pose2.Logmap(est_pose.between(true_pose))) # Pose2.Logmap gives vector in tangent space TODO: check if correct implementation
        pose_errors.append(error)
    
    # Compute landmark errors
    for i, true_landmark in enumerate(ground_truth['landmarks']):
        est_landmark = estimated.atPoint2(L(i))
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
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create simulation configuration
    simParams = NonlinearSimParams(
        poses=[
            np.array([0.0, 0.0, 0.0]),  # X0
            np.array([2.0, 0.0, 0.0]),  # X1
            np.array([4.0, 0.0, 0.0])   # X2
        ],
        landmarks=[
            np.array([2.0, 2.0]),  # L0
            np.array([4.0, 2.0])   # L1
        ],
        observations={ # could potentially use max distance to determine this
            0: [0],     # X0 sees L0
            1: [0],     # X1 sees L0
            2: [1]      # X2 sees L1
        },
        Q_vec=np.array([0.05, 0.05, 0.0]),
        R_vec=np.array([0.08, 0.08]),
        P_x0_vec=np.array([0.0, 0.0, 0.0])  # No noise on prior
    )
    
    # Run simulation
    simulator = RobotSimulatorSE2(simParams)
    sim_data = simulator.simulate()
    
    # Print ground truth
    print("Ground Truth Poses:")
    for i, pose in enumerate(sim_data['ground_truth']['poses']):
        print(f"  X{i}: {pose}")
    
    print("\nGround Truth Landmarks:")
    for i, landmark in enumerate(sim_data['ground_truth']['landmarks']):
        print(f"  L{i}: {landmark}")
    
    # Build factor graph with different noise models than simulation
    # This represents our (potentially incorrect) belief about the noise
    fgParams = NonlinearFactorGraphParams(
        Q_vec = np.array([0.1, 0.1, 0.0]),  
        R_vec = np.array([0.1, 0.1]),
        P_x0_vec = np.array([0.0, 0.0, 0.0])
    )
    nfg, initial_estimate = build_nonlinear_factor_graph(sim_data, fgParams)
    
    # Solve the factor graph
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(nfg, initial_estimate, params)
    result = optimizer.optimize()

    print("\nEstimated Values:")
    for i in range(len(simParams.poses)):
        print(f"  X{i}: {result.atPose2(X(i))}")
    
    for i in range(len(simParams.landmarks)):
        print(f"  L{i}: {result.atPoint2(L(i))}")
    
    # Compute and print errors
    errors = compute_error(result, sim_data['ground_truth'])
    print(f"\nErrors:")
    print(f"  Pose RMSE: {errors['pose_rmse']:.4f}")
    print(f"  Landmark RMSE: {errors['landmark_rmse']:.4f}")
    
    # Example of running with perfect noise knowledge
    print("\n" + "="*50)
    print("Running with perfect noise knowledge:")

    nfg_perfect, initial_estimate_perfect = build_nonlinear_factor_graph(
        sim_data,
        NonlinearFactorGraphParams(
            Q_vec = simParams.Q_vec,
            R_vec = simParams.R_vec,
            P_x0_vec = simParams.P_x0_vec
        )
    )

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(nfg_perfect, initial_estimate_perfect, params)
    result_perfect = optimizer.optimize()
    errors_perfect = compute_error(result_perfect, sim_data['ground_truth'])
    print(f"  Pose RMSE: {errors_perfect['pose_rmse']:.4f}")
    print(f"  Landmark RMSE: {errors_perfect['landmark_rmse']:.4f}")


# @dataclass
# class SimulationConfig:
#     """Configuration for the simulation"""

#     # Poses (ground truth)
#     poses: List[np.ndarray] = None # (x,y,theta)
    
#     # Landmark positions (ground truth)
#     landmarks: List[np.ndarray] = None # (x,y)
    
#     # Noise parameters for simulation (actual noise added to measurements)
#     prior_noise_sim: np.ndarray = np.array([0.0, 0.0, 0.0])  # Usually no noise on prior
#     odometry_noise_sim: np.ndarray = np.array([0.05, 0.05, 0.01])
#     measurement_noise_sim: np.ndarray = np.array([0.08, 0.08])

#     # Observation pattern: which poses see which landmarks
#     # Format: {pose_idx: [landmark_indices]}
#     observations: Dict[int, List[int]] = None
    
#     # Default test case
#     def __post_init__(self): 
#         if self.poses is None:
#             self.poses = [
#                 np.array([0.0, 0.0, 0.0]),  # X1
#                 np.array([2.0, 0.0, 0.0]),  # X2
#                 np.array([4.0, 0.0, 0.0])   # X3
#             ]
#         if self.landmarks is None:
#             self.landmarks = [
#                 np.array([2.0, 2.0]),  # L1
#                 np.array([4.0, 2.0])   # L2
#             ]
        
#         if self.observations is None:
#             # Default: X1 and X2 see L1, X3 sees L2 (could potentially use max distance instead of hardcoding)
#             self.observations = {
#                 0: [0],  # X1 sees L1
#                 1: [0],  # X2 sees L1
#                 2: [1]   # X3 sees L2
#             }