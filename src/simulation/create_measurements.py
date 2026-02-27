import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from config import InferenceConfig, SimulationConfig, SLAMConfig


class RobotSimulatorSE2:
    """
    Simulates a robot (pose in SE(2)) moving with odometry and observing landmarks.
    Returns noisy measurements (odometry and landmark observations) based on ground truth data.
    """

    def __init__(self, gt_data: dict, cfg: SimulationConfig):
        self.gt_poses2 = [gtsam.Pose2(*pose) for pose in gt_data["poses"]]
        self.gt_landmarks = gt_data["landmarks"]
        self.gt_observations = gt_data["observations"]  # dict[int, list[int]]

        self.odometry_noise_sampler = gtsam.Sampler(
            sigmas=cfg.noise.odometry_std, seed=cfg.odom_seed
        )
        self.measurement_noise_sampler = gtsam.Sampler(
            sigmas=cfg.noise.measurement_std, seed=cfg.meas_seed # TODO: check order of measurement vec
        )

    def generate_odometry_measurements(self) -> list[tuple[np.ndarray, int, int]]:
        """
        Generate odometry measurements between consecutive poses.
        Returns list of (measurement, from_idx, to_idx)
        """
        measurements = []

        for i in range(len(self.gt_poses2) - 1):
            # Ground truth odometry
            pose_i = self.gt_poses2[i]
            pose_i1 = self.gt_poses2[i + 1]
            true_odom = pose_i.between(pose_i1)

            # Add noise
            odom_noise_vec = self.odometry_noise_sampler.sample()
            # noisy_odom = true_odom.compose(gtsam.Pose2(odom_noise_vec)) # gtsam Pose2(np.array) uses exact Exp-map as input is intepreted as canonical coordinates \f$ [T_x,T_y,\theta] \f$ (Lie algebra
            noisy_odom = true_odom.compose(
                gtsam.Pose2(*odom_noise_vec)
            )  # gtsam Pose2(x,y,theta) deos not use Exp-map, just direct construction (same result as below)
            # noisy_odom = true_odom.retract(odom_noise_vec) # gtsam Pose2.retract(np.array) uses approximate Exp-map, as GTSAM_SLOW_BUT_CORRECT_EXPMAP-flag is set to False in gtsam source

            measurements.append((noisy_odom, i, i + 1))

        return measurements

    def generate_landmark_measurements(
        self,
    ) -> dict[int, list[tuple[float, gtsam.Rot2]]]:
        """
        Generate landmark measurements.
        Returns:
            dict[int, list[tuple[float, gtsam.Rot2]]]
            Mapping from pose index -> list of (range, bearing) measurements.

        """
        measurements: dict[int, list[tuple[float, gtsam.Rot2]]] = dict()

        for pose_idx, landmark_indices in self.gt_observations.items():
            pose_i = self.gt_poses2[pose_idx]
            for landmark_idx in landmark_indices:
                landmark_j = self.gt_landmarks[landmark_idx]

                true_range = pose_i.range(landmark_j)
                true_bearing = pose_i.bearing(landmark_j)

                meas_noise_vec = self.measurement_noise_sampler.sample()
                noisy_range = true_range + meas_noise_vec[0]
                noisy_bearing = true_bearing.retract(
                    np.array([meas_noise_vec[1]])
                )  # retract expects np.array

                measurements[pose_idx].append((noisy_range, noisy_bearing))

        return measurements

    def simulate(self) -> dict:
        """Run complete simulation and return all data"""

        # Generate measurements
        z_odometry = self.generate_odometry_measurements()
        z_landmarks = self.generate_landmark_measurements()

        return {"odometry": z_odometry, "landmarks": z_landmarks}


def DynamicRobotSimulatorSE2():
    # TODO: Implement a dynamic simulator where the robot moves based on velocity commands
    pass


def build_nonlinear_factor_graph(
    sim_data: dict, gt_data: dict, cfg: InferenceConfig,
) -> gtsam.NonlinearFactorGraph:
    """
    Build a Nonlinear factor graph from simulated data.

    Args:
        sim_data: Dictionary containing simulation results
        gt_data: Dictionary containing ground truth data
        cfg: Configuration with inference and simulation parameters and noise models

    Returns:
        Configured nfg: NonlinearFactorGraph, initial_estimate: Values
    """
    nfg = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Create noise models
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.initial_vec)
    odometry_noise_model = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.odometry_std) 
    measurement_noise_model = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.measurement_std) # TODO: check order of measurement vec

    # Add prior factor
    prior_mean = gtsam.Pose2(*gt_data["poses"][0])
    nfg.add(gtsam.PriorFactorPose2(X(0), prior_mean, prior_noise_model))
    initial_estimate.insert(X(0), prior_mean)

    # Add odometry factors
    for odom, from_idx, to_idx in sim_data["odometry"]:
        nfg.add(
            gtsam.BetweenFactorPose2(X(from_idx), X(to_idx), odom, odometry_noise_model)
        )
        previous_pose = initial_estimate.atPose2(X(from_idx))
        predicted_pose = previous_pose.compose(odom)
        initial_estimate.insert(X(to_idx), predicted_pose)

    # If dead reckoning is enabled, do not add landmark factors
    if cfg.dead_reckoning:
        return nfg, initial_estimate

    # Add landmark measurement factors
    for i, meas_list in sim_data["landmarks"].items():
        ids = gt_data["observations"][i]  # !ground-truth associations from simulator!
        for j, (z_range, z_bearing) in zip(ids, meas_list):
            nfg.add(
                gtsam.BearingRangeFactor2D(
                    X(i), L(j), z_bearing, z_range, measurement_noise_model
                )
            )
            if not initial_estimate.exists(L(j)):
                delta_x = z_range * np.cos(z_bearing.theta())
                delta_y = z_range * np.sin(z_bearing.theta())
                predicted_pose = initial_estimate.atPose2(X(i))
                landmark_initial = predicted_pose.transformFrom(
                    gtsam.Point2(delta_x, delta_y)
                )
                initial_estimate.insert(L(j), landmark_initial)

    return nfg, initial_estimate


def compute_error(estimated: gtsam.Values, ground_truth: dict) -> dict[str, float]:
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

    # Convert ground truth poses to gtsam Pose2
    gt_poses2 = [gtsam.Pose2(*pose) for pose in ground_truth["poses"]]

    # Compute pose errors
    for i, true_pose in enumerate(gt_poses2):
        est_pose = estimated.atPose2(X(i))
        error = np.linalg.norm(
            gtsam.Pose2.Logmap(est_pose.between(true_pose))
        )  # Pose2.Logmap gives vector in tangent space TODO: check if correct implementation
        pose_errors.append(error)

    # Compute landmark errors
    for i, true_landmark in enumerate(ground_truth["landmarks"]):
        est_landmark = estimated.atPoint2(L(i))
        error = np.linalg.norm(est_landmark - true_landmark)
        landmark_errors.append(error)

    return {
        "pose_rmse": np.sqrt(np.mean(np.square(pose_errors))),
        "landmark_rmse": np.sqrt(np.mean(np.square(landmark_errors))),
        "pose_errors": pose_errors,
        "landmark_errors": landmark_errors,
    }


# Example usage
if __name__ == "__main__":
    

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create ground truth data
    gt_data = {
        "poses": [
            np.array([0.0, 0.0, 0.0]),  # X0
            np.array([2.0, 0.0, 0.0]),  # X1
            np.array([4.0, 0.0, 0.0]),  # X2
        ],
        "landmarks": [
            np.array([2.0, 2.0]),  # L0
            np.array([4.0, 2.0]),  # L1
        ],
        "observations": {  # could potentially use max distance to determine this
            0: [0],  # X0 sees L0
            1: [0],  # X1 sees L0
            2: [1],  # X2 sees L1
        },
    }

    cfg_sim = SimulationConfig()
    cfg_sim.noise.x_std = 0.05
    cfg_sim.noise.y_std = 0.05
    cfg_sim.noise.theta_std_deg = 0.1
    cfg_sim.noise.range_std = 0.08
    cfg_sim.noise.bearing_std_deg = 3

    # Run simulation
    simulator = RobotSimulatorSE2(gt_data, cfg_sim)
    sim_data = simulator.simulate()

    # Print ground truth
    print("Ground Truth Poses:")
    for i, pose in enumerate(gt_data["poses"]):
        print(f"  X{i}: {pose}")

    print("\nGround Truth Landmarks:")
    for i, landmark in enumerate(gt_data["landmarks"]):
        print(f"  L{i}: {landmark}")

    # Build factor graph with different noise models than simulation
    # This represents our (potentially incorrect) belief about the noise
    cfg_inf = InferenceConfig()
    cfg_inf.noise.odometry_std = np.array([0.1, 0.1, 0.0]) # TODO: implement setters
    cfg_inf.noise.measurement_std = np.array([0.1, 0.1])  # TODO: implement setters
    cfg_inf.noise.initial_vec = np.array([0.0, 0.0, 0.0]) # TODO: assign setters

    nfg, initial_estimate = build_nonlinear_factor_graph(sim_data, gt_data, cfg_inf)

    # Solve the factor graph
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(nfg, initial_estimate, params)
    result = optimizer.optimize()

    print("\nEstimated Values:")
    for i in range(len(gt_data["poses"])):
        print(f"  X{i}: {result.atPose2(X(i))}")

    for i in range(len(gt_data["landmarks"])):
        print(f"  L{i}: {result.atPoint2(L(i))}")

    # Compute and print errors
    errors = compute_error(result, gt_data)
    print(f"\nErrors:")
    print(f"  Pose RMSE: {errors['pose_rmse']:.4f}")
    print(f"  Landmark RMSE: {errors['landmark_rmse']:.4f}")

    # Example of running with perfect noise knowledge
    print("\n" + "=" * 50)
    print("Running with perfect noise knowledge:")

    cfg_inf.noise.odometry_std = cfg_sim.noise.odometry_std
    cfg_inf.noise.measurement_std = cfg_sim.noise.measurement_std
    cfg_inf.noise.initial_vec = cfg_sim.noise.initial_vec

    nfg_perfect, initial_estimate_perfect = build_nonlinear_factor_graph(
        sim_data,
        gt_data,
        cfg_inf,
    )

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        nfg_perfect, initial_estimate_perfect, params
    )
    result_perfect = optimizer.optimize()
    errors_perfect = compute_error(result_perfect, gt_data)
    print(f"  Pose RMSE: {errors_perfect['pose_rmse']:.4f}")
    print(f"  Landmark RMSE: {errors_perfect['landmark_rmse']:.4f}")
