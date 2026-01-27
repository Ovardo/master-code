import gtsam
import numpy as np
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
from gtbook.display import show
from gtsam.symbol_shorthand import X, L
from ekf_slam import EKFSLAM
from utils.utils_math import kstr, pose2_to_array, rotmat2
from simulation.data_generator_nonlinear import RobotSimulatorSE2, build_nonlinear_factor_graph, compute_error
from div.tuning import NonlinearSimParams, NonlinearFactorGraphParams
from utils.utils_plot import plot_result, plot_ellipse, plot_se2_covariance_on_manifold_gtsam, MultivariateNormalParameters



def nonlinear_batch_slam_example_with_and_without_landamrks():
    
    simParams = NonlinearSimParams(
        poses=[
            np.array([0.0, 0.0, 0.0]),  # X0
            np.array([2.0, 0.0, 0.0]),  # X1
            np.array([4.0, 0.0, 0.0])   # X2
        ],
        landmarks=[
            np.array([2.0, 2.0]),  # L0
            np.array([4.0, 2.0]),  # L1
        ],
        observations={ # could potentially use max distance to determine this
            0: [0],     # X0 sees L0
            1: [0],     # X1 sees L0
            2: [1]      # X2 sees L1
        },
        Q_vec = np.array([0.03, 0.05, 0.05]),  # Process noise (x, y, theta)
        R_vec = np.array([0.01, 0.05]),       # Measurement (bearing, range)
        P_x0_vec = np.array([0.0, 0.0, 0.0]), # Prior noise

        odom_seed=42,
        meas_seed=42,
    )

    simulator = RobotSimulatorSE2(simParams)
    sim_data = simulator.simulate()

    pose_vars = [X(i) for i in range(len(sim_data['ground_truth']['poses']))]
    landmark_vars = [L(i) for i in range(len(sim_data['ground_truth']['landmarks']))]
    all_vars = pose_vars + landmark_vars

    poses_gt = sim_data['ground_truth']['poses']
    landmarks_gt = sim_data['ground_truth']['landmarks']
    num_landmarks = len(landmarks_gt)

    odometry_meas = sim_data['measurements']['odometry']
    landmark_meas = sim_data['measurements']['landmarks']   # dict: pose_idx -> list[(range, Rot2)]
    associations = sim_data['associations']                 # dict: pose_idx -> list[landmark_idx]

    # Inference parameters (std)
    Q_vec = np.array([0.05, 0.1, 0.2])
    R_vec = np.array([0.02, 0.1]) # (Bearing, range) GTSAM ordering
    P_x0_vec = np.array([0.2, 0.2, 0.1])
    
    # ------------------------------------------------------------------
    # EKF SLAM
    # ------------------------------------------------------------------
    Px_0 = np.diag(P_x0_vec**2)               # 3x3

    # Initial pose from ground truth (or use prior mean from sim_data['measurements']['prior'])
    p0 = poses_gt[0]
    x0 = np.array([p0.x(), p0.y(), p0.theta()])

    # Initial state vector: pose + landmarks (start landmarks at origin or some guess)
    eta_0 = np.hstack([
        x0,
        np.ones(2 * num_landmarks)  # m0, m1, ... (no prior info)
    ])

    # Full initial covariance
    P_0 = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
    P_0[:3, :3] = Px_0
    # landmark covariances will be filled with big_var inside EKFSLAM if zeros

    ekf = EKFSLAM(
        eta_0=eta_0,
        P_0=P_0,
        Q_vec=Q_vec,
        R_vec=np.flip(R_vec).T   , # flip to (range, bearing)
        big_var=1e6,
    )

    # Convert simulator measurements into EKF-friendly format:
    # pose -> list of (z_ij, landmark_idx), where z_ij = [range, bearing]
    from collections import defaultdict
    meas_by_pose = defaultdict(list)

    for pose_idx, meas_list in landmark_meas.items():
        lm_indices = associations[pose_idx]
        assert len(meas_list) == len(lm_indices), "Mismatch in measurements and associations."

        for (rng, bearing_rot), lm_idx in zip(meas_list, lm_indices):
            # Rot2 -> angle in radians
            bearing_angle = bearing_rot.theta()   # or .angle(), depending on GTSAM version
            z_ij = np.array([rng, bearing_angle]) # [range, bearing]
            meas_by_pose[pose_idx].append((z_ij, lm_idx))
    
    eta_hist = []
    P_hist = []

    # Optional: initial update at pose 0 if there are already landmark measurements
    ekf.update(meas_by_pose.get(0, []))
    eta_hist.append(ekf.eta.copy())
    P_hist.append(ekf.P.copy())

    current_pose_idx = 0

    for noisy_odom, from_idx, to_idx in odometry_meas:
        # Convert Pose2 odometry to body-frame increment u = [dx, dy, dtheta]
        # For Pose2 between(i, i+1), translation is expressed in pose_i frame (body frame).
        u_k = np.array([noisy_odom.x(), noisy_odom.y(), noisy_odom.theta()])

        # 1. Prediction
        ekf.predict(u_k)
        current_pose_idx = to_idx

        # 2. Update with all landmark measurements at this pose
        ekf.update(meas_by_pose.get(current_pose_idx, []))

        eta_hist.append(ekf.eta.copy())
        P_hist.append(ekf.P.copy())

    # --------------------------------------------------
    # EKF dead reckoning (prediction only, no updates)
    # --------------------------------------------------
    eta_hist_dr = []
    P_hist_dr = []

    ekf_dr = EKFSLAM(
        eta_0=eta_0,
        P_0=P_0,
        Q_vec=Q_vec,
        R_vec=np.flip(R_vec).T,  # (range, bearing)
        big_var=1e6,
    )

    eta_hist_dr.append(ekf_dr.eta.copy())
    P_hist_dr.append(ekf_dr.P.copy())

    for noisy_odom, from_idx, to_idx in odometry_meas:
        u_k = np.array([noisy_odom.x(), noisy_odom.y(), noisy_odom.theta()])
        ekf_dr.predict(u_k)
        eta_hist_dr.append(ekf_dr.eta.copy())
        P_hist_dr.append(ekf_dr.P.copy())

    # --------------------------------------------------
    # Build EKF multivariate normal parameter lists
    # --------------------------------------------------
    # EKF with landmarks (SLAM)
    ekf_poses_marginals = []
    for eta_k, P_k in zip(eta_hist, P_hist):
        mu = eta_k[:3]
        cov = P_k[:3, :3]
        pose_mean = gtsam.Pose2(mu[0], mu[1], mu[2])
        ekf_poses_marginals.append(MultivariateNormalParameters(pose_mean, cov))

    ekf_landmarks_marginals = []
    eta_final = eta_hist[-1]
    P_final = P_hist[-1]
    m_final = eta_final[3:].reshape(-1, 2)

    for j in range(num_landmarks):
        mx, my = m_final[j]
        mean_lm = gtsam.Point2(mx, my)
        cov_lm = P_final[3 + 2*j:3 + 2*j + 2, 3 + 2*j:3 + 2*j + 2]
        ekf_landmarks_marginals.append(MultivariateNormalParameters(mean_lm, cov_lm))

    # EKF dead reckoning (no landmarks)
    ekf_poses_marginals_dr = []
    for eta_k, P_k in zip(eta_hist_dr, P_hist_dr):
        mu = eta_k[:3]
        cov = P_k[:3, :3]
        pose_mean = gtsam.Pose2(mu[0], mu[1], mu[2])
        ekf_poses_marginals_dr.append(MultivariateNormalParameters(pose_mean, cov))

    # --------------------------------------------------
    # FACTOR GRAPH SLAM
    # --------------------------------------------------

    fgParams = NonlinearFactorGraphParams(
        Q_vec = Q_vec,
        R_vec = R_vec,
        P_x0_vec = P_x0_vec
    )

    exact_map = True

    # GTSAM with landmakrs
    nfg, initial_estimate = build_nonlinear_factor_graph(sim_data, fgParams)

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(nfg, initial_estimate, params)
    result = optimizer.optimize()

    marginals = gtsam.Marginals(nfg, result) # linearized factor graph at result point

    # Compare individual marginal covariances
    print("\nMarginal covariances with landmarks:")
    for i, key in enumerate(all_vars, start=1):
        cov = marginals.marginalCovariance(key)
        print(f"Cov[{kstr(key)}]:\n", np.round(cov, 4))

    poses_marginals = []
    for key in pose_vars:
        mean = result.atPose2(key)
        cov = marginals.marginalCovariance(key)
        poses_marginals.append(MultivariateNormalParameters(mean, cov))
    
    landmarks_marginals = []
    for key in landmark_vars:
        cov = marginals.marginalCovariance(key)
        mean = result.atPoint2(key)
        landmarks_marginals.append(MultivariateNormalParameters(mean, cov))
    
    fig, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
    
    plot_result(ax, poses_marginals, landmarks_marginals, sample_points=False, exact_map=exact_map)
    ax.plot([],[], 'or', label=r'$\hat{x}$')
    ax.plot([],[], 'ob', label=r'$\hat{m}$')

    # Overlay EKF SLAM results
    for i, pose in enumerate(ekf_poses_marginals):
        pose_mean = pose.mean.translation()
        cov = pose.covariance[0:2, 0:2]
        ax.plot(pose_mean[0], pose_mean[1], 'ok', markersize=5, label=r'EKF' if i == 0 else "")
        if exact_map:
            plot_se2_covariance_on_manifold_gtsam(ax, pose, fill_alpha=0.0, fill_color="k", linestyle='--')
        else:
            plot_ellipse(ax, MultivariateNormalParameters(pose_mean, cov), fill_alpha=0.0, fill_color="k", linestyle='--')

    
    for i, landmark in enumerate(ekf_landmarks_marginals):
        mean = landmark.mean
        cov = landmark.covariance
        ax.plot(mean[0], mean[1], 'ok', markersize=5)
        plot_ellipse(ax, MultivariateNormalParameters(mean, cov), fill_alpha=0.0, fill_color="k", linestyle='--')

    for i, pose in enumerate(poses_gt):
        pos = pose.translation()
        ax.plot(pos[0], pos[1], 'rx', markersize=10, label=r'$x_{GT}$' if i == 0 else "")
    
    for i, landmark in enumerate(landmarks_gt):
        ax.plot(landmark[0], landmark[1], 'bx', markersize=10, label=r'$m_{GT}$' if i == 0 else "")
    
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()

    # GTSAM without landmarks (dead reckoning)
    fgParams.dead_reckoning = True
    nfg2, initial_estimate2 = build_nonlinear_factor_graph(sim_data, fgParams)

    optimizer2 = gtsam.LevenbergMarquardtOptimizer(nfg2, initial_estimate2, params)
    result2 = optimizer2.optimize()

    marginals2 = gtsam.Marginals(nfg2, result2) # linearized factor graph at result point
    poses_marginals2 = []
    for key in pose_vars:
        mean = result2.atPose2(key)
        cov = marginals2.marginalCovariance(key)
        poses_marginals2.append(MultivariateNormalParameters(mean, cov))

    print("\nMarginal covariances without landmarks:")
    for key in pose_vars:
        cov = marginals2.marginalCovariance(key)
        print(f"Cov[{kstr(key)}]:\n", np.round(cov, 4))
    

    fig2, ax2 = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
  
    plot_result(ax2, poses_marginals2, [], sample_points=False, exact_map=exact_map)
    ax2.plot([],[], 'or', label=r'$\hat{x}$')

    # Overlay EKF dead reckoning results
    for i, pose in enumerate(ekf_poses_marginals_dr):
        pose_mean = pose.mean.translation()
        cov = pose.covariance[0:2, 0:2]
        ax2.plot(pose_mean[0], pose_mean[1], 'ok', markersize=5, label=r'EKF' if i == 0 else "")
        if exact_map:
            plot_se2_covariance_on_manifold_gtsam(ax2, pose, fill_alpha=0.0, fill_color="k", linestyle='--')
        else:
            plot_ellipse(ax2, MultivariateNormalParameters(pose_mean, cov), fill_alpha=0.0, fill_color="k", linestyle='--')

    for i, pose in enumerate(poses_gt):
        pos = pose.translation()
        ax2.plot(pos[0], pos[1], 'rx', markersize=10, label=r'$x_{GT}$' if i == 0 else "")

    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.legend()
    plt.show()

    # fig.savefig('figures/slam_nonlinear_with_landmarks.pdf', bbox_inches='tight')
    # fig2.savefig('figures/slam_nonlinear_without_landmarks.pdf', bbox_inches='tight')

if __name__ == "__main__":
    nonlinear_batch_slam_example_with_and_without_landamrks()
