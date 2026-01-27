import gtsam
import numpy as np
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import X, L
import gtsam.utils.plot as gtsam_plot
from utilities.utils import kstr
from data.data_generator_linear import SimulationConfig, RobotSimulatorR2, build_linear_factor_graph
from collections import defaultdict
from KF_SLAM import KalmanFilterSlam
from utilities.plot_utils import plot_result, MultivariateNormalParameters



def linear_batch_slam_example_with_and_without_landamrks():
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
        observations={  # subset visibility
            0: [0],     # X1 sees L1
            1: [0],     # X2 sees L1
            2: [1]      # X3 sees L2
        },
        prior_noise_sim=np.array([0, 0]),
        odometry_noise_sim=np.array([0.1, 0.1]),
        measurement_noise_sim=np.array([0.1, 0.1])
    )

    simulator = RobotSimulatorR2(config)
    sim_data = simulator.simulate()

    pose_vars = [X(i+1) for i in range(len(sim_data['ground_truth']['poses']))]
    landmark_vars = [L(i+1) for i in range(len(sim_data['ground_truth']['landmarks']))]
    all_vars = pose_vars + landmark_vars

    poses_gt = sim_data['ground_truth']['poses']
    landmarks_gt = sim_data['ground_truth']['landmarks']

    # Inference parameters
    Px_0 = np.diag([0.1, 0.1])  # initial pose covariance
    x_0 = np.array([0.0, 0.0])  # initial pose mean
    Q = np.diag([0.2, 0.2])     # odometry noise covariance
    R = np.diag([0.05, 0.05])   # measurement noise covariance
    eta_0 = np.array(x_0.tolist() + [0, 0] * len(landmarks_gt))

    # Kalman filter for verification purposes
    KF = KalmanFilterSlam(
        eta_0=eta_0,
        Px_0=Px_0**2,
        Q=Q**2,
        R=R**2
    )

    odometry_meas = sim_data['measurements']['odometry']
    landmark_meas = sim_data['measurements']['landmarks']

    # Group landmark measurements by pose index
    meas_by_pose = defaultdict(list)
    for z_ij, pose_idx, landmark_idx in landmark_meas:
        meas_by_pose[pose_idx].append((z_ij, landmark_idx))

    # =============================================================
    # A) KF-SLAM WITH LANDMARK MEASUREMENTS
    # =============================================================
    eta_kf = KF.eta.copy()
    P_kf = KF.P.copy()

    eta_hist_meas = []
    P_hist_meas = []

    # Update at initial pose X1 (index 0) if measurements exist
    eta_kf, P_kf = KF.update(eta_kf, P_kf, meas_by_pose.get(0, []))
    eta_hist_meas.append(eta_kf.copy())
    P_hist_meas.append(P_kf.copy())

    current_pose_idx = 0
    for u_k, from_idx, to_idx in odometry_meas:
        # Prediction
        eta_kf, P_kf = KF.predict(eta_kf, P_kf, u_k)
        current_pose_idx = to_idx

        # Update with all measurements from that pose
        eta_kf, P_kf = KF.update(eta_kf, P_kf, meas_by_pose.get(current_pose_idx, []))

        eta_hist_meas.append(eta_kf.copy())
        P_hist_meas.append(P_kf.copy())

    # =============================================================
    # B) KF-SLAM WITHOUT LANDMARK MEASUREMENTS (DEAD RECKONING)
    # =============================================================
    eta_dr = eta_0.copy()
    P_dr = KF.P.copy()

    eta_hist_dr = [eta_dr.copy()]
    P_hist_dr = [P_dr.copy()]

    current_pose_idx = 0
    for u_k, from_idx, to_idx in odometry_meas:
        eta_dr, P_dr = KF.predict(eta_dr, P_dr, u_k)
        current_pose_idx = to_idx

        eta_hist_dr.append(eta_dr.copy())
        P_hist_dr.append(P_dr.copy())

    # -------------------------------------------------------------
    # 1. GTSAM: WITH LANDMARKS
    # -------------------------------------------------------------
    gfg1 = build_linear_factor_graph(
        sim_data,
        prior_noise_fg=Px_0.diagonal(),
        odometry_noise_fg=Q.diagonal(),
        measurement_noise_fg=R.diagonal(),
    )

    solution1 = gfg1.optimize()
    marginals1 = gtsam.Marginals(gfg1, solution1)

    print("1. GTSAM marginal covariances WITH landmark-measurements")
    for key in all_vars:
        cov = marginals1.marginalCovariance(key)
        print(f"Cov[{kstr(key)}]:")
        print(cov)

    # KF pose covariances WITH measurements
    print("\n1. KF pose covariances WITH landmark-measurements")
    for i, (eta_kf_i, P_kf_i) in enumerate(zip(eta_hist_meas, P_hist_meas), start=1):
        print(f"KF Cov pose X{i}:")
        print(P_kf_i[:2, :2])

    poses_marginals1 = []
    for key in pose_vars:
        mean = solution1.at(key)
        cov = marginals1.marginalCovariance(key)
        poses_marginals1.append(MultivariateNormalParameters(mean, cov))
    
    landmarks_marginals1 = []
    for key in landmark_vars:
        cov = marginals1.marginalCovariance(key)
        mean = solution1.at(key)
        landmarks_marginals1.append(MultivariateNormalParameters(mean, cov))
    

    # Plot GTSAM + KF + ground truth
    fig1, ax1 = plt.subplots(figsize=(4, 3))

    # Plot GTSAM estimates with covariances
    # for key in all_vars:
    #     mean = solution1.at(key)
    #     cov = marginals1.marginalCovariance(key)

    #     if kstr(key)[0] == 'X':
    #         # GTSAM poses: red
    #         gtsam_plot.plot_point2_on_axes(ax1, point=mean, linespec='r', P=cov)
    #     elif kstr(key)[0] == 'L':
    #         # GTSAM landmarks: blue
    #         gtsam_plot.plot_point2_on_axes(ax1, point=mean, linespec='b', P=cov)
    plot_result(ax1, poses_marginals1, landmarks_marginals1, sample_points=False, exact_map=False )

    # # Plot KF estimates (poses only; landmarks in green if you want)
    # for i, (eta_kf_i, P_kf_i) in enumerate(zip(eta_hist_meas, P_hist_meas), start=1):
    #     pose_mean_kf = eta_kf_i[:2]
    #     pose_cov_kf = P_kf_i[:2, :2]
    #     # KF pose: black circles
    #     gtsam_plot.plot_point2_on_axes(ax1, point=pose_mean_kf, linespec='ko', P=pose_cov_kf)

    # # Optionally KF landmarks (final estimate)
    # m_kf_final = eta_hist_meas[-1][2:].reshape(-1, 2)
    # P_kf_final = P_hist_meas[-1]
    # for j, m_j in enumerate(m_kf_final):
    #     cov_mj = P_kf_final[2+2*j:2+2*j+2, 2+2*j:2+2*j+2]
    #     gtsam_plot.plot_point2_on_axes(ax1, point=m_j, linespec='go', P=cov_mj)

    # Ground truth
    for i, pose in enumerate(poses_gt):
        ax1.plot(pose[0], pose[1], 'rx', markersize=8, label=r'$x_{GT}$' if i == 0 else "")
    for i, landmark in enumerate(landmarks_gt):
        ax1.plot(landmark[0], landmark[1], 'bx', markersize=8, label=r'$m_{GT}$' if i == 0 else "")

    # Phantom points for legend
    ax1.plot([], [], 'ro', label=r'$\hat{x}$')
    ax1.plot([], [], 'bo', label=r'$\hat{m}$')
    # ax1.plot([], [], 'ko', label='KF pose')
    # ax1.plot([], [], 'go', label='KF landmark')

    # Axis limits based on all estimates
    x_coords = []
    y_coords = []
    for key in all_vars:
        point = solution1.at(key)
        x_coords.append(point[0])
        y_coords.append(point[1])
    for pose in poses_gt:
        x_coords.append(pose[0])
        y_coords.append(pose[1])

    margin = 1.0
    xlim = (min(x_coords) - margin, max(x_coords) + margin)
    ylim = (min(y_coords) - margin, max(y_coords) + margin)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.grid(True)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_aspect('equal', 'box')
    ax1.legend()
    fig1.tight_layout()

    # -------------------------------------------------------------
    # 2. GTSAM: WITHOUT LANDMARKS (dead reckoning)
    # -------------------------------------------------------------
    gfg2 = build_linear_factor_graph(
        sim_data,
        prior_noise_fg=Px_0.diagonal(),
        odometry_noise_fg=Q.diagonal(),
        measurement_noise_fg=R.diagonal(),
        dead_reckoning=True
    )

    solution2 = gfg2.optimize()
    marginals2 = gtsam.Marginals(gfg2, solution2)

    poses_marginals2 = []
    for key in pose_vars:
        mean = solution2.at(key)
        cov = marginals2.marginalCovariance(key)
        poses_marginals2.append(MultivariateNormalParameters(mean, cov))
    

    print("\n2. GTSAM marginal covariances WITHOUT landmark-measurements")
    for key in pose_vars:
        cov = marginals2.marginalCovariance(key)
        print(f"Cov[{kstr(key)}]:")
        print(cov)

    # KF pose covariances WITHOUT measurements
    print("\n2. KF pose covariances WITHOUT landmark-measurements (dead reckoning)")
    for i, (eta_dr_i, P_dr_i) in enumerate(zip(eta_hist_dr, P_hist_dr), start=1):
        print(f"KF Cov pose X{i}:")
        print(P_dr_i[:2, :2])

    # Plot dead reckoning: GTSAM + KF + GT
    fig2, ax2 = plt.subplots(figsize=(4, 3))

    # GTSAM dead-reckoning poses
    plot_result(ax2, poses_marginals2, [], sample_points=False, exact_map=False )
    # for key in pose_vars:
    #     mean = solution2.at(key)
    #     cov = marginals2.marginalCovariance(key)
    #     gtsam_plot.plot_point2_on_axes(ax2, point=mean, linespec='r', P=cov)

    # # KF dead-reckoning poses
    # for i, (eta_dr_i, P_dr_i) in enumerate(zip(eta_hist_dr, P_hist_dr), start=1):
    #     pose_mean_kf = eta_dr_i[:2]
    #     pose_cov_kf = P_dr_i[:2, :2]
    #     gtsam_plot.plot_point2_on_axes(ax2, point=pose_mean_kf, linespec='ko', P=pose_cov_kf)

    # Ground truth poses
    for i, pose in enumerate(poses_gt):
        ax2.plot(pose[0], pose[1], 'rx', markersize=8, label=r'$x_{GT}$' if i == 0 else "")

    # Legends
    ax2.plot([], [], 'ro', label=r'$\hat{x}$')
    # ax2.plot([], [], 'ko', label='KF pose')

    ax2.grid(True)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal', 'box')
    ax2.legend()
    fig2.tight_layout()

    plt.show()

    # Save figures as PDF
    # fig1.savefig('figures/slam_linear_with_landmarks.pdf', bbox_inches='tight')
    # fig2.savefig('figures/slam_linear_without_landmarks.pdf', bbox_inches='tight')


if __name__ == "__main__":
    linear_batch_slam_example_with_and_without_landamrks()
