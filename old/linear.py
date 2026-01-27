import gtsam
import numpy as np
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import X, L
import gtsam.utils.plot as gtsam_plot
from utilities.utils import kstr
from data.data_generator_linear import SimulationConfig, RobotSimulatorR2, build_linear_factor_graph

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
    prior_noise_sim=np.array([0, 0]),
    odometry_noise_sim=np.array([0, 0]),
    measurement_noise_sim=np.array([0, 0])
)

simulator = RobotSimulatorR2(config)
sim_data = simulator.simulate()

gfg = build_linear_factor_graph(
    sim_data,
    prior_noise_fg=np.array([0.05, 0.05]), # use true noise for now
    odometry_noise_fg=np.array([0.1, 0.1]),  
    measurement_noise_fg=np.array([0.1, 0.1])
)

pose_vars = [X(i+1) for i in range(len(sim_data['ground_truth']['poses']))]
landmark_vars = [L(i+1) for i in range(len(sim_data['ground_truth']['landmarks']))]
all_vars = pose_vars + landmark_vars

solution = gfg.optimize()
print(f"solution = {solution} with error {gfg.error(solution)}")

# # Convert to Gaussian Bayes Net
# gbn = gfg.eliminateSequential()
# #show(gbn)

# R, d = gbn.matrix() # Suquare root information matrix (R) and vector (d)
# information_matrix = R.T @ R
# covariance_matrix = np.linalg.inv(R.T @ R) 

# Marginals object
marginals = gtsam.Marginals(gfg, solution)

# Compute global covariance from Hessian
information_matrix, _ = gfg.hessian()
P_information = np.linalg.inv(information_matrix)
joint = marginals.jointMarginalCovariance(all_vars)
joint_cov = joint.fullMatrix()




# Compare individual pose covariances
for i, key in enumerate(all_vars, start=1):
    cov_from_joint = joint.at(key, key)
    cov_from_marg = marginals.marginalCovariance(key)

    print(f"{kstr(key)}:")
    print("From full joint cov matrix:\n", cov_from_joint)
    print("From marginals:\n", cov_from_marg)
    print("Equal? ->", np.allclose(cov_from_joint, cov_from_marg, atol=1e-10))
    print("-"*50)

# Compare joint covariance (all poses together)
joint_cov = marginals.jointMarginalCovariance(all_vars).fullMatrix()
print("Joint covariance (marginals) vs. full inverse match? ->",
      np.allclose(joint_cov, P_information, atol=1e-8))
np.set_printoptions(precision=3, suppress=True)

fig, axs = plt.subplots(2, 1, figsize=(22, 6))

for ax, title, cov_source in zip(axs,
    ["From Information Matrix (full inversion)", "From Marginals API (per-variable)"],
    ["joint", "marginals"]
):

    ax.set_aspect('equal')
    ax.set_title(title)

    for i, var in enumerate(all_vars, start=1):
        mean = solution.at(var)

        if cov_source == "joint":
            cov = joint.at(var, var)
        elif cov_source == "marginals":
            cov = marginals.marginalCovariance(var)

        if kstr(var)[0] == 'X':
            gtsam_plot.plot_point2_on_axes(ax, point=mean, linespec='r', P=cov)
        elif kstr(var)[0] == 'L':
            gtsam_plot.plot_point2_on_axes(ax, point=mean, linespec='b', P=cov)

plt.tight_layout()
plt.show()