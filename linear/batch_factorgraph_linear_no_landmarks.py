import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
from utilities.plot_utils import plot_result, MultivariateNormalParameters
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

"""Based on the example https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/PlanarSLAMExample.py"""


def batch_factorgraph_example():
    # Create an empty nonlinear factor graph.
    graph = gtsam.NonlinearFactorGraph()

    # Create the keys for the poses.
    X1 = X(1)
    X2 = X(2)
    X3 = X(3)
    pose_variables = [X1, X2, X3]

    # Add a prior on pose X1 at the origin.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(gtsam.PriorFactorPose2(X1, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))

    # Add odometry factors between X1,X2 and X2,X3, respectively
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(gtsam.BetweenFactorPose2(
        X1, X2, gtsam.Pose2(2.0, 0.0, 0.0), odometry_noise))
    graph.add(gtsam.BetweenFactorPose2(
        X2, X3, gtsam.Pose2(2.0, 0.0, 0.0), odometry_noise))

    
    # Create (deliberately inaccurate) initial estimate
    initial_estimate = gtsam.Values()
    initial_estimate.insert(X1, gtsam.Pose2(-0.25, 0.20, 0.15))
    initial_estimate.insert(X2, gtsam.Pose2(2.30, 0.10, -0.20))
    initial_estimate.insert(X3, gtsam.Pose2(4.10, 0.10, 0.10))
 
    # Create an optimizer.
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Solve the MAP problem.
    result = optimizer.optimize()

    # Calculate marginal covariances for all variables.
    marginals = gtsam.Marginals(graph, result)

    # Extract marginals
    pose_marginals = []
    for var in pose_variables:
        pose_marginals.append(MultivariateNormalParameters(result.atPose2(var), marginals.marginalCovariance(var)))

    # You can extract the joint marginals like this.
    joint_all = marginals.jointMarginalCovariance(gtsam.KeyVector(pose_variables))
    print("Joint covariance over all variables:")
    print(joint_all.fullMatrix())

    # Plot the marginals.
    plot_result(pose_marginals, [])
    
    result.print("\nFinal Result:")
    
    print("Marginal covariances:")
    print("X1:\n", marginals.marginalCovariance(X1))
    print("X2:\n", marginals.marginalCovariance(X2))
    print("X3:\n", marginals.marginalCovariance(X3))
    
    gtsam_plot.plot_pose2(0, result.atPose2(X1), 0.5, marginals.marginalCovariance(X1))
    gtsam_plot.plot_pose2(0, result.atPose2(X2), 0.5, marginals.marginalCovariance(X2))
    gtsam_plot.plot_pose2(0, result.atPose2(X3), 0.5, marginals.marginalCovariance(X3))
    plt.axis('equal')
    plt.grid('on')
    plt.show()


if __name__ == "__main__":
    batch_factorgraph_example()
