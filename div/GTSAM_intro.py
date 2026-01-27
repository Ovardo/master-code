
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional
from functools import partial




def plot_incremental_trajectory_2d(fignum, values, keys, scale=1.0, marginals=None, time_interval=0.5):
    """
    Incrementally plot a 2D trajectory using Pose2s in `values`.

    Args:
        fignum: Integer representing the figure number to use for plotting.
        values: gtsam.Values containing the poses.
        keys: List of keys to plot.
        scale: Value to scale the poses by.
        marginals: Marginals object for uncertainty ellipses.
        time_interval: Time in seconds to pause between each rendering.
    """
    fig = plt.figure(fignum)
    axes = fig.gca()
    axes.clear()
    plt.grid('on')
    for key in keys:
        if values.exists(key):
            pose = values.atPose2(key)
            cov = marginals.marginalCovariance(key) if marginals is not None else None
            gtsam_plot.plot_pose2_on_axes(axes, pose, scale, cov)
            plt.draw()
            plt.pause(time_interval)
    axes.autoscale()

    plt.grid('on')
    plt.show()






if __name__ == "__main__":

    graph = gtsam.NonlinearFactorGraph()

    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 0.0), priorNoise))

    # Add odometry factors
    model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    graph.add(gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2.0, 0.0, 0.0), model))
    graph.add(gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2.0, 0.0, np.pi/2), model))
    graph.add(gtsam.BetweenFactorPose2(3, 4, gtsam.Pose2(2.0, 0.0, np.pi/2), model))
    graph.add(gtsam.BetweenFactorPose2(4, 5, gtsam.Pose2(2.0, 0.0, np.pi/2), model))

    # Add pose constraint
    graph.add(gtsam.BetweenFactorPose2(5, 2, gtsam.Pose2(2.0, 0.0, np.pi/2), model))


    initialEstimate = gtsam.Values()
    initialEstimate.insert(1, gtsam.Pose2(0.1, 0.2, 0.15))
    initialEstimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initialEstimate.insert(3, gtsam.Pose2(4.1, -0.1, 0.1))
    initialEstimate.insert(4, gtsam.Pose2(4.3, 1.6, 1.5))
    initialEstimate.insert(5, gtsam.Pose2(2.5, 2.2, 2.9))

    result = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate).optimize()
    #result.print()


    marginals = gtsam.Marginals(graph, result)
    #np.set_printoptions(precision=4, suppress=True)
    # marginals.print("Marginals:\n")
    # print("x1 covariance:\n" + str(marginals.marginalCovariance(1)))
    # print("x2 covariance:\n" + str(marginals.marginalCovariance(2)))
    # print("x3 covariance:\n" + str(marginals.marginalCovariance(3)))
    # print("x4 covariance:\n" + str(marginals.marginalCovariance(4)))
    # print("x5 covariance:\n" + str(marginals.marginalCovariance(5)))


    # Example usage:
    keys = [1, 2, 3, 4, 5]  # Change as needed for your scenario

    plot_incremental_trajectory_2d(1, result, keys, scale=0.4, marginals=marginals, time_interval=1)
