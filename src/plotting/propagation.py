import gtsam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

def confidence_ellipse_2d(ax, 
                          pose: gtsam.Pose2,
                          cov: np.ndarray,
                          scale: float = 1, 
                          **kwargs) -> None:
    """Draw a 2-D confidence ellipse for a 2×2 covariance matrix."""
    k = 2.447746830681 # 95% confidence interval for 2 DOF

    eigvals, eigvecs = np.linalg.eigh(cov[:2,:2]) # only x and y
    eigvals = np.maximum(eigvals, 0.0)  # clamp floating-point negatives
    angle   = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width   = np.sqrt(eigvals[0]) * 2 * k * scale
    height  = np.sqrt(eigvals[1]) * 2 * k * scale

    ellipse = Ellipse(xy=tuple(pose.translation()),
                      width=width,
                      height=height,
                      angle=np.degrees(angle),
                      **kwargs)
    
    ax.add_patch(ellipse)


def main():
    u = 1.0 # m/s
    v = 0.0 # m/s
    w = -0.2 # rad/s
    dt = 0.5 # s

    twist = np.array([u, v, w])
    sigma = np.diag([0.1, 0.1, 0.1])**2 # std dev for x, y, theta
    
    pose0 = gtsam.Pose2(1,1,np.pi/2)
    pose = pose0
    cov = np.eye(3) * 0.01

    poses = [pose]
    covs = [cov]

    for i in range(10):
        odo = gtsam.Pose2.Expmap(twist*dt)
        
        H1 = np.zeros((3, 3), order="F")
        H2 = np.zeros((3, 3), order="F")
        
        pose = pose.compose(odo, H1, H2)
        poses.append(pose)
        
        cov = H1 @ cov @ H1.T + H2 @ sigma @ H2.T
        covs.append(cov)

    # twist = np.array([u, 0, -w])

    # for i in range(20):
    #     odo = gtsam.Pose2.Expmap(twist*dt)
        
    #     H1 = np.zeros((3, 3), order="F")
    #     H2 = np.zeros((3, 3), order="F")
        
    #     pose = pose.compose(odo, H1, H2)
    #     poses.append(pose)
        
    #     cov = H1 @ cov @ H1.T + H2 @ sigma @ H2.T
    #     covs.append(cov)
    
    # twist = np.array([v, 0, w])

    # for i in range(20):
    #     odo = gtsam.Pose2.Expmap(twist*dt)
        
    #     H1 = np.zeros((3, 3), order="F")
    #     H2 = np.zeros((3, 3), order="F")
        
    #     pose = pose.compose(odo, H1, H2)
    #     poses.append(pose)
        
    #     cov = H1 @ cov @ H1.T + H2 @ sigma @ H2.T
    #     covs.append(cov)

    fig, ax = plt.subplots()
    ax.plot([p.x() for p in poses], [p.y() for p in poses], marker='o')
    for pose, cov in zip(poses, covs):
        
        # distribution of body frame wrt to the world frame
        R = pose.rotation().matrix()
        cov[:2,:2] = R @ cov[:2, :2] @ R.T
        
        confidence_ellipse_2d(
            ax,
            pose=pose,
            cov=cov,
            edgecolor='tab:red',
            facecolor='none',
            alpha=0.7,
            label='Rotation' if pose == poses[0] else '',
        )
        
        # distribution of invere, i.e uncertainty of the world frame wrt to the body frame
        pose_rel = pose0.inverse().compose(pose)
        Adj = pose_rel.AdjointMap() 
        cov = Adj @ cov @ Adj.T
        
        confidence_ellipse_2d(
            ax,
            pose=pose0,
            cov=cov,
            edgecolor='tab:blue',
            facecolor='none',
            alpha=0.7,
            label='Adjoint' if pose == poses[0] else '',
        )


    ax.set_title("Pose2 Expmap Integration")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid()
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()