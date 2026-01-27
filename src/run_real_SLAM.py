# Addapted from Odin Aleksander Severinsen Graded Assigment 2 code in TTK4250
import numpy as np
import matplotlib.pyplot as plt
import gtsam 
from gtsam.utils import plot as gtsam_plot

from scipy.io import loadmat
from pathlib import Path

from tqdm import tqdm

from utils.utils_victoria_park import detectTrees, odometry, Car

def main():
    # %% Load data
    victoria_park_foler = Path(__file__).parents[1].joinpath("data/victoria_park")
    realSLAM_ws = {
        **loadmat(str(victoria_park_foler.joinpath("aa3_dr"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_lsr2"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_gpsx"))),
    }

    timeOdo = (realSLAM_ws["time"] / 1000).ravel()
    timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
    timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

    steering = realSLAM_ws["steering"].ravel()
    speed = realSLAM_ws["speed"].ravel()
    LASER = (
        realSLAM_ws["LASER"] / 100
    )  # Divide by 100 to be compatible with Python implementation of detectTrees
    La_m = realSLAM_ws["La_m"].ravel()
    Lo_m = realSLAM_ws["Lo_m"].ravel()

    K = timeOdo.size
    mK = timeLsr.size
    Kgps = timeGps.size

    # %% Parameters

    L = 2.83  # axel distance
    H = 0.76  # center to wheel encoder
    a = 0.95  # laser distance in front of first axel
    b = 0.5  # laser distance to the left of center

    car = Car(L, H, a, b)

    #sigmas = np.array([0.0001, 0.001, 0.1 * np.pi / 180])  # TODO tune
    # CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    #Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas) 
    Q = np.diag([0.1, 0.1, 0.5 * np.pi / 180]) ** 2  # (x, y, theta)
    R = np.diag([2 * np.pi / 180,  0.05]) ** 2  # (bearing, range)

    alpha_individual = 0.999
    alpha_joint = 0.99999

    sensorOffset = np.array([car.a + car.L, car.b])

    x0 = np.array([Lo_m[0], La_m[0], 36 * np.pi / 180])
    P_x0 = np.array([0.05,0.05,np.deg2rad(0.5)]) 

    mk_first = 1  # first seems to be a bit off in timing
    mk = mk_first
    t = timeOdo[0]

    # %% Initialize SLAM
    from factor_graph_slam import FactorGraphSLAM, SLAMVisualizer
    from div.tuning import NonlinearFactorGraphParams

    
    fgParams = NonlinearFactorGraphParams(
        Q_vec=np.sqrt(np.diag(Q)),
        R_vec=np.sqrt(np.diag(R)),
        P_x0_vec=P_x0,
        association_type="jcbb",
        alpha_individual=alpha_individual,
        alpha_joint=alpha_joint,
        r_local=90.0, # Victoria park max range is around 80m
        use_isam=True,
    )

    slam = FactorGraphSLAM(fgParams, gtsam.Pose2(*x0))
    slam.current_step = 1 # quick fix as we do nt assume measurement at step 0

    Delta = gtsam.Pose2()  # accumulated odometry between laser measurements

    # %% Run SLAM (dead reckoning for odometry only)
    N = 1000

    poses_dead_reckoning = []
    x_prev = gtsam.Pose2(*x0)
    poses_dead_reckoning.append(x_prev)
    # Deac reckoning
    for k in range(1, N): 
        dt = timeOdo[k + 1] - t
        t = timeOdo[k + 1]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        x_new = x_prev.compose(gtsam.Pose2(*odo))
        x_prev = x_new
        poses_dead_reckoning.append(x_new)
    

    fig, ax = plt.subplots(figsize=(13, 8))
    
    x_coords = [pose.x() for pose in poses_dead_reckoning]
    y_coords = [pose.y() for pose in poses_dead_reckoning]
    ax.plot(x_coords, y_coords, 'k-', alpha=0.7, label=r'$\hat{x}_{DR}$'),
    
        
    # %% Run SLAM
    k_z = 0
    for k in tqdm(range(1, N)):
        
        if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
            k_z += 1

            dt = timeLsr[mk] - t
            if dt < 0:  # avoid assertions as they can be optimized avay?
                raise ValueError("negative time increment")

            # ? reset time to this laser time for next post predict
            t = timeLsr[mk]
            
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            Delta = Delta.compose(gtsam.Pose2(*odo)) 
            
            z = detectTrees(LASER[mk])
            
            meas = []
            for z_j in z:
                meas.append((z_j[0], gtsam.Rot2(z_j[1])))  # (range, bearing)

            slam.process_step(Delta, meas)
            Delta = gtsam.Pose2()  # reset odometry

            mk += 1

        if k < K - 1:
            dt = timeOdo[k + 1] - t
            t = timeOdo[k + 1]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            Delta = Delta.compose(gtsam.Pose2(*odo)) # TODO: scale odoemtry noise accordingly
            #slam.process_step(gtsam.Pose2(*odo), [])

    
    
    # %% Visualize final result
    marginals = slam.get_marginals()
    fig, ax = SLAMVisualizer.plot_final_result(slam, marginals, poses_dead_reckoning=poses_dead_reckoning, ax=ax)
    fig.savefig("figures/jcbb_vp.pdf", bbox_inches="tight")
    plt.show()
    
    # %%
    # SLAMVisualizer.plot_measurement_space(slam, step=70)
    # plt.show()

    # %%
    fig, ax = SLAMVisualizer.plot_NIS(slam)
    fig.savefig("figures/jcbb_vp_nis.pdf", bbox_inches="tight")
    plt.show()

    # %%
    # SLAMVisualizer.plot_step_by_step(slam)
    # plt.show()

    # # %%
    # SLAMVisualizer.plot_measurement_space_step_by_step(slam)
    # plt.show()

# %%

    # %% 

if __name__ == "__main__":
    main()



