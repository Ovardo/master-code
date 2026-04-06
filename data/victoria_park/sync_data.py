from scipy.io import loadmat
import numpy as np
from pathlib import Path



def sync_data():

    data_folder = Path(__file__).parents[1].joinpath("victoria_park")
    # Load .mat files
    vp_data = {
        **loadmat(str(data_folder.joinpath("matlab/aa3_dr"))),
        **loadmat(str(data_folder.joinpath("matlab/aa3_lsr2"))),
        **loadmat(str(data_folder.joinpath("matlab/aa3_gpsx"))),
    }

    # Laser data (lsr)
    Z_lsr = vp_data["LASER"] / 100 # (K_lsr, 361) convert from cm to m 
    T_lsr = vp_data["TLsr"].ravel() / 1000 # (K_lsr,) convert ms to s 

    # Odometry data (odo)
    V_odo = vp_data["speed"].ravel() # (K_odo,)
    A_odo = vp_data["steering"].ravel() # (K_odo,)
    T_odo = vp_data["time"].ravel()  / 1000  # (K_odo,) convert ms t

    # GPS data (gps)
    La_gps = vp_data["La_m"].ravel() # (K_gps,)
    Lo_gps = vp_data["Lo_m"].ravel() # (K_gps,)
    T_gps = vp_data["timeGps"].ravel()  / 1000. #  (K_gps,) convert ms to s 

    K_odo = T_odo.size
    K_lsr = T_lsr.size
    K_gps = T_gps.size

    # Synchronize data based on timestamps
    lsr = list()
    odo = list()
    gps = list()

    k_lsr = 1 # begin at 1 as the first laser measurement is a bit off
    t_lsr = T_lsr[k_lsr]
    t = T_odo[0] - 0.025 # to avoid zero dt for first iteration (assumed t_odo0 < t_lsr0)

    for k_odo in range(K_odo-1):
        
        t_odo = T_odo[k_odo]
        has_laser = t_lsr < T_odo[k_odo+1] and k_lsr < K_lsr - 1

        if has_laser:
            t = t_odo
            odo.append((t, V_odo[k_odo], A_odo[k_odo]))
            lsr.append((t, Z_lsr[k_lsr]))
            k_lsr += 1
            t_lsr = T_lsr[k_lsr]
        else:
            t = t_odo
            odo.append((t, V_odo[k_odo], A_odo[k_odo]))
            lsr.append(None)


    with open(data_folder.joinpath("synced/odo.txt"), "w") as f:
        for odo_entry in odo:
            t_odo, v_odo, a_odo = odo_entry
            f.write(f"{t_odo:.3f}, {v_odo}, {a_odo}\n")

    with open(data_folder.joinpath("synced/lsr.txt"), "w") as f:
        for lsr_entry in lsr:
            if lsr_entry is None:
                f.write("None\n")
            else:
                t_lsr, z_lsr = lsr_entry
                z_lsr_str = ", ".join(map(str, z_lsr))
                f.write(f"{t_lsr:.3f}, {z_lsr_str}\n")



if __name__ == "__main__":
    sync_data()


    
    



    
    
