import numpy as np
from matplotlib import pyplot as plt

from data_loader import VictoriaParkLoader


def main():
    
    data_loader = VictoriaParkLoader()
    
    T_lsr = data_loader.lidar[:, 0]
    T_odo = data_loader.odometry[:, 2]

    lsr_dt = np.diff(T_lsr)
    odo_dt = np.diff(T_odo)

    avg_lsr_sample_time = np.mean(lsr_dt)
    avg_odo_sample_time = np.mean(odo_dt)

    avg_lsr_freq = 1.0 / avg_lsr_sample_time
    avg_odo_freq = 1.0 / avg_odo_sample_time

    print(f"Average lidar sample time: {avg_lsr_sample_time:.4f} s")
    print(f"Average odometry sample time: {avg_odo_sample_time:.4f} s")
    print(f"Average lidar frequency: {avg_lsr_freq:.2f} Hz")
    print(f"Average odometry frequency: {avg_odo_freq:.2f} Hz")

    t0 = min(T_lsr[0], T_odo[0])
    t1 = t0 + 1.0

    T_lsr = T_lsr[(T_lsr >= t0) & (T_lsr <= t1)] - t0
    T_odo = T_odo[(T_odo >= t0) & (T_odo <= t1)] - t0

    plt.figure(figsize=(10, 5))

    plt.vlines(T_lsr, 0, 1, color='r', label=f'Laser scan times ({avg_lsr_freq:.2f} Hz, {avg_lsr_sample_time*1000:.2f} ms)')
    plt.vlines(T_odo, 0, 0.5, color='b', label=f'Odometry times ({avg_odo_freq:.2f} Hz, {avg_odo_sample_time*1000:.2f} ms)')
    plt.title("Timing of Laser Scans and Odometry Measurements")
    plt.xlabel("Time (s)")
    plt.ylim(0, 1.5)
    plt.yticks([])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()