import numpy as np
from models.dynamicmodels import OdometrySE2
from models.measurementmodels import RangeBearing
from utils.utils_math import rotmat2, ssa


class EKFSLAM:
    def __init__(self, eta_0, P_0, Q_vec, R_vec, big_var=1e6):
        """
        Extended Kalman Filter for SLAM in SE(2) with range-bearing measurements.

        Parameters
        ----------
        eta_0 : np.ndarray, shape (3 + 2M,)
            Initial state mean [x, y, theta, m0x, m0y, m1x, m1y, ...].
        P_0 : np.ndarray, shape (3 + 2M, 3 + 2M)
            Initial covariance matrix (pose + landmarks).
            Typically top-left 3x3 = diag(P_x0_vec**2) and landmarks large variance.
        Q_vec : np.ndarray, shape (3,)
            [sigma_x, sigma_y, sigma_theta] for odometry noise (body frame).
        R_vec : np.ndarray, shape (2,)
            [sigma_range, sigma_bearing] note GTSAM uses (bearing, range) ordering.
        big_var : float
            Large variance to initialize unknown landmark covariances.

        """
        self.eta = eta_0.copy()
        self.P = P_0.copy()

        self.num_m = (len(eta_0) - 3) // 2
        self.n = len(eta_0)

        # Motion model
        sigma_x, sigma_y, sigma_theta = Q_vec
        self.motion_model = OdometrySE2(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_theta=sigma_theta,
        )

        # Measurement model:
        sigma_range, sigma_bearing = R_vec
        self.measurement_model = RangeBearing(
            sigma_range=sigma_range,
            sigma_bearing=sigma_bearing,
        )

        # Initialize landmark covariances to big_var if not already set
        for i in range(self.num_m):
            idx = 3 + 2 * i
            if self.P[idx, idx] == 0.0 and self.P[idx + 1, idx + 1] == 0.0:
                self.P[idx, idx] = big_var
                self.P[idx + 1, idx + 1] = big_var

        self.initialized_landmarks = np.zeros(self.num_m, dtype=bool)


    # ------------------------------------------------------------------
    # Prediction step (full state & covariance)
    # ------------------------------------------------------------------
    def predict(self, u):
        """
        EKF prediction step.

        Parameters
        ----------
        u : np.ndarray, shape (3,)
            Odometry increment in body frame: [dx_b, dy_b, dtheta].

        Returns
        -------
        eta_pred, P_pred : np.ndarray
            Predicted mean and covariance (also stored in self.eta, self.P).

        """
        x_prev = self.eta[:3]
        m_prev = self.eta[3:].reshape(-1, 2)

        # Predicted state
        x_pred = self.motion_model.f_x(x_prev, u)
        x_pred[2] = ssa(x_pred[2])
        m_pred = m_prev.copy()
        eta_pred = np.hstack((x_pred, m_pred.flatten()))

        # Jacobian of full state transition wrt eta
        F_big = self.motion_model.F(x_prev, m_prev, u)  # (3+2M, 3+2M)

        # Process noise in pose, embedded into full state
        Q_pose = self.motion_model.Q(x_prev, u)         # (3x3)
        Q_big = np.zeros_like(self.P)
        Q_big[:3, :3] = Q_pose

        P_pred = F_big @ self.P @ F_big.T + Q_big

        # Store and return
        self.eta = eta_pred
        self.P = P_pred
        return self.eta, self.P

    # ------------------------------------------------------------------
    # Update step with subset of landmark measurements
    # ------------------------------------------------------------------
    def update(self, measurements_for_pose):
        """
        EKF update step using all landmark measurements at the current pose.

        measurements_for_pose : list of (z_ij, landmark_idx)
            z_ij         : np.ndarray, shape (2,), [range, bearing]
            landmark_idx : int, 0-based index in the map

        Returns
        -------
        eta_upd, P_upd : updated mean and covariance (also stored in self.eta, self.P)

        """
        if len(measurements_for_pose) == 0:
            return self.eta, self.P

        # 1) Initialize any uninitialized landmarks, and only use
        #    already-initialized ones for the actual EKF update
        used_measurements = []
        for z_ij, lm_idx in measurements_for_pose:
            if not self.initialized_landmarks[lm_idx]:
                # First observation: use it only to initialize the landmark
                self.initialize_landmark(lm_idx, z_ij)
                # Skip adding this measurement to used_measurements
            else:
                used_measurements.append((z_ij, lm_idx))

        # If no initialized landmarks yet, nothing to update with
        if len(used_measurements) == 0:
            return self.eta, self.P

        x = self.eta[:3]
        m_all = self.eta[3:].reshape(-1, 2)
        n = self.n
        K = len(used_measurements)

        z_meas = np.zeros(2 * K)
        z_pred = np.zeros(2 * K)
        H = np.zeros((2 * K, n))

        for i, (z_ij, lm_idx) in enumerate(used_measurements):
            r = slice(2 * i, 2 * i + 2)

            z_meas[r] = z_ij
            m_i = m_all[lm_idx]
            z_pred_i = self.measurement_model.h_(x, m_i)  # [range, bearing]
            z_pred[r] = z_pred_i

            Hx_i = self.measurement_model.H_x(x, m_i)     # (2x3)
            Hm_i = self.measurement_model.H_m(x, m_i)     # (2x2)

            H[r, 0:3] = Hx_i
            col = 3 + 2 * lm_idx
            H[r, col:col+2] = Hm_i

        # Measurement noise covariance (block-diag)
        Sigma_z_single = np.diag([
            self.measurement_model.sigma_range**2,
            self.measurement_model.sigma_bearing**2
        ])
        R_big = np.kron(np.eye(K), Sigma_z_single)

        # Innovation covariance
        S = H @ self.P @ H.T + R_big

        # Kalman gain
        K_gain = self.P @ H.T @ np.linalg.inv(S)

        # Innovation with bearing wrapping
        y = z_meas - z_pred
        for i in range(K):
            # bearing is second entry of each [range, bearing] pair
            y[2*i + 1] = ssa(y[2*i + 1])

        # State update
        eta_upd = self.eta + K_gain @ y
        eta_upd[2] = ssa(eta_upd[2])

        # Covariance update
        I = np.eye(n)
        P_upd = (I - K_gain @ H) @ self.P

        self.eta = eta_upd
        self.P = P_upd
        return self.eta, self.P


    def initialize_landmark(self, lm_idx, z_ij):
        r, b = z_ij
        x, y, theta = self.eta[:3]

        # Compute mean (same as before)
        R_w_b = rotmat2(theta)
        delta_b = np.array([r*np.cos(b), r*np.sin(b)])
        m_world = np.array([x, y]) + R_w_b @ delta_b

        # Write mean into state
        idx = 3 + 2*lm_idx
        self.eta[idx:idx+2] = m_world

        # -------------------------------
        # Initialize covariance P_m
        # -------------------------------
        # Pose covariance
        P_x = self.P[:3,:3]

        # Jacobian wrt pose x = [x, y, theta]
        J_x = np.array([
            [1, 0, -r*np.sin(theta + b)],
            [0, 1,  r*np.cos(theta + b)]
        ])

        # Measurement covariance R for one landmark
        R_z = np.diag([
            self.measurement_model.sigma_range**2,
            self.measurement_model.sigma_bearing**2
        ])

        # Jacobian wrt measurement z = [range, bearing]
        J_z = np.array([
            [np.cos(theta + b), -r * np.sin(theta + b)],
            [np.sin(theta + b),  r * np.cos(theta + b)]
        ])

        P_m = J_x @ P_x @ J_x.T + J_z @ R_z @ J_z.T # TODO; do the same for FG

        # Write into global covariance
        self.P[idx:idx+2, idx:idx+2] = P_m

        self.initialized_landmarks[lm_idx] = True
