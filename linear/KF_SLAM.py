import numpy as np

class KalmanFilterSlam:
    def __init__(self, eta_0, Px_0, Q, R, big_var=1e6):
        """
        eta_0 : initial mean, shape (2 + 2M,)
        Px_0  : initial covariance of pose (2x2)
        Q     : odometry noise covariance (2x2)
        R     : measurement noise covariance (2x2)
        big_var : large variance for unobserved landmarks (approx. "no prior")
        """
        self.eta = eta_0.copy()
        self.Q = Q
        self.R = R

        self.num_m = (len(eta_0) - 2) // 2
        n = len(eta_0)

        # Covariance
        self.P = np.zeros((n, n))
        self.P[:2, :2] = Px_0

        # Large variance for landmarks (no prior)
        for i in range(self.num_m):
            landmark_idx = 2 + 2 * i
            self.P[landmark_idx, landmark_idx] = big_var
            self.P[landmark_idx + 1, landmark_idx + 1] = big_var

    def predict(self, eta, P, u):
        """Prediction step: x_{k+1} = x_k + u, landmarks static."""
        eta_pred = eta.copy()
        P_pred = P.copy()

        # Pose update
        eta_pred[:2] += u
        # Covariance update (only pose block gets Q)
        P_pred[:2, :2] += self.Q

        return eta_pred, P_pred

    def update(self, eta, P, measurements_for_pose):
        """
        Update step using all landmark measurements available at the *current* pose.

        measurements_for_pose: list of (z_ij, landmark_idx), where
          - z_ij : (2,) noisy measurement m_j - x + noise
          - landmark_idx : int, 0-based index into the landmark list

        If no measurements, this is a no-op.
        """
        if len(measurements_for_pose) == 0:
            return eta, P

        x = eta[:2]                       # (2,)
        m = eta[2:].reshape(-1, 2)        # (M, 2)
        n = len(eta)
        K_meas = len(measurements_for_pose)

        # Measurement vector (stacked 2D measurements) and prediction
        z_meas = np.zeros(2 * K_meas)
        z_pred = np.zeros(2 * K_meas)

        # Jacobian H (2*K_meas x n)
        H = np.zeros((2 * K_meas, n))

        for i, (z_ij, landmark_idx) in enumerate(measurements_for_pose):
            r = slice(2 * i, 2 * i + 2)

            # Measurement model: z = m_j - x
            z_meas[r] = z_ij
            z_pred[r] = m[landmark_idx] - x

            # Jacobian w.r.t pose x: -I_2
            H[r, 0:2] = -np.eye(2)

            # Jacobian w.r.t landmark m_j: I_2 at the correct location
            col = 2 + 2 * landmark_idx
            H[r, col:col + 2] = np.eye(2)

        # Measurement covariance: block-diagonal of R
        R_big = np.kron(np.eye(K_meas), self.R)

        # Innovation covariance
        S = H @ P @ H.T + R_big

        # Kalman gain
        K_gain = P @ H.T @ np.linalg.inv(S)

        # Innovation
        y = z_meas - z_pred

        # Update
        eta_upd = eta + K_gain @ y
        I = np.eye(n)
        P_upd = (I - K_gain @ H) @ P

        return eta_upd, P_upd