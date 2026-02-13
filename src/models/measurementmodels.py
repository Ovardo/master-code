from dataclasses import dataclass

import numpy as np

from utils.utils_math import cartesian2polar, rotmat2, ssa


@dataclass
class RangeBearing:
    """Sensor model for range and bearing measurements to landmarks.

    Measurement vector: z = [range, bearing]
    """

    sigma_range: float
    sigma_bearing: float
    max_range: float = 7.5  # [m] (currently not used)
    max_fov: float = 2 * np.pi  # [rad] (currently not used)

    def h_(self, x: np.ndarray, m_i: np.ndarray) -> np.ndarray:
        """Calculate the measurement prediction function h_ for a single landmark.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m_i : np.ndarray, shape=(2,)
            the landmark position of landmark i

        Returns
        -------
        np.ndarray, shape=(2,)
            the predicted measurement.
        """
        R_w_b = rotmat2(x[2])
        delta_w = (
            m_i - x[:2]
        )  # TODO: add some handling if landmark is very close to robot position
        delta_b = R_w_b.T @ delta_w
        r, theta = cartesian2polar(delta_b[0], delta_b[1])
        z_i = np.array([r, ssa(theta)])
        return z_i

    def h(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Calculate the measurement prediction function h for multiple landmarks.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(#landmarks, 2)
            the stacked measurements.
        """
        z = np.zeros((m.shape[0], 2))

        for i in range(m.shape[0]):
            z[i, :] = self.h_(x, m[i, :])

        return z

    def H_x(self, x: np.ndarray, m_i: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of h_ with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m_i : np.ndarray, shape=(2,)
            the landmark position of landmark i

        Returns
        -------
        np.ndarray, shape=(2,3)
            The Jacobian of h_ wrt. x.
        """
        Hx = np.zeros((2, 3))

        delta_w = m_i - x[:2]
        r = np.linalg.norm(delta_w)

        if r < 1e-6:
            raise ValueError(
                "Jacobian undefined for landmark very close to robot position."
            )  # TODO: handle better

        Hx[0, :2] = -delta_w / r
        Hx[0, 2] = 0.0
        Hx[1, :2] = -(delta_w @ rotmat2(np.pi / 2).T) / (r**2)
        Hx[1, 2] = -1.0

        return Hx

    def H_m(self, x: np.ndarray, m_i: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of h_ with respect to m_i.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m_i : np.ndarray, shape=(2,)
            the landmark position of landmark i

        Returns
        -------
        np.ndarray, shape=(2,2)
            The Jacobian of h_ wrt. m_i.
        """
        Hx = self.H_x(x, m_i)
        Hm = -Hx[:, :2]

        return Hm

    def H(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h wrt. eta (x, m).

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta (x, m).
        """

        num_m = m.shape[0]

        Hx = np.zeros((2 * num_m, 3))
        Hm = np.zeros((2 * num_m, 2 * num_m))

        for i in range(num_m):
            Hx[2 * i : 2 * i + 2, :] = self.H_x(x, m[i, :])
            Hm[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = self.H_m(x, m[i, :])

        H = np.hstack((Hx, Hm))
        return H

    def R(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Calculate the measurement noise covariance matrix R.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 2 * #landmarks)
            The measurement noise covariance matrix R.

        """
        num_m = m.shape[0]
        R = np.diag(np.tile([self.sigma_range**2, self.sigma_bearing**2], num_m))
        return R

    def predict_measurement(self, x, P_pred, m):
        """Predict a measurement to a single landmark."""
        z = self.h(x, m)
        H = self.H(x, m)
        R = self.R(x, m)

        S = H @ P_pred @ H.T + R

        return z, S

    def predict_measurements(self, eta_pred, P_pred):
        """Predict measurements to all landmarks."""
        x_pred = eta_pred[:3]
        landmarks = eta_pred[3:].reshape(-1, 2)

        z_pred = self.h(x_pred[:3], landmarks.reshape(-1, 2))
        H = self.H(x_pred, landmarks)
        R = self.R(x_pred, landmarks)

        S = H @ P_pred @ H.T + R

        return z_pred, S

    def predict_measurement_covariance(self, x_pred, m_pred, P_pred):
        """Predict measurement covariance to all landmarks."""
        H = self.H(x_pred, m_pred)
        R = self.R(x_pred, m_pred)

        S = H @ P_pred @ H.T + R

        return S
