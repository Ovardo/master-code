from dataclasses import dataclass

import numpy as np

from utils.utils_math import rotmat2, ssa


@dataclass
class OdometrySE2:
    """Dynamic model for robot in SE(2) using odometry as control input.

    TODO: State vector: x = [x, y, theta]
    Control vector: u = [dx, dy, dtheta] 
    """
    sigma_x: float
    sigma_y: float
    sigma_theta: float

    def f_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the motion prediction function f_x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the control input / odometry 

        Returns
        -------
        np.ndarray, shape=(3,)
            the predicted pose.
        """
        R_w_b = rotmat2(x[2])

        x_pred = np.zeros_like(x)
        x_pred[:2] = x[:2] + R_w_b @ u[:2]
        x_pred[2] = ssa(x[2] + u[2])

        return x_pred

    def f(self, x, m, u):
        """Calculate the motion prediction function f for multiple landmarks.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.
        u : np.ndarray, shape=(3,)
            the control input / odometry

        Returns
        -------
        np.ndarray, shape=(3 + 2 * #landmarks,)
            the predicted state (robot pose and landmarks).
        """
        num_m = m.shape[0]

        eta_pred = np.zeros((3 + 2 * num_m,))

        eta_pred[:3] = self.f_x(x, u)
        eta_pred[3:] = m.flatten()

        return eta_pred


    def F_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f_x with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the control input / odometry

        Returns
        -------
        np.ndarray, shape=(3,3)
            The Jacobian of f_x wrt. x.
        """
        Fx = np.eye(3)

        Fx[0,2] = -np.sin(x[2]) * u[0] - np.cos(x[2]) * u[1]
        Fx[1,2] =  np.cos(x[2]) * u[0] - np.sin(x[2]) * u[1]

        return Fx

    def F_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f_x with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the control input / odometry

        Returns
        -------
        np.ndarray, shape=(3,3)
            The Jacobian of f_x wrt. u.
        """
        Fu = np.zeros((3,3))

        R_w_b = rotmat2(x[2])
        Fu[:2,:2] = R_w_b
        Fu[2,2] = 1.0

        return Fu

    def F(self, x: np.ndarray, m: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of f wrt. eta (x, m).

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.
        u : np.ndarray, shape=(3,)
            the control input / odometry

        Returns
        -------
        np.ndarray, shape=(3, 3 + 2 * #landmarks)
            the jacobian of f wrt. eta (x, m).
        """
        num_m = m.shape[0]

        Fx = self.F_x(x, u)
        Fm = np.eye((2 * num_m))

        F = np.zeros((3 + 2 * num_m, 3 + 2 * num_m))
        F[:3, :3] = Fx
        F[3:, 3:] = Fm
        return F

    def Q(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the discrete process noise covariance matrix Q by
        rotating body-frame noise into the world frame.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the control input / odometry

        Returns
        -------
        np.ndarray, shape=(3,3)
            The process noise covariance matrix Q.
        """
        Sigma_u = np.diag([self.sigma_x**2, self.sigma_y**2, self.sigma_theta**2])
        F_u = self.F_u(x, u) 
        Q = F_u @ Sigma_u @ F_u.T # body-frame noise rotated into the world frame
        return Q

    def predict_pose(self, x_prev, Px_prev, u):
        """Predict the next pose distribution.
        """
        x_pred = self.f_x(x_prev, u)
        Fx = self.F_x(x_prev, u)
        Px_pred = Fx @ Px_prev @ Fx.T + self.Q(x_prev, u)
        return x_pred, Px_pred

    def predict_landmarks(self, m_prev, Pm_prev):
        """Predict the next landmarks distribution (identity).
        """
        m_pred = m_prev
        Pm_pred = Pm_prev
        return m_pred, Pm_pred

    def predict_state(self, eta_prev, P_prev, u):
        """Predict the next state distribution.
        """
        # TODO: optimize
        x_prev = eta_prev[:3]
        m_prev = eta_prev[3:]

        x_pred, Px_pred = self.predict_pose(x_prev, P_prev[:3,:3], u)
        m_pred, Pm_pred = self.predict_landmarks(m_prev, P_prev[3:,3:])
        eta_pred = np.hstack((x_pred, m_pred))

        P_pred = P_prev.copy()
        P_pred[:3,:3] = Px_pred
        P_pred[3:,3:] = Pm_pred  

        return eta_pred, P_pred


@dataclass
class OdometryR2:
    """Dynamic model for a robot in R2 using odometry as control input.

    State vector: x = [x, y]
    Control vector: u = [dx, dy] 
    """
    pass