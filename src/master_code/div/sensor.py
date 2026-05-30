from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from master_code.config import SlamConfig
from master_code.utils import cartesian2polar, make_psd, rotmat2, ssa

import gtsam


def get_sensor_model(cfg: SlamConfig) -> RangeBearing:
    """Factory function to create a sensor model based on the config."""
    return RangeBearing(
        sigma_range=cfg.noise.sigma_range,
        sigma_bearing=cfg.noise.sigma_bearing_rad,
        max_range=cfg.sensor.max_range,
        max_fov=np.deg2rad(cfg.sensor.fov_deg),
    )


@dataclass
class RangeBearing:
    """GTSAM implementation of range and bearing measurement model."""
    sigma_range: float
    sigma_bearing: float
    max_range: float  # [m] (currently not used)
    max_fov: float  # [rad] (currently not used)
    # sensor_offset: np.ndarray = np.zeros(2)  # [x_offset, y_offset] in robot frame
    
    def _h(self, T_k: gtsam.Pose2, l_i: np.ndarray[np.float64], filter: bool = False) -> np.ndarray:
        """Calculate the measurement prediction function h_ for a single landmark.

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        l_i : np.ndarray, shape=(2,)
            the landmark position of landmark i

        Returns
        -------
        np.ndarray, shape=(2,)
            the predicted measurement z_k^i.
        """
        r = T_k.range(l_i)
        b = T_k.bearing(l_i).theta()

        if filter and (r > self.max_range or abs(b) > self.max_fov/2):
            return np.array([np.inf, np.inf])

        return np.array([r, b])

    def h(self, T_k: gtsam.Pose2, map: np.ndarray[np.float64]) -> np.ndarray:
        """Calculate the measurement prediction function h for multiple landmarks.

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        map : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(#landmarks, 2)
            the stacked measurements z_k.
        """
        z = np.zeros((map.shape[0], 2))

        for i in range(map.shape[0]):
            z[i, :] = self._h(T_k, map[i, :])

        return z
    
    def _h_inverse(self, T_W_B: gtsam.Pose2, z_i: np.ndarray[np.float64]) -> np.ndarray:
        """Calculate the inverse measurement function h^-1 for a single landmark.

        Parameters
        ----------
        T_W_Bk : gtsam.Pose2
            pose of robot body frame (B) relative to world frame (W) 
        z_i : np.ndarray, shape=(2,)
            the measurement of landmark i (range, bearing)

        Returns
        -------
        np.ndarray, shape=(2,)
            the landmark position in world frame.
        """
        r, b = z_i

        B_l = gtsam.Rot2(b).rotate(gtsam.Point2(r, 0.0))
        W_l = T_W_B.transformFrom(B_l)
        return W_l
    
    def h_inverse(self, T_W_B: gtsam.Pose2, z: np.ndarray[np.float64]) -> np.ndarray:
        """Calculate the inverse measurement function h^-1 for multiple landmarks.

        Parameters
        ----------
        T_W_B : gtsam.Pose2
            pose of robot body frame (B) relative to world frame (W) 
        z : np.ndarray, shape=(#landmarks, 2)
            the measurements of landmarks (range, bearing)

        Returns
        -------
        np.ndarray, shape=(#landmarks, 2)
            the landmark positions in world frame.
        """
        map = np.zeros((z.shape[0], 2))

        for i in range(z.shape[0]):
            map[i, :] = self._h_inverse(T_W_B, z[i, :])

        return map

    def _H(self, T_k: gtsam.Pose2, l_i: np.ndarray[np.float64]) -> np.ndarray:
        """Calculate the Jacobian of h_ with respect to x.

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        l_i : np.ndarray, shape=(2,)
            the landmark position of landmark i

        Returns
        -------
        H1_ : np.ndarray, shape=(2,3)
            the Jacobian of h_ wrt T_k
        H2_ : np.ndarray, shape=(2,2)
            the Jacobian of h_ wrt l_i
        """
        H1_r = np.zeros((1,3), order='F')
        H1_b = np.zeros((1,3), order='F')
        
        H2_r = np.zeros((1,2), order='F')
        H2_b = np.zeros((1,2), order='F')
       
        try:
            T_k.range(l_i, H1_r, H2_r)
            T_k.bearing(l_i, H1_b, H2_b)
            
            H1_ = np.vstack((H1_r, H1_b))
            H2_ = np.vstack((H2_r, H2_b))
        except TypeError:
            x = np.array([T_k.x(), T_k.y(), T_k.theta()])
            numpy_model = RangeBearingNumpy(
                sigma_range=self.sigma_range,
                sigma_bearing=self.sigma_bearing,
                max_range=self.max_range,
                max_fov=self.max_fov,
            )
            H1_ = numpy_model.H_x(x, l_i)
            H2_ = numpy_model.H_m(x, l_i)

        return H1_, H2_

    def H(self, T_k: gtsam.Pose2, map: np.ndarray[np.float64]) -> np.ndarray:
        """Calculate the jacobian of h wrt. (x, m).

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        map : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. (x, m).
        """

        num_m = map.shape[0]

        H1 = np.zeros((2 * num_m, 3))
        H2 = np.zeros((2 * num_m, 2 * num_m))

        for i in range(num_m):
            H1_, H2_ = self._H(T_k, map[i, :])
            H1[2 * i : 2 * i + 2, :] = H1_
            H2[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = H2_

        H = np.hstack((H1, H2))
        return H
    
    def R_(self, T_k: gtsam.Pose2, l_i: np.ndarray[np.float64]) -> np.ndarray:
        """
        Calculate the measurement noise covariance matrix R for a single landmark.

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        l_i : np.ndarray, shape=(2,)
            the landmark position of landmark i

        Returns
        -------
        np.ndarray, shape=(2, 2)
            The measurement noise covariance matrix R for a single landmark.
        """
        R_ = np.diag([self.sigma_range**2, self.sigma_bearing**2])
        return R_

    def R(self, T_k: gtsam.Pose2, map: np.ndarray[np.float64]) -> np.ndarray:
        """
        Calculate the measurement noise covariance matrix R for multiple landmarks.

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        map : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 2 * #landmarks)
            The measurement noise covariance matrix R.
        """
        R = np.zeros((2 * map.shape[0], 2 * map.shape[0]))
        for i in range(map.shape[0]): # Can be optimized but allows for different noise per landmark if desired
            R[2*i : 2*i+2, 2*i : 2*i+2] = self.R_(T_k, map[i, :])
        return R
    
    def innovation_covariance(self, T_k: gtsam.Pose2, map: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Calculate the innovation covariance matrix S for predicted measurements.

        Parameters
        ----------
        T_k : gtsam.Pose2
            the robot pose at timestep k
        map : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.
        P : np.ndarray, shape=(3 + 2 * #landmarks, 3 + 2 * #landmarks)
            covariance of the joint state (robot pose and landmarks).

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 2 * #landmarks)
            The innovation covariance matrix S.
        """
        H = self.H(T_k, map)
        R = self.R(T_k, map)

        S = H @ P @ H.T + R

        return make_psd(S)


@dataclass
class RangeBearingNumpy:
    """Numpy implementation of range and bearing measurement model."""
    sigma_range: float
    sigma_bearing: float
    max_range: float  # [m] (currently not used)
    max_fov: float  # [rad] (currently not used)
    # sensor_offset: np.ndarray = np.zeros(2)  # [x_offset, y_offset] in robot frame
    
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
        delta_w = m_i - x[:2] # TODO: better handling of landmark close to robot pos
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
        Hx = np.zeros((2,3))

        delta_w = m_i - x[:2]
        r = np.linalg.norm(delta_w)

        if r < 1e-6:
            raise ValueError("Jacobian undefined for landmark very close to robot position.") # TODO: handle better

        Hx[0,:2] = -delta_w / r
        Hx[0,2] = 0.0
        Hx[1,:2] = -(delta_w @ rotmat2(np.pi/2).T) / (r**2)
        Hx[1,2] = -1.0

        # Make wrt to body frame
        Hx[:, :2] = Hx[:, :2] @ rotmat2(x[2]) 

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
        Hm = np.zeros((2,2))

        delta_w = m_i - x[:2]
        r = np.linalg.norm(delta_w)

        if r < 1e-6:
            raise ValueError("Jacobian undefined for landmark very close to robot position.") # TODO: handle better

        Hm[0,:2] = delta_w / r
        Hm[1,:2] = (delta_w @ rotmat2(np.pi/2).T) / (r**2)

        # Hx = self.H_x(x, m_i)
        # Hm = -Hx[:, :2]

        return Hm

    def H(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h wrt. (x, m).

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. (x, m).
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

    def predicted_measurement(self, x: np.ndarray, m: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted measurements of landmarks m"""
        z = self.h(x, m)
        H = self.H(x, m)
        R = self.R(x, m)

        S = H @ P @ H.T + R

        return z, S
    
    def predicted_measurement_covariance(self, x: np.ndarray, m: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Predict measurement covariance of landmarks m.
        
        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        m : np.ndarray, shape=(#landmarks, 2)
            landmarks stacked.
        P : np.ndarray, shape=(3 + 2 * #landmarks, 3 + 2 * #landmarks)
            covariance of the joint state (robot pose and landmarks).

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 2 * #landmarks)
            The predicted measurement covariance matrix S.
        """
        H = self.H(x, m)
        R = self.R(x, m)

        S = H @ P @ H.T + R

        return S




 
