# Vicotria Park utils 
# Shamelessly stolen from here: https://github.com/ramanans1/EKF-SLAM/blob/master/tree_extraction.py
# Small modifications by Odin Aleksander Severinsen 

from dataclasses import dataclass

import numpy as np

from utils.utils_math import ssa


@dataclass(frozen=True)
class Car:
    """
    Car parameters for Victoria Park dataset. See data/victoria_park/info.txt 
    for info and data/victoria_park/car.bmp for drawing 
    """
    L: float = 2.83 # acxel distance
    H: float = 0.76 # center to wheel encoder
    a: float = 0.95 # laser distance in front of first axel
    b: float = 0.5 # laser distance to the left of center


def detectTrees(scan):

    M11 = 75
    M10 = 1
    daa = 5 * np.pi / 306
    M2 = 1.5
    M2a = 10 * np.pi / 360
    M3 = 3
    M5 = 1
    daMin2 = 2 * np.pi / 360

    RR = scan

    AA = np.array(range(361)) * np.pi / 360

    (ii1,) = np.where(RR < M11)

    L1 = len(ii1)
    if L1 < 1:
        return []

    R1 = RR[ii1]
    A1 = AA[ii1]

    ii2 = np.flatnonzero((np.abs(np.diff(R1)) > M2) | (np.diff(A1) > M2a))

    L2 = len(ii2) + 1
    ii2u = np.append(ii2, L1 - 1)
    ii2 = np.insert(ii2 + 1, 0, 0)
    # ii2u = int16([ ii2, L1 ])
    # ii2  = int16([1, ii2+1 ])

    # %ii2 , size(R1) ,

    R2 = R1[ii2]
    A2 = A1[ii2]

    A2u = A1[ii2u]
    R2u = R1[ii2u]

    x2 = R2 * np.cos(A2)
    y2 = R2 * np.sin(A2)
    x2u = R2u * np.cos(A2u)
    y2u = R2u * np.sin(A2u)

    flag = np.zeros(L2)

    L3 = 0
    M3c = M3 * M3

    if L2 > 1:
        L2m = L2 - 1
        dx2 = x2[1:L2] - x2u[:L2m]
        dy2 = y2[1:L2] - y2u[:L2m]

        dl2 = dx2 * dx2 + dy2 * dy2
        ii3 = np.flatnonzero(dl2 < M3c)
        L3 = len(ii3)
        if L3 > 0:
            flag[ii3] = 1
            flag[ii3 + 1] = 1

        if L2 > 2:
            L2m = L2 - 2
            dx2 = x2[2:L2] - x2u[0:L2m]
            dy2 = y2[2:L2] - y2u[0:L2m]

            dl2 = dx2 * dx2 + dy2 * dy2
            ii3 = np.flatnonzero(dl2 < M3c)
            L3b = len(ii3)
            if L3b > 0:
                flag[ii3] = 1
                flag[ii3 + 2] = 1
                L3 = L3 + L3b

            if L2 > 3:
                L2m = L2 - 3
                dx2 = x2[3:L2] - x2u[0:L2m]
                dy2 = y2[3:L2] - y2u[0:L2m]

                dl2 = dx2 * dx2 + dy2 * dy2
                ii3 = np.flatnonzero(dl2 < M3c)
                L3b = len(ii3)
                if L3b > 0:
                    flag[ii3] = 1
                    flag[ii3 + 3] = 1
                    L3 = L3 + L3b

    if L2 > 1:
        ii3 = np.array(range(L2 - 1))
        ii3 = np.flatnonzero(
            (A2[ii3 + 1] - A2u[ii3]) < daMin2
        )  # objects close (in angle) from viewpoint.
        L3b = len(ii3)
        if L3b > 0:
            ff = R2[ii3 + 1] > R2u[ii3]  # which object is in the back?
            ii3 = ii3 + ff
            flag[ii3] = 1  # mark them for the deletion
            L3 = L3 + L3b
        iixx = ii3

    if L3 > 0:
        ii3 = np.flatnonzero(flag == 0)
        L3 = len(ii3)
        ii4 = ii2[ii3].astype(np.float64)
        ii4u = ii2u[ii3].astype(np.float64)
        R4 = R2[ii3]
        R4u = R2u[ii3]
        A4 = A2[ii3]
        A4u = A2u[ii3]
        x4 = x2[ii3]
        y4 = y2[ii3]
        x4u = x2u[ii3]
        y4u = y2u[ii3]
    else:
        ii4 = ii2.astype(np.float64)
        ii4u = ii2u.astype(np.float64)
        R4 = R2
        R4u = R2u
        A4 = A2
        A4u = A2u
        x4 = x2
        y4 = y2
        x4u = x2u
        y4u = y2u

    dx2 = x4 - x4u
    dy2 = y4 - y4u
    dl2 = dx2 * dx2 + dy2 * dy2

    ii5 = np.flatnonzero(dl2 < (M5 * M5))
    L5 = len(ii5)
    if L5 < 1:
        return np.zeros((0, 2))

    R5 = R4[ii5]
    R5u = R4u[ii5]
    A5 = A4[ii5]
    A5u = A4u[ii5]
    ii4 = ii4[ii5]
    ii4u = ii4u[ii5]

    ii5 = np.flatnonzero((R5 > M10) & (A5 > daa) & (A5u < (np.pi - daa)))

    L5 = len(ii5)
    if L5 < 1:
        return np.zeros((0, 2))

    R5 = R5[ii5]
    R5u = R5u[ii5]
    A5 = A5[ii5]
    A5u = A5u[ii5]
    ii4 = ii4[ii5]
    ii4u = ii4u[ii5]
    dL5 = (A5u + np.pi / 360 - A5) * (R5 + R5u) / 2

    compa = np.abs(R5 - R5u) < (dL5 / 3)

    ii6 = np.flatnonzero(~compa)
    ii6 = ii4[ii6]

    ii5 = np.flatnonzero(compa)
    L5 = len(ii5)
    if L5 < 1:
        return np.zeros((0, 2))

    R5 = R5[ii5]
    R5u = R5u[ii5]
    A5 = A5[ii5]
    A5u = A5u[ii5]
    ii4 = ii4[ii5]
    ii4u = ii4u[ii5]
    dL5 = dL5[ii5]

    auxi = (ii4 + ii4u) / 2
    iia = np.floor(auxi)
    iib = np.ceil(auxi)

    Rs = (R1[iia.astype(int)] + R1[iib.astype(int)]) / 2

    ranges = Rs + dL5 / 2.0
    angles = (A5 + A5u) / 2.0 - np.pi / 2
    diameters = dL5

    # z = np.array([[ranges], [angles]]).squeeze().T
    z = np.vstack((ranges, angles)).T  # keeps the dims
    # if z.shape != (2,): # to check for equality, all passed 19.oct 23:45 until k=3000
    #     assert np.allclose(np.vstack((ranges, angles)).T, z)
    # else:
    #     assert np.allclose(np.vstack((ranges, angles)).T[0], z)
    return z


def odometry_func(v_e, alpha, dt):
    car = Car() # use default car parameters
    
    H = car.H
    L = car.L

    # R = L / np.tan(alpha) # turning radius of the path (inf if alpha=0)
    # v_e = v_c * (1 - H / R) --> v_c = v_e / (1 - H/R) = v_e / (1 - H * np.tan(alpha) / L)
    # omega = v_c * np.tan(alpha) / L  # angular velocity (omega = v/R = v_c * tan(alpha) / L)
    # twist = np.array([v_c, 0.0, omega]) # [v_x, v_y, omega]
    # integrate twist to get odometry increment (using exact integration for unicycle model)
    
    v_c = v_e / (1 - H * np.tan(alpha) / L) 
    dp = dt * v_c * np.tan(alpha) / L
    dx = dt * v_c * np.sinc(dp / np.pi)
    if np.abs(dp) < 0.001:
        dy = dt * v_c * (dp / 2 - dp ** 3 / 24 + dp ** 5 / 720) # Taylor approximation
    else:
        dy = dt * v_c * (1 - np.cos(dp)) / dp

    odo = np.array([dx, dy, dp])

    # import gtsam
    twist = np.array([v_c*dt, 0.0, dp]) # [v_x, v_y, omega*dt]
    odos = gtsam.Pose2.Expmap(twist)
    J_exp = gtsam.Pose2.ExpmapDerivative(twist)

    return odo

import gtsam
import numpy as np


def odom_increment_and_jac_from_ve_alpha(ve, alpha, dt):
    """
    Build Pose2 increment via GTSAM Expmap + propagate input covariance Su on [ve, alpha]
    to Qu on [dx, dy, dtheta].

    Su: 2x2 covariance of [ve, alpha]
    """
    car = Car() # use default car parameters
    L = car.L
    H = car.H

    t = np.tan(alpha)
    sec2 = 1.0 / (np.cos(alpha)**2)
    k = H / L
    d = 1.0 - k * t

    vc = ve / d
    omega = (vc / L) * t

    # ---- twist for Pose2.Expmap ----
    twist = np.array([vc*dt, 0.0, omega*dt], dtype=float)

    # ---- Get Expmap + ExpmapDerivative from GTSAM ----
    odo = gtsam.Pose2.Expmap(twist)
    J_exp_twist = gtsam.Pose2.ExpmapDerivative(twist)
    
    # ---- d(twist)/d[ve, alpha] ----
    dvc_dve = 1.0 / d
    dvc_da  = ve * (k * sec2) / (d*d)

    drho1_dve = dt * dvc_dve
    drho1_da  = dt * dvc_da

    ddp_dve = (dt / L) * t * dvc_dve
    ddp_da  = (dt / L) * (t * dvc_da + vc * sec2)

    J_twist_u = np.array([
        [drho1_dve, drho1_da],
        [0.0,       0.0      ],
        [ddp_dve,   ddp_da   ],
    ], dtype=float)

    # ---- Chain rule to odometry coordinates ----
    J_exp_u = J_exp_twist @ J_twist_u

    return odo, J_exp_u


def _ab_and_derivs(theta, eps=1e-6):
    """
    a = sin(theta)/theta
    b = (1 - cos(theta))/theta
    and derivatives a', b' w.r.t theta

    Uses Taylor expansions near 0 for numerical stability.
    """
    th = float(theta)
    ath = abs(th)

    if ath < eps:
        # Series:
        # a = 1 - th^2/6 + th^4/120
        # b = th/2 - th^3/24 + th^5/720
        # a' = -th/3 + th^3/30 - th^5/840
        # b' = 1/2 - th^2/8 + th^4/144
        th2 = th*th
        th3 = th2*th
        th4 = th2*th2
        th5 = th4*th

        a  = 1.0 - th2/6.0 + th4/120.0
        b  = th/2.0 - th3/24.0 + th5/720.0
        ap = -th/3.0 + th3/30.0 - th5/840.0
        bp = 0.5 - th2/8.0 + th4/144.0
    else:
        s = np.sin(th)
        c = np.cos(th)
        a = s / th
        b = (1.0 - c) / th
        ap = (th*c - s) / (th*th)
        bp = (th*s - (1.0 - c)) / (th*th)

    return a, b, ap, bp

def odometry_expmap_w_jacobian(ve, alpha, dt, car, dy_sign=+1.0):
    """
    SE(2) Expmap-based odometry increment + Jacobian wrt inputs [ve, alpha].

    Inputs:
      ve    : wheel encoder speed (your measured)
      alpha : steering angle
      dt    : timestep
      car   : object with fields L, H  (wheelbase and offset)
      dy_sign: set to -1.0 if you want dy = vc*dt*(cos-1)/theta (your current sign)

    Returns:
      odo : np.array([dx, dy, dtheta])
      J   : 3x2 Jacobian d(odo)/d[ve, alpha]
    """
    L = car.L
    H = car.H

    t = np.tan(alpha)
    sec2 = 1.0 / (np.cos(alpha)**2)  # d/dalpha tan(alpha)

    k = H / L
    d = 1.0 - k * t                  # denominator

    # vc = ve / d
    vc = ve / d

    # omega = (vc/L) * tan(alpha)
    omega = (vc / L) * t

    # theta = omega dt
    theta = omega * dt

    # SE(2) left-Jacobian-like V(theta)
    a, b, ap, bp = _ab_and_derivs(theta)

    # rho = [vc*dt, 0]
    rho1 = vc * dt

    # Translation t = V(theta) rho, with rho2=0:
    dx = a * rho1
    dy = dy_sign * (b * rho1)  # choose sign convention

    dtheta = theta

    odo = np.array([dx, dy, dtheta], dtype=float)

    # ---- Vehicle-part derivatives: vc and theta wrt ve, alpha ----
    # dvc/dve = 1/d
    dvc_dve = 1.0 / d
    # dvc/dalpha = ve * (k * sec2) / d^2
    dvc_da  = ve * (k * sec2) / (d*d)

    # theta = dt * (vc/L) * t
    # dtheta/dve = dt/L * t * dvc/dve
    dth_dve = (dt / L) * t * dvc_dve
    # dtheta/dalpha = dt/L * ( t*dvc/da + vc*sec2 )
    dth_da  = (dt / L) * (t * dvc_da + vc * sec2)

    # rho1 = vc*dt
    drho1_dve = dt * dvc_dve
    drho1_da  = dt * dvc_da

    # ---- Geometry-part derivatives via chain rule ----
    # dx = a(theta) * rho1
    # d(dx)/du = a' * dtheta/du * rho1 + a * drho1/du
    ddx_dve = ap * dth_dve * rho1 + a * drho1_dve
    ddx_da  = ap * dth_da  * rho1 + a * drho1_da

    # dy = sign * b(theta) * rho1
    ddy_dve = dy_sign * (bp * dth_dve * rho1 + b * drho1_dve)
    ddy_da  = dy_sign * (bp * dth_da  * rho1 + b * drho1_da)

    # dtheta derivatives already computed
    J = np.array([
        [ddx_dve, ddx_da],
        [ddy_dve, ddy_da],
        [dth_dve, dth_da],
    ], dtype=float)

    return odo, J






# def odometry_w_jacobian(ve, alpha, dt, psi):
#     car = Car() # use default car parameters
    
#     H = car.H
#     L = car.L
#     a = car.a
#     b = car.b

#     tan_a = np.tan(alpha)
#     cos_p = np.cos(psi)
#     sin_p = np.sin(psi)

#     # central velocity (same as before; used for odometry + also inside Jacobian entries)
#     v_c = ve / (1.0 - tan_a * (H / L))

#     # Odometry increment
#     el1 = dt * (v_c*cos_p - (v_c/L)*tan_a*(a*sin_p + b*cos_p))
#     el2 = dt * (v_c*sin_p + (v_c/L)*tan_a*(a*cos_p - b*sin_p))
#     el3 = dt * (v_c/L) * tan_a
#     el31 = ssa(el3)

#     odometry = np.array([[el1, el2, el31]])

#     # ---------- Jacobian wrt [v_c, alpha] ----------
#     sec2_a = 1.0 / (np.cos(alpha) ** 2)  # d/dalpha tan(alpha)

#     # d/dv_c
#     d1_dvc = dt * (cos_p - (1.0/L)*tan_a*(a*sin_p + b*cos_p))
#     d2_dvc = dt * (sin_p + (1.0/L)*tan_a*(a*cos_p - b*sin_p))
#     d3_dvc = dt * (1.0/L) * tan_a

#     # d/dalpha (treating v_c as independent of alpha)
#     d1_dalpha = dt * (-(v_c/L) * sec2_a * (a*sin_p + b*cos_p))
#     d2_dalpha = dt * ( (v_c/L) * sec2_a * (a*cos_p - b*sin_p))
#     d3_dalpha = dt * ( (v_c/L) * sec2_a)

#     J = np.array([
#         [d1_dvc, d1_dalpha],
#         [d2_dvc, d2_dalpha],
#         [d3_dvc, d3_dalpha],
#     ])

#     return odometry, J

 





