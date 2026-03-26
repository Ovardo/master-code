import sys

import gtsam
import numpy as np
import sympy as sp

from utils.utils_victoria_park import Car


def symbolic_jacobian():
    ve, alpha, dt, L, H = sp.symbols('ve alpha dt L H')

    vc = ve / (1 - H * sp.tan(alpha) / L) 
    
    dp = dt * vc * sp.tan(alpha) / L
    dx = dt * vc * sp.sinc(dp / sp.pi)
    dy = dt * vc * (1 - sp.cos(dp)) / dp

    odo = sp.simplify(sp.Matrix([dx, dy, dp]))
    J_odo_u = sp.simplify(odo.jacobian([ve, alpha]))

    sp.pprint(odo)
    sp.pprint(J_odo_u)

    # Convert symbolic to numerical function
    params = {'L': Car.L, 'H': Car.H}
    
    odo_func = sp.lambdify((ve, alpha, dt), odo.subs(params), 'numpy')
    J_odo_u_func = sp.lambdify((ve, alpha, dt), J_odo_u.subs(params), 'numpy')

    return odo_func, J_odo_u_func


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

def odometry_expmap_w_jacobian(ve, alpha, dt, dy_sign=+1.0):
    """
    SE(2) Expmap-based odometry increment + Jacobian wrt inputs [ve, alpha].

    Inputs:
      ve    : wheel encoder speed (your measured)
      alpha : steering angle
      dt    : timestep
      dy_sign: set to -1.0 if you want dy = vc*dt*(cos-1)/theta (your current sign)

    Returns:
      odo : np.array([dx, dy, dtheta])
      J   : 3x2 Jacobian d(odo)/d[ve, alpha]
    """
    car = Car() # use default car parameters
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



if __name__ == "__main__":
    

    ve = 0.0
    alpha = 0.0
    dt = 0.1

    odo_func, J_func = symbolic_jacobian()
    
    sp.pprint(odo_func)
    sp.pprint(J_func)
    

    odo1 = odo_func(ve, alpha, dt)
    jac1 = J_func(ve, alpha, dt)

    odo2, jac2 = odometry_expmap_w_jacobian(ve, alpha, dt)
    odo3, jac3 = odom_increment_and_jac_from_ve_alpha(ve, alpha, dt)

    print(f"Symbolic:\n {odo1}")
    print(f"Analytic:\n {odo2}")
    print(f"GTSAM:\n {odo3}")

    print(f"Symbolic:\n {jac1}")
    print(f"Analytic:\n {jac2}")
    print(f"GTSAM:\n {jac3}")
   