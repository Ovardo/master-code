import numpy as np
import gtsam

def rotmat2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def ssa(angle):
    """Smallest Signed Angle between -pi and pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def cartesian2polar(x: float, y: float) -> tuple[float, float]:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T)/2 to clean up small asymmetries from numerics."""
    return (A + A.T) / 2

def make_psd(A: np.ndarray) -> np.ndarray:
    """Converts a matrix A into a positive semi-definite matrix."""
    A_sym = symmetrize(A)
    eigval, eigvec = np.linalg.eigh(A_sym)
    eigval[eigval < 0] = 0
    psd_matrix = eigvec @ np.diag(eigval) @ eigvec.T
    return psd_matrix

def pose2_to_array(p: gtsam.Pose2) -> np.ndarray:
    """Convert gtsam.Pose2 -> np.array([x, y, theta])"""
    return np.array([p.x(), p.y(), p.theta()])

