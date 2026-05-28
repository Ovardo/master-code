import numpy as np
import gtsam

def rotmat2(thetas):
    """2x2 if single theta, or Nx2x2 if array of thetas."""
    thetas = np.asarray(thetas)

    c = np.cos(thetas)
    s = np.sin(thetas)

    return np.stack([
        np.stack([ c, -s], axis=-1),
        np.stack([ s,  c], axis=-1)
    ], axis=-2)

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

def reorder_covariance_naive(cov: np.ndarray) -> np.ndarray:
    """
    Reorder covariance from [lm_id1, lm_id2, ..., pose] to [pose, lm_id1, lm_id2, ...], 
    assuming pose is last 3 entries and rest are landmarks with id(j) < id(j+1)
    NOTE: this needs more attention when introducing other variable types than X and L in GTSAM
    or when when landmark ordering is not guaranteed to be by id...
    """
    n = cov.shape[0]
    perm = np.r_[n-3:n, 0:n-3]
    cov_new = cov[np.ix_(perm, perm)]
    return cov_new

