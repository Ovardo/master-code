import numpy as np


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
    # 1. Symmetrize the matrix
    C = symmetrize(A)

    # 2. Compute eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(C)

    # 3. Set negative eigenvalues to zero
    eigval[eigval < 0] = 0

    # 4. Reconstruct the matrix
    psd_matrix = eigvec @ np.diag(eigval) @ eigvec.T
    return psd_matrix

