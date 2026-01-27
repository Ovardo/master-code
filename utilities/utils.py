import gtsam
import numpy as np

def kstr(key: int) -> str: 
    # gtsam.DefaultKeyFormatter(key) does almost the same x1 vs X1 etc
    return f"{chr(gtsam.symbolChr(key)).upper()}{gtsam.symbolIndex(key)}"

def pose2_to_array(p: gtsam.Pose2) -> np.ndarray:
    """Convert Pose2 -> np.array([x, y, theta])"""
    return np.array([p.x(), p.y(), p.theta()])

def rotmat2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def cartesian2polar(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T)/2 to clean up small asymmetries from numerics."""
    return 0.5 * (A + A.T)

def value_at(values, key: int):
    """
    Retrieve a stored variable from GTSAM Values / VectorValues.
    Returns a numpy array (for VectorValues / Point*) or a typed GTSAM object (Pose*/Rot*).
    """
    if values.exists(key) is False:
        raise KeyError(f"Key {key} not found in provided Values/VectorValues")
    
    if isinstance(values, gtsam.VectorValues):
        return values.at(key)

    # Common typed getters for gtsam.Values
    getters = (
        ("atPose3"),
        ("atPose2"),
        ("atPoint3"),
        ("atPoint2"),
        ("atRot2"),
        ("atRot3"),
        ("atVector"), # (covers both atPoint2 and atPoint3)
    )
    # TODO: explore use at<ClassType> instead of trying all(?)
    for name in getters:
        f = getattr(values, name, None)
        if f is None:
            continue
        try:
            return f(key)
        except Exception:
            continue

    raise ValueError(f"Could not retrieve key {key} from Values using known getters")

def value_dim(value) -> int:
    """Return tangent-space dim for a stored value.

    - numpy arrays -> size
    - GTSAM types -> .dim()
    """
    if isinstance(value, np.ndarray):
        return int(value.size)
    if hasattr(value, "dim"):
        try:
            return int(value.dim())
        except Exception as e:
            raise TypeError(f"dim() failed for value of type {type(value)}: {e}")
    raise TypeError(f"Cannot infer dimension for value of type {type(value)}")