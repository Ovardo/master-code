"""Utility helpers for GTSAM."""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import gtsam
import numpy as np

from utils.utils_math import symmetrize

# -------------------------------
# Core utilities
# -------------------------------

def kstr(key: int) -> str: 
    # gtsam.DefaultKeyFormatter(key) does almost the same x1 vs X1 etc
    return f"{chr(gtsam.symbolChr(key)).upper()}{gtsam.symbolIndex(key)}"

def pose2_to_array(p: gtsam.Pose2) -> np.ndarray:
    """Convert Pose2 -> np.array([x, y, theta])"""
    return np.array([p.x(), p.y(), p.theta()])

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


# -------------------------------
# Covariance reordering utils
# ! keys are sorted first by symbol char, then by index
# -------------------------------


"""
Utility helpers for reordering block-structured covariance / information
matrices in GTSAM between different key orderings (e.g., elimination order
vs. sorted-key order).

Works with mixed variable types (Pose2/Pose3/Point2/Point3/Rot2/Rot3 and
vector-valued states). Dimensions are inferred from a provided Values and/or
Marginals object, or you can pass a dims dict explicitly.
"""


def _block_indices(keys: Sequence[int], dims: Mapping[int, int]) -> tuple[dict[int, tuple[int, int]], int]:
    """Compute contiguous [start, end) index slices for each key.

    Returns (index_map, total_dim).
    """
    idx_map: dict[int, tuple[int, int]] = {}
    total = 0
    for k in keys:
        d = int(dims[k])
        idx_map[k] = (total, total + d)
        total += d
    return idx_map, total


def infer_dims(
    keys: Iterable[int],
    values: Optional["gtsam.Values"] = None,
    marginals: Optional["gtsam.Marginals"] = None,
) -> dict[int, int]:
    """Infer per-key block dimensions.

    Preference order:
      1) Values (cheap): try vector getter, then typed getters
      2) Marginals (universal, slower): use marginal covariance size
    """
    dims: dict[int, int] = {}
    keys_list = list(keys)

    for k in keys_list:
        # Try Values first
        if values is not None:
            try:
                v = value_at(values, k)
                dims[k] = value_dim(v)
                continue
            except Exception:
                pass
        # Fall back to Marginals
        if marginals is not None:
            try:
                d = int(marginals.marginalCovariance(k).shape[0])
                dims[k] = d
                continue
            except Exception:
                pass
        # Give helpful name if possible
        sym_str = None
        if gtsam is not None:
            try:
                s = gtsam.Symbol(k)
                sym_str = f"{chr(s.chr())}{s.index()}"
            except Exception:
                pass
        name = sym_str or str(k)
        raise ValueError(
            f"Could not infer dimension for key {name}. Provide dims=... or pass values/marginals."
        )

    return dims


# -------------------------------
# Ordering helpers
# -------------------------------


def elimination_keys(ordering) -> list[int]:
    """Return keys in the elimination ordering of a GTSAM Ordering object."""
    return [ordering.at(i) for i in range(ordering.size())]


# -------------------------------
# Covariance reordering
# -------------------------------

def reorder_covariance_blocks(
    cov: np.ndarray,
    source_keys: Sequence[int],
    target_keys: Sequence[int],
    dims: Mapping[int, int],
    symmetrize_out: bool = True,
) -> np.ndarray:
    """Reorder a block-structured covariance from source_keys order to target_keys order
    using block slicing (no dense permutation matrix).
    """
    if set(source_keys) != set(target_keys):
        raise ValueError("source_keys and target_keys must contain the same keys")

    src_idx, total = _block_indices(source_keys, dims)
    if cov.shape != (total, total):
        raise ValueError("Covariance shape does not match block structure")

    # Stack rows into target order
    row_blocks = [cov[src_idx[k][0]:src_idx[k][1], :] for k in target_keys]
    tmp = np.vstack(row_blocks)

    # Stack cols into target order
    col_blocks = [tmp[:, src_idx[k][0]:src_idx[k][1]] for k in target_keys]
    out = np.hstack(col_blocks)

    return symmetrize(out) if symmetrize_out else out


def reorder_covariance(
    cov: np.ndarray,
    source_keys: Sequence[int],
    target_keys: Sequence[int],
    dims: Mapping[int, int],
) -> np.ndarray:
    """Reorder using an explicit (block) permutation matrix. Usually slower than
    reorder_covariance_blocks but kept for completeness/testing.
    """
    n = sum(int(dims[k]) for k in source_keys)
    if cov.shape != (n, n):
        raise ValueError("Covariance shape does not match dimensions")

    # build source/target index maps
    src_idx, _ = _block_indices(source_keys, dims)
    tgt_idx, _ = _block_indices(target_keys, dims)

    P = np.zeros((n, n))
    for k in source_keys:
        i0, i1 = src_idx[k]
        j0, j1 = tgt_idx[k]
        P[j0:j1, i0:i1] = np.eye(int(dims[k]))

    return symmetrize(P @ cov @ P.T)


def reorder_covariance_auto(
    cov: np.ndarray,
    source_keys: Sequence[int],
    target_keys: Sequence[int],
    *,
    dims: Optional[Mapping[int, int]] = None,
    values: Optional["gtsam.Values"] = None,
    marginals: Optional["gtsam.Marginals"] = None,
    fast: bool = True,
) -> np.ndarray:
    """Reorder covariance without precomputing dims.

    Provide either dims, or a Values/Marginals to infer dims. If fast=True, use
    block slicing implementation; otherwise use explicit permutation matrix.
    """
    if dims is None:
        dims = infer_dims(set(source_keys) | set(target_keys), values=values, marginals=marginals)
    if fast:
        return reorder_covariance_blocks(cov, source_keys, target_keys, dims)
    return reorder_covariance(cov, source_keys, target_keys, dims)


# -------------------------------
# Convenience: slicing to a subset (e.g., joint marginals)
# -------------------------------

def slice_covariance_for_keys(
    cov: np.ndarray,
    source_keys: Sequence[int],
    subset_keys: Sequence[int],
    dims: Mapping[int, int],
) -> np.ndarray:
    """Extract the joint covariance for a subset of keys, preserving subset order."""
    src_idx, total = _block_indices(source_keys, dims)
    if cov.shape != (total, total):
        raise ValueError("Covariance shape does not match block structure")

    # rows
    row_blocks = [cov[src_idx[k][0]:src_idx[k][1], :] for k in subset_keys]
    tmp = np.vstack(row_blocks)
    # cols
    col_blocks = [tmp[:, src_idx[k][0]:src_idx[k][1]] for k in subset_keys]
    out = np.hstack(col_blocks)
    return symmetrize(out)


# -------------------------------
# Example usage (documentation purposes only)
# -------------------------------
# from gtsam import Ordering
# ordering = Ordering.ColamdGaussianFactorGraph(gfg)
# elim_keys = elimination_keys(ordering)
# R, d = gbn.matrix()
# P_elim = np.linalg.inv((R.T @ R))
# P_sorted = reorder_covariance_auto(P_elim, elim_keys, sorted(elim_keys), values=result, marginals=marginals)
