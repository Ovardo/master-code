"""
Utility helpers for reordering block-structured covariance / information
matrices in GTSAM between different key orderings (e.g., elimination order
vs. sorted-key order).

Works with mixed variable types (Pose2/Pose3/Point2/Point3/Rot2/Rot3 and
vector-valued states). Dimensions are inferred from a provided Values and/or
Marginals object, or you can pass a dims dict explicitly.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Optional
from utilities.utils import symmetrize, value_at, value_dim
import numpy as np

try:
    import gtsam  # type: ignore
except Exception:  # pragma: no cover
    gtsam = None  # Allows linting without GTSAM present


# -------------------------------
# Core small utilities
# -------------------------------

def _block_indices(keys: Sequence[int], dims: Mapping[int, int]) -> Tuple[Dict[int, Tuple[int, int]], int]:
    """Compute contiguous [start, end) index slices for each key.

    Returns (index_map, total_dim).
    """
    idx_map: Dict[int, Tuple[int, int]] = {}
    total = 0
    for k in keys:
        d = int(dims[k])
        idx_map[k] = (total, total + d)
        total += d
    return idx_map, total


# -------------------------------
# Dimension inference
# -------------------------------

def infer_dims(
    keys: Iterable[int],
    values: Optional["gtsam.Values"] = None,
    marginals: Optional["gtsam.Marginals"] = None,
) -> Dict[int, int]:
    """Infer per-key block dimensions.

    Preference order:
      1) Values (cheap): try vector getter, then typed getters
      2) Marginals (universal, slower): use marginal covariance size
    """
    dims: Dict[int, int] = {}
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

def elimination_keys(ordering) -> List[int]:
    """Return keys in the elimination ordering of a GTSAM Ordering object."""
    return [ordering.at(i) for i in range(ordering.size())]


def sort_keys_human(keys: Sequence[int]) -> List[int]:
    """Sort keys lexicographically by (symbol char, index) if gtsam is available,
    else default to numeric sort. Useful to match Marginals/Hessian conventions.
    """
    if gtsam is None:
        return sorted(keys)
    def key_tuple(k: int):
        try:
            s = gtsam.Symbol(k)
            return (s.chr(), s.index())
        except Exception:
            return ("\0", k)
    return sorted(keys, key=key_tuple)


# -------------------------------
# Covariance reordering
# -------------------------------

def reorder_covariance_blocks(
    cov: np.ndarray,
    source_keys: Sequence[int],
    target_keys: Sequence[int],
    dims: Mapping[int, int],
    *, symmetrize_out: bool = True,
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
# sorted_keys = sort_keys_human(elim_keys)
# R, d = gbn.matrix()
# P_elim = np.linalg.inv((R.T @ R))
# P_sorted = reorder_covariance_auto(P_elim, elim_keys, sorted_keys, values=result, marginals=marginals)
