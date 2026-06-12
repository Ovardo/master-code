from __future__ import annotations

from pprint import pprint

import gtsam
import numpy as np


def compute_algebraic_connectivity(isam: gtsam.ISAM2, estimate: gtsam.Values) -> float:
    """Compute the algebraic connectivity (Fiedler value) of the factor graph.

    The algebraic connectivity is the second-smallest eigenvalue of the graph
    Laplacian. It quantifies how well connected the SLAM graph is: values close
    to zero indicate a weakly connected (near-disconnected) graph.

    Args:
        isam: The iSAM2 instance holding the factor graph.
        estimate: The current estimate, used to enumerate the graph nodes.

    Returns:
        The second-smallest eigenvalue (lambda_2) of the graph Laplacian.
    """
    graph = isam.getFactorsUnsafe()
    keys = estimate.keys()
    num_nodes = len(keys)

    if num_nodes < 2:
        return 0.0

    # Map GTSAM keys to dense indices (0 .. num_nodes - 1).
    key_map = {keys[i]: i for i in range(num_nodes)}

    # Build the adjacency matrix from binary factors (odometry, range-bearing).
    A = np.zeros((num_nodes, num_nodes))
    for i in range(graph.size()):
        factor = graph.at(i)
        if factor is None:
            continue

        factor_keys = factor.keys()
        if len(factor_keys) == 2:
            idx1 = key_map[factor_keys[0]]
            idx2 = key_map[factor_keys[1]]
            A[idx1, idx2] = 1
            A[idx2, idx1] = 1

    # Graph Laplacian L = D - A.
    D = np.diag(np.sum(A, axis=1))
    L = D - A

    # Eigenvalues of a symmetric matrix, ascending; lambda_2 is the second smallest.
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_2 = float(eigenvalues[1])

    return lambda_2


def covariance_equivalence_metrics(Sigma_global, Sigma_steiner, eps=1e-12):
    Delta = Sigma_steiner - Sigma_global

    norm_global = np.linalg.norm(Sigma_global, ord="fro")
    norm_delta = np.linalg.norm(Delta, ord="fro")
    rel_fro_error = norm_delta / (norm_global + eps)

    max_abs_error = np.max(np.abs(Delta))

    block_sizes = [3] + [2] * 20
    cross_error = max_cross_covariance_block_error(Delta, block_sizes)


    return {
        "dimension": Sigma_global.shape[0],
        "norm_global": norm_global,
        "norm_delta": norm_delta,
        "relative_frobenius_error": rel_fro_error,
        "max_absolute_error": max_abs_error,
        "max_cross_covariance_block_error": cross_error,
    }

def max_cross_covariance_block_error(Delta, block_sizes):
    """
    block_sizes: list of variable dimensions, e.g.
                 [3, 2, 2, 2, ...] for pose + 2D landmarks
    """
    starts = np.cumsum([0] + block_sizes[:-1])
    ends = np.cumsum(block_sizes)

    mask = np.ones_like(Delta, dtype=bool)

    # Remove diagonal variable blocks
    for start, end in zip(starts, ends):
        mask[start:end, start:end] = False

    return np.max(np.abs(Delta[mask]))


if __name__ == '__main__':
    cov_global = np.load('/Users/ovar/Documents/Master/master_code/runs/real/20260606_161504_final_cov_old/final_joint_covariance.npz')
    cov_steiner = np.load('/Users/ovar/Documents/Master/master_code/runs/real/20260606_144011_final_cov/final_joint_covariance.npz')
    print(cov_global.files)
    print(cov_steiner.files)
    metrics = covariance_equivalence_metrics(cov_global['covariance'], cov_steiner['covariance'])
    pprint(metrics, sort_dicts=False)