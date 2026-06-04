from __future__ import annotations

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
