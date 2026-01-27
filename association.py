import numpy as np
from scipy.stats import chi2
from functools import lru_cache
from scipy.linalg import cho_factor, cho_solve  # if you have SciPy

from utilities import utils


chi2isf_cached = lru_cache(maxsize=None)(chi2.isf)


def do_data_association(z, x, m, measurement_model):
    """Perform data association using JCBB.

    Parameters
    ----------
    z : np.ndarray, shape=(M,2)
        the measurements stacked.
    x : np.ndarray, shape=(3,)
        the robot state.
    m : np.ndarray, shape=(L,2)
        the landmarks stacked.
    P : np.ndarray, shape=(3 + 2 * L, 3 + 2 * L)
        the state covariance matrix.
    measurement_model : MeasurementModel
        the measurement model.

    Returns
    -------
    np.ndarray, shape=(M,)
        the association vector.
    """
    max_range = measurement_model.max_range + 10  # some margin
    max_fov = measurement_model.max_fov + 0.1  # some margin

    zbar = measurement_model.h(x, m) # (L,2)
    zbar_filtered = []
    indices_filtered = []
    m_filtered = []

    for i, (z_i, m_i) in enumerate(zip(zbar, m)):
        if z_i[0] < max_range and abs(z_i[1]) < max_fov / 2:
            zbar_filtered.extend(z_i)
            indices_filtered.append(i)
            m_filtered.extend(m_i)

    zbar_filtered = np.array(zbar_filtered)
    indices_filtered = np.array(indices_filtered)
    m_filtered = np.array(m_filtered)

    if len(indices_filtered) == 0:
        return np.full(z.shape[0] // 2, -1, dtype=int)
    
    landmark_keys_filtered = [L(idx + 1) for idx in indices_filtered]  # +1 for 1-based indexing
    last_pose_key = [X(k)]
    state_keys_filtered = last_pose_key + landmark_keys_filtered

    P_filtered = marginals.jointMarginalCovariance(state_keys_filtered).fullMatrix()
    P_filtered = reorder_covariance_auto(P_filtered, source_keys=sorted(state_keys_filtered),
                                        target_keys=state_keys_filtered, values=result)

    S_filtered = measurement_model.predict_measurements_covariance(x, m_filtered, P_filtered)

    alpha1 = 0.99  # individual compatibility threshold
    alpha2 = 0.99  # joint compatibility threshold

    a = JCBB(z, zbar_filtered, S_filtered, alpha1, alpha2)

    return a



def JCBB(z, zbar, S, alpha_individual, alpha_joint):
    #assert len(z.shape) == 1, "z must be in one row in JCBB"
    #assert z.shape[0] % 2 == 0, "z must be equal in x and y"
    L = zbar.shape[0]
    M = z.shape[0]
    
    if M == 0:
        return np.array([], dtype=int) # no measurements, no associations
    if L == 0:
        return np.full(M, -1, dtype=int) # no landmarks, all unassociated
    
    assert S.shape == (2 * L, 2 * L), "S must be of shape (2*L, 2*L) in JCBB"

    a = np.full(M, -1, dtype=int)
    abest = np.full(M, -1, dtype=int)

    # ic has measurements rowwise and predicted measurements columnwise
    ic = individualCompatibility(z, zbar, S)
    g2 = chi2.isf(1-alpha_individual, 2)
    order = np.argsort(np.amin(ic, axis=1))
    zo = z[order]
    ico = ic[order]
    j = 0

    abesto = JCBBrec(zo, zbar, S, alpha_joint, g2, j, a, ico, abest)

    abest[order] = abesto

    return abest

def JCBBrec(z, zbar, S, alpha_joint, g2, j, a, ic, abest):
    M = z.shape[0]
    assert isinstance(M, int), "M in JCBBrec must be int"
    n = num_associations(a)

    if j >= M:  # end of recursion
        if n > num_associations(abest) or ( (n >= num_associations(abest)) and (NIS(z, zbar, S, a) < NIS(z, zbar, S, abest)) ):
            abest = a
        # else abest = previous abest from the input
        return abest
    else:  # still at least one measurement to associate
        I = np.argsort(ic[j, ic[j, :] < g2])
        # allinds = np.array(range(ic.shape[1]), dtype=int)
        usableinds = np.where(ic[j, :] < g2)[0]  # allinds[ic[j, :] < g2]
        # if np.any(np.where(ic[j, :] < g2)[0] != usableinds):
        #     raise ValueError

        for i in usableinds[I]:
            a[j] = i
            # jointly compatible?
            if NIS(z, zbar, S, a) < chi2isf_cached(1-alpha_joint, 2 * (n + 1)):
                # We need to decouple ici from ic, so copy is required
                ici = ic[j:, i].copy()
                ic[j:, i] = np.inf  # landmark not available any more.

                # Needs to explicitly copy a for recursion to work
                abest = JCBBrec(z, zbar, S, alpha_joint, g2, j+1, a.copy(), ic, abest)
                ic[j:, i] = ici  # set landmark available again for next round.

        if n + (M - j - 2) >= num_associations(abest):
            a[j] = -1
            abest = JCBBrec(z, zbar, S, alpha_joint, g2, j+1, a, ic, abest)

    return abest

def individualCompatibility(z, zbar, S):
    """Compute the individual compatibility matrix.

    Parameters
    ----------
    z : np.ndarray, shape=(M,L)
        the measurements stacked.
    zbar : np.ndarray, shape=(L,2)
        the predicted measurements stacked.
    S : np.ndarray, shape=(2 * L, 2 * L)
        the innovation covariance matrix.

    Returns
    -------
    np.ndarray, shape=(M, L)
        the individual compatibility matrix.
    """
    M = z.shape[0]
    L = zbar.shape[0]
    ic = np.zeros((M, L)) # TODO; might by a mix between M, L here and in thesis (not important as long as consistent)

    for i in range(M):
        for j in range(L):
            dz = z[i] - zbar[j]
            dz[1] = utils.wrapToPi(dz[1]) # Important!

            S_jj = S[2*j:2*j+2, 2*j:2*j+2]
            ic[i, j] = float(dz.T @ np.linalg.solve(S_jj, dz))

    return ic

def make_psd(A):
    """
    Converts a matrix A into a positive semi-definite matrix.
    """
    # 1. Symmetrize the matrix
    C = (A + A.T) / 2

    # 2. Compute eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(C)

    # 3. Set negative eigenvalues to zero
    eigval[eigval < 0] = 0

    # 4. Reconstruct the matrix
    psd_matrix = eigvec @ np.diag(eigval) @ eigvec.T
    return psd_matrix


def NIS(z, zbar, S, a):
    """Compute the Normalized Innovation Squared (NIS) for a given association.

    Parameters
    ----------
    z : np.ndarray, shape=(M,2)
        Stacked measurements.
    zbar : np.ndarray, shape=(L,2)
        Stacked predicted measurements.
    S : np.ndarray, shape=(2*L, 2*L)
        Innovation covariance matrix.
    a : np.ndarray, shape=(M,)
        Association vector, -1 for unassociated, >=0 for index into zbar.

    Returns
    -------
    float
        The NIS value.
    """
    # Boolean mask: which measurements are associated?
    is_ass = a >= 0
    if not np.any(is_ass):
        return np.inf

    # Associated indices
    ass_idxs = a[is_ass].astype(int)

    # Extract associated measurements and predictions
    ztest = z[is_ass]
    zbartest = zbar[ass_idxs]

    # Innovation
    v = ztest - zbartest

    # Wrap bearing (or angle) component in 2D form, then flatten once
    v[:, 1] = utils.wrapToPi(v[:, 1])
    v = v.ravel()

    # Build index vector for the relevant 2x2 blocks in S
    # Each landmark j uses rows/cols 2*j and 2*j+1
    base = 2 * ass_idxs
    inds = np.empty(2 * ass_idxs.size, dtype=int)
    inds[0::2] = base
    inds[1::2] = base + 1

    # Extract submatrix S_test
    Stest = S[np.ix_(inds, inds)]

    # Since S is a covariance, it should be SPD -> use Cholesky for speed/stability
    c, lower = cho_factor(Stest, overwrite_a=False, check_finite=False)
    y = cho_solve((c, lower), v, check_finite=False)

    # Quadratic form v^T S^{-1} v
    nis = float(v @ y)
    return nis


def num_associations(array):
    return np.count_nonzero(array > -1)

