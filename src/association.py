"""Association module for data association in SLAM."""

from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

from config import InferenceConfig
from utils import utils_math

chi2isf_cached = lru_cache(maxsize=None)(chi2.isf)


class Associator:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg

    def associate(self, z: np.ndarray, zbar: np.ndarray, innovation_covariance: np.ndarray) -> np.ndarray:
        """Associate measurements to predicted measurements using method specified in self.cfg.association_method.
        
        Parameters
        ----------
        z : np.ndarray, shape=(M,2)
            Stacked measurements.
        zbar : np.ndarray, shape=(L,2)
            Stacked predicted measurements.
        innovation_covariance : np.ndarray, shape=(2*L, 2*L)
            Innovation covariance matrix.

        Returns
        -------
        np.ndarray, shape=(M,)
            Association vector with elements:
            - \>=0 for index into zbar
            - -1 for unassociated (new landmark or outlier)
            - -2 for ambiguous association (unsure if outlier/new landmark or valid match)
        """
        method = self.cfg.association_method

        if method == 'jcbb':
            associations = JCBB_assocation(z, zbar, innovation_covariance, self.cfg.alpha_individual, self.cfg.alpha_joint)
        elif method == 'ml':
            associations = ML_association(z, zbar, innovation_covariance, self.cfg.alpha_individual)
        elif method == 'gt':
            pass # to be implemented
        else:
            raise ValueError(f"Unknown association method: {self.cfg.association_method}")
       
        return associations
    

def JCBB_assocation(z, zbar, S, alpha_individual, alpha_joint):
    #assert len(z.shape) == 1, "z must be in one row in JCBB"
    #assert z.shape[0] % 2 == 0, "z must be equal in x and y"
    
    L = zbar.shape[0] # num predicted measurements 
    M = z.shape[0] # num actual measurements

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

    abesto = _JCBBrec(zo, zbar, S, alpha_joint, g2, j, a, ico, abest)

    abest[order] = abesto

    return abest


def _JCBBrec(z, zbar, S, alpha_joint, g2, j, a, ic, abest):
    M = z.shape[0]
    assert isinstance(M, int), "M in JCBBrec must be int"
    n = num_associations(a)

    if j >= M:  # end of recursion
        if n > num_associations(abest) or ( (n >= num_associations(abest)) and (NIS(z, zbar, S, a) < NIS(z, zbar, S, abest)) ):
            abest = a
        # else abest = previous abest from the input
        return abest

    # still at least one measurement to associate...
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
            abest = _JCBBrec(z, zbar, S, alpha_joint, g2, j+1, a.copy(), ic, abest)
            ic[j:, i] = ici  # set landmark available again for next round.

    if n + (M - j - 2) >= num_associations(abest):
        a[j] = -1
        abest = _JCBBrec(z, zbar, S, alpha_joint, g2, j+1, a, ic, abest)

    return abest


def ML_association(z, zbar, S, alpha_individual):
    """Maximum Likelihood association based on individual compatibility.

    Parameters
    ----------
    z : np.ndarray, shape=(M,2)
        Stacked measurements.
    zbar : np.ndarray, shape=(L,2)
        Stacked predicted measurements.
    S : np.ndarray, shape=(2*L, 2*L)
        Innovation covariance matrix.
    alpha_individual : float
        Confidence level for individual compatibility test.

    Returns
    -------
    np.ndarray, shape=(M,)
        Association vector, -1 for unassociated, >=0 for index into zbar.
    """
    M = z.shape[0]
    L = zbar.shape[0]

    if M == 0:
        return np.array([], dtype=int) # no measurements, no associations
    if L == 0:
        return np.full(M, -1, dtype=int) # no landmarks, all unassociated
    
    a = np.full(M, -1, dtype=int)  # Initialize all as unassociated

    # Compute individual compatibility matrix
    ic = individualCompatibility(z, zbar, S)

    # Threshold for individual compatibility (chi-squared with 2 DOF)
    threshold_new = chi2.isf(1-alpha_individual, df=2)
    threshold_ambigious = chi2.isf(1-0.95, df=2) # TODO: floating threshold

    for i in range(M):
        # Find the best match for measurement i
        j_best = np.argmin(ic[i])
        if ic[i, j_best] > threshold_new:
            a[i] = -1  # No association, measurement i is an outlier
        elif ic[i, j_best] > threshold_ambigious:
            a[i] = -2  # Ambiguous association, could be an outlier or a valid match
        else:
            a[i] = j_best  # Associate measurement i with predicted measurement j_best

    return a



def individualCompatibility(z, zbar, S):
    """
    Compute the individual compatibility matrix.

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
            dz[1] = utils_math.ssa(dz[1]) # Important!

            S_jj = S[2*j:2*j+2, 2*j:2*j+2]
            ic[i, j] = float(dz.T @ np.linalg.solve(S_jj, dz))

    return ic


def NIS(z, zbar, S, a):
    """
    Compute the Normalized Innovation Squared (NIS) for a given association.

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
    v[:, 1] = utils_math.ssa(v[:, 1])
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

