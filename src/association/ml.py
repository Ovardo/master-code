""" Maximum Likelihood (Individual Compatibility) association """
import numpy as np
import scipy.stats
from association.jcbb import individualCompatibility

def maximum_likelihood(z, zbar, S, alpha_individual):
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
    threshold_new = scipy.stats.chi2.ppf(alpha_individual, df=2)
    threshold_ambigious = scipy.stats.chi2.ppf(0.90, df=2)

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