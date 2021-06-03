import numpy as np
from sklearn.decomposition import randomized_svd

def compute_leverage(matrixA, low_rank=False, n_components=20,
                     n_iter=5):
    '''
    Computes leverage scores of the input matrix A with two possible options
    1) Exact Leverage Scores
    2) Low-rank leverage scores

    In each case the leverage scores are computed using the SVD. Here we
    compute the row leverage scores for the imput matrix A and a leverage score
    vector corresponding to each row is returned.

    In either case, the SVD is computed as: U, S, V = svd(matrixA)

    Parameters
    ----------
    matrixA: 2D array

    low-rank: bool (optional)
    default: False

    n_components: int (optional)
    default: 20

    n_iter: int (optional)
    default: 5

    Returns
    -------
    lev_vec: 1D array
    '''
    # Transpose is taken for computing row leverage scores
    _ , _, v_mat = np.linalg.svd(matrixA.T, full_matrices=False)

    # faster approximation of the SVD using randomized SVD from sklearn.
    if low_rank:
        _, _, v_mat = randomized_svd(matrixA.T,
                                     n_components=n_components,
                                     n_iter=n_iter,
                                     random_state=None)

    # gets the row-norms
    lev_vec = np.sum(v_mat ** 2, axis=0)
    return lev_vec