import numpy as np
import sys
sys.path.append('..')

from leverage_scores import compute_leverage

def uniform_sampling(matrixA, s, mode_3D=True):
    '''
    Performs uniform sampling on the rows of the matrix, i.e. axis=0

    Parameters
    ----------
    matrixA : 2D array
        Any 2D array that needs to be sketched.
    s : sketch size
        DESCRIPTION.
    mode_3D : bool, optional
        Switch off, if not using in conjunction with P2S2. The default is True.

    Returns
    -------
    2D matrix
        Sketched matrix

    '''
    m, n = matrixA.shape
    idx_vec = np.random.choice(m, s, replace=True)
    matrixC = matrixA[idx_vec]
    if mode_3D:
        return matrixC[:, np.newaxis, :]
    else:
        return matrixC

def deter_row_sample(matrixA, s):
    '''
    Deterministically chooses the top s rows of the matrix A that is given
    as input. Note: Always the exact leverage scores need to be computed when
    using this function.

    Parameters
    ----------
    matrixA : 2D array

    s : int
        sketch size.

    Returns
    -------
    2D array
        s x d matrix with rows corresponding to the top s leverage scores.

    '''
    lev_scores = compute_leverage(matrixA)
    idx_vec = np.argsort(lev_scores, axis=0)[::-1][range(s)]
    matrixC = matrixA[idx_vec, :]
    return matrixC[:, np.newaxis, :]

def row_sample(matrixA, s, prob_vec):
    '''
    Samples the rows of a matrix according to the statistical leverage scores.
    The leverage scores are taken as input to the function in prob_vec.
    This function needs to be used in conjunction with the compute_leverage
    function.

    Parameters
    ----------
    matrixA : 2D array

    s : int
        sketch size.

    Returns
    -------
    2D array

    '''
    m = matrixA.shape[0]
    prob_vec /= sum(prob_vec)
    idx_vec = np.random.choice(m, s, replace=False, p=prob_vec)
    scaling_vec = np.sqrt(s * prob_vec[idx_vec]) + 1e-10
    matrixC = matrixA[idx_vec] / scaling_vec.reshape(len(scaling_vec),1)
    return matrixC[:, np.newaxis, :]