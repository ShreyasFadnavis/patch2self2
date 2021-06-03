import numpy as np
import sys
sys.path.append('..')

from countsketch import count_sketch
from srft import srft

def lev_approx(matrixA, lev_sketch_type, lev_sketch_size=5):
    '''

    Parameters
    ----------
    matrixA : 2D array

    lev_sketch_type : 'string'

    lev_sketch_size : int, optional
        The default is 5.

    Returns
    -------
    lev_vec : 1D array
        Array of approximate leverage scores.

    '''

    m, n = matrixA.shape
    s = int(n * lev_sketch_size)

    if lev_sketch_type == 'countsketch':
        matrixB = np.squeeze(count_sketch(matrixA, s))

    elif lev_sketch_type == 'srft':
        matrixB = np.squeeze(srft(matrixA, s))

    _, S, V = np.linalg.svd(matrixB, full_matrices=False)

    matrixT = V.T / S
    matrixY = np.dot(matrixA, matrixT)

    lev_vec = np.sum(matrixY ** 2, axis=1)
    return lev_vec