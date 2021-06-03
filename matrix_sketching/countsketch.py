import numpy as np

def count_sketch(matrixA, s, mode_3D=True):
    '''
    Uses the CountSketch algorithm to compute: matrixC= S * matrixA

    Parameters
    ----------
    matrixA: 2D array

    s: sketch size

    Returns
    -------
    matrixC: 2D array
    '''
    m, n = matrixA.shape
    matrixC = np.zeros([s, n])
    hashedIndices = np.random.choice(s, m, replace=True)
    randSigns = np.random.choice(2, m, replace=True) * 2 - 1

    matrixA = matrixA * randSigns.reshape(m, 1)

    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[i] = np.sum(matrixA[idx], 0)

    if mode_3D:
        return matrixC[:, np.newaxis, :]
    else:
        return matrixC