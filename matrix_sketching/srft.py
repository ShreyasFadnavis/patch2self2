import numpy as np

def _real_fft(matrixA):
    '''
    Helper function to perform real FFT on the input matrix
    The output matrix C is computed as C = F*A where:
    F is the n-by-n orthogonal real FFT matrix

    Parameters
    ----------
    matrixA: 2D array

    Returns
    -------
    matrixC: 2D array
    '''
    n_int = matrixA.shape[0]
    fft_mat = np.fft.fft(matrixA, n=None, axis=0) / np.sqrt(n_int)
    if n_int % 2 == 1:
        cutoff_int = int((n_int+1) / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int, n_int))
    else:
        cutoff_int = int(n_int/2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int+1, n_int))
    matrixC = fft_mat.real
    matrixC[idx_real_vec] *= np.sqrt(2)
    matrixC[idx_imag_vec] = fft_mat[idx_imag_vec].imag * np.sqrt(2)
    return matrixC[:, np.newaxis, :]

def srft(matrixA, s, mode_3D=True):
    '''
    Subsampled Randomized Fourier Transform (SRFT)
    Computes a sketched matrix C based on the Fourier tranform computed
    in `_real_fft`. The sketching matrix S is used to obtain the sketched
    matrix C as: C = SA

    Parameters
    ----------
    MatrixA: 2D array

    s: sketch size

    Returns
    ----------
    matrixC: 2D array (sketched output)
    '''
    n_int = matrixA.shape[0]
    sign_vec = np.random.choice(2, n_int) * 2 - 1
    idx_vec = np.random.choice(n_int, s, replace=False)
    a_mat = sign_vec.reshape(n_int,1) * matrixA
    a_mat = _real_fft(matrixA)
    matrixC = matrixA[idx_vec] * np.sqrt(n_int / s)

    if mode_3D:
        return matrixC[:, np.newaxis, :]
    else:
        return matrixC
