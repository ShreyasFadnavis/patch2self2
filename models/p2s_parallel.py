import numpy as np
from warnings import warn
import time
from dipy.utils.optpkg import optional_package
import dipy.core.optimize as opt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from sklearn import linear_model

sklearn, has_sklearn, _ = optional_package('sklearn')
linear_model, _, _ = optional_package('sklearn.linear_model')

if not has_sklearn:
    warn(sklearn._msg)

def _vol_denoise(train, vol_idx, model, data_shape, alpha):
    
    if isinstance(model, str):
        if model.lower() == 'ols':
            model_instance = linear_model.LinearRegression(copy_X=False)
        elif model.lower() == 'ridge':
            model_instance = linear_model.Ridge(copy_X=False, alpha=alpha)
        elif model.lower() == 'lasso':
            model_instance = linear_model.Lasso(copy_X=False, max_iter=50, alpha=alpha)
        else:
            raise ValueError(f"Invalid model string: {model}. Should be 'ols', 'ridge', or 'lasso'.")
    elif isinstance(model, linear_model.BaseEstimator):
        model_instance = model
    else:
        raise ValueError("Model should either be a string or an instance of sklearn.linear_model BaseEstimator.")
    
    cur_x, y = _vol_split(train, vol_idx)
    model_instance.fit(cur_x.T, y.T)

    return model_instance.predict(cur_x.T).reshape(data_shape[0], data_shape[1], data_shape[2])

def _vol_split(train, vol_idx):
    
    mask = np.zeros(train.shape[0], dtype=bool)
    mask[vol_idx] = True
    cur_x = train[~mask]
    cur_x = cur_x.reshape((train.shape[0]-1)*train.shape[1], train.shape[2])

    # Center voxel of the selected block
    y = train[vol_idx, train.shape[1]//2, :]
    
    return cur_x, y

def _extract_3d_patches(arr, patch_radius, mask=None):
    
    if mask is not None and arr.shape[:-1] != mask.shape:
        raise ValueError("Input array and mask must have the same shape for the first 3 dimensions.")
    
    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius, dtype=int)
    patch_size = 2 * patch_radius + 1

    dim = arr.shape[-1]

    all_patches = []

    # loop around and find the 3D patch for each direction
    for i in range(patch_radius[0], arr.shape[0] - patch_radius[0]):
        for j in range(patch_radius[1], arr.shape[1] - patch_radius[1]):
            for k in range(patch_radius[2], arr.shape[2] - patch_radius[2]):
                if mask is None or mask[i, j, k]:  # Only extract patch if no mask is provided or if it's True in the mask
                    ix1 = i - patch_radius[0]
                    ix2 = i + patch_radius[0] + 1
                    jx1 = j - patch_radius[1]
                    jx2 = j + patch_radius[1] + 1
                    kx1 = k - patch_radius[2]
                    kx2 = k + patch_radius[2] + 1

                    X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(np.prod(patch_size), dim)
                    all_patches.append(X)

    return np.array(all_patches).T

# TODO: masking param needs to be added here?
def patch2self(data, bvals, patch_radius=(0, 0, 0), model='ols',
               b0_threshold=50, out_dtype=None, alpha=1.0, verbose=False,
               b0_denoising=True, clip_negative_vals=False,
               shift_intensity=True, n_jobs=1):
    
    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius

    patch_radius = np.asarray(patch_radius, dtype=int)

    if not data.ndim == 4:
        raise ValueError("Patch2Self can only denoise on 4D arrays.", data.shape)

    if data.shape[3] < 10:
        warn("The input data has less than 10 3D volumes. Patch2Self may not give denoising performance.")

    if out_dtype is None:
        out_dtype = data.dtype

    b0_idx = np.argwhere(bvals <= b0_threshold)
    dwi_idx = np.argwhere(bvals > b0_threshold)

    data_b0s = np.squeeze(np.take(data, b0_idx, axis=3))
    data_dwi = np.squeeze(np.take(data, dwi_idx, axis=3))

    denoised_b0s = np.empty(data_b0s.shape, dtype=out_dtype)
    denoised_dwi = np.empty(data_dwi.shape, dtype=out_dtype)
    denoised_arr = np.empty(data.shape, dtype=out_dtype)

    if verbose:
        t1 = time.time()

    if data_b0s.ndim == 3 or not b0_denoising:
        if verbose:
            print("b0 denoising skipped...")
        denoised_b0s = data_b0s
    else:
        train_b0 = _extract_3d_patches(np.pad(data_b0s, ((patch_radius[0], patch_radius[0]),
                                                         (patch_radius[1], patch_radius[1]),
                                                         (patch_radius[2], patch_radius[2]),
                                                         (0, 0)), mode='constant'),
                                       patch_radius=patch_radius)
        
        results_b0 = Parallel(n_jobs=n_jobs)(
            delayed(_vol_denoise)(train_b0, vol_idx, model, data_b0s.shape, alpha=alpha) 
            for vol_idx in tqdm(range(data_b0s.shape[3]), desc="B0 Volumes", leave=False)
        )

        for vol_idx, result in enumerate(results_b0):
            denoised_b0s[..., vol_idx] = result
            if verbose:
                print("Denoised b0 Volume: ", vol_idx)

    train_dwi = _extract_3d_patches(np.pad(data_dwi, ((patch_radius[0], patch_radius[0]),
                                                      (patch_radius[1], patch_radius[1]),
                                                      (patch_radius[2], patch_radius[2]),
                                                      (0, 0)), mode='constant'),
                                    patch_radius=patch_radius)
    results_dwi = Parallel(n_jobs=n_jobs)(
        delayed(_vol_denoise)(train_dwi, vol_idx, model, data_dwi.shape, alpha=alpha) 
        for vol_idx in tqdm(range(data_dwi.shape[3]), desc="DWI Volumes", leave=False)
    )

    for vol_idx, result in enumerate(results_dwi):
        denoised_dwi[..., vol_idx] = result
        if verbose:
            print("Denoised DWI Volume: ", vol_idx)

    if verbose:
        t2 = time.time()
        print('Total time taken for Patch2Self: ', t2-t1, " seconds")

    if data_b0s.ndim == 3:
        denoised_arr[:, :, :, b0_idx[0][0]] = denoised_b0s
    else:
        for i, idx in enumerate(b0_idx):
            denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_b0s[..., i])

    for i, idx in enumerate(dwi_idx):
        denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_dwi[..., i])

    if shift_intensity and not clip_negative_vals:
        for i in range(0, denoised_arr.shape[3]):
            shift = np.min(data[..., i]) - np.min(denoised_arr[..., i])
            denoised_arr[..., i] = denoised_arr[..., i] + shift

    elif clip_negative_vals and not shift_intensity:
        denoised_arr.clip(min=0, out=denoised_arr)

    elif clip_negative_vals and shift_intensity:
        msg = 'Both `clip_negative_vals` and `shift_intensity` cannot be True.'
        msg += ' Defaulting to `clip_negative_vals`...'
        warn(msg)
        denoised_arr.clip(min=0, out=denoised_arr)

    return denoised_arr
