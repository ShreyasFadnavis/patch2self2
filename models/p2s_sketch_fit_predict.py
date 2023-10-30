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

def _fit_denoising_model(train, vol_idx, model, alpha):
    """Fit the denoising model for a single volume."""
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
    
    return model_instance

def _predict_with_denoising_model(model_instance, train, vol_idx, data_shape):
    """Use the fitted model to predict the denoised volume."""
    cur_x, _ = _vol_split(train, vol_idx)
    return model_instance.predict(cur_x.T).reshape(data_shape[0], 
                                                   data_shape[1], 
                                                   data_shape[2])

def count_sketch(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([s, n])
    hashedIndices = np.random.choice(s, m, replace=True)
    # a m-by-1 {+1, -1} vector
    randSigns = np.random.choice(2, m, replace=True) * 2 - 1

    # flip the signs of 50% rows of A
    matrixA = matrixA * randSigns.reshape(m, 1)

    # this loop directly computes matrixC= S * matrixA
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[i] = np.sum(matrixA[idx], 0)

    return matrixC[:, np.newaxis, :]

class patch2self:

    def __init__(self, patch_radius=(0, 0, 0), model='ols', b0_threshold=50, 
                 out_dtype=None, alpha=1.0, verbose=False, b0_denoising=True, 
                 clip_negative_vals=False, shift_intensity=True, n_jobs=1, s=1000):
        
        self.patch_radius = patch_radius
        self.model = model
        self.b0_threshold = b0_threshold
        self.out_dtype = out_dtype
        self.alpha = alpha
        self.verbose = verbose
        self.b0_denoising = b0_denoising
        self.clip_negative_vals = clip_negative_vals
        self.shift_intensity = shift_intensity
        self.n_jobs = n_jobs
        self.s = s
        
        self.fitted_models_b0 = []
        self.fitted_models_dwi = []

    def fit(self, data, bvals):
        # Check data dimensions
        if not data.ndim == 4:
            raise ValueError("Patch2Self can only denoise on 4D arrays.", data.shape)

        if data.shape[3] < 10:
            warn("The input data has less than 10 3D volumes. Patch2Self may not give optimal denoising performance.")
        
        if self.out_dtype is None:
            self.out_dtype = data.dtype
        
        b0_idx = np.argwhere(bvals <= self.b0_threshold)
        dwi_idx = np.argwhere(bvals > self.b0_threshold)
        
        data_b0s = np.squeeze(np.take(data, b0_idx, axis=3))
        data_dwi = np.squeeze(np.take(data, dwi_idx, axis=3))
        
        # Extract patches for B0
        if data_b0s.ndim != 3 and self.b0_denoising:
            self.train_b0 = _extract_3d_patches(np.pad(data_b0s, ((self.patch_radius[0], self.patch_radius[0]),
                                                         (self.patch_radius[1], self.patch_radius[1]),
                                                         (self.patch_radius[2], self.patch_radius[2]),
                                                         (0, 0)), mode='constant'),
                                       patch_radius=self.patch_radius)
            self.fitted_models_b0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_denoising_model)(self.train_b0, vol_idx, self.model, alpha=self.alpha) 
                for vol_idx in tqdm(range(data_b0s.shape[3]), desc="Fitting models for B0 volumes", leave=False)
            )
            
        # Extract patches for DWI
        self.train_dwi = _extract_3d_patches(np.pad(data_dwi, ((self.patch_radius[0], self.patch_radius[0]),
                                                      (self.patch_radius[1], self.patch_radius[1]),
                                                      (self.patch_radius[2], self.patch_radius[2]),
                                                      (0, 0)), mode='constant'),
                                    patch_radius=self.patch_radius)
        
        sampled_train_dwi = count_sketch(np.squeeze(self.train_dwi).T, self.s).T
        self.fitted_models_dwi = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_denoising_model)(sampled_train_dwi, vol_idx, self.model, alpha=self.alpha) 
            for vol_idx in tqdm(range(data_dwi.shape[3]), desc="Fitting models for DWI volumes", leave=False)
        )

    def predict(self, data, bvals):
        # Initialize arrays for output
        b0_idx = np.argwhere(bvals <= self.b0_threshold)
        dwi_idx = np.argwhere(bvals > self.b0_threshold)
        
        data_b0s = np.squeeze(np.take(data, b0_idx, axis=3))
        data_dwi = np.squeeze(np.take(data, dwi_idx, axis=3))
        
        denoised_b0s = np.empty(data_b0s.shape, dtype=self.out_dtype)
        denoised_dwi = np.empty(data_dwi.shape, dtype=self.out_dtype)
        denoised_arr = np.empty(data.shape, dtype=self.out_dtype)
        
        # Predict B0s
        if self.fitted_models_b0:
            results_b0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_with_denoising_model)(fitted_model, self.train_b0, vol_idx, data_b0s.shape)
                for vol_idx, fitted_model in enumerate(self.fitted_models_b0)
            )
            for vol_idx, result in enumerate(results_b0):
                denoised_b0s[..., vol_idx] = result
        
        # Predict DWIs
        results_dwi = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_with_denoising_model)(fitted_model, self.train_dwi, vol_idx, data_dwi.shape)
            for vol_idx, fitted_model in enumerate(self.fitted_models_dwi)
        )
        for vol_idx, result in enumerate(results_dwi):
            denoised_dwi[..., vol_idx] = result

        if data_b0s.ndim == 3:
            denoised_arr[:, :, :, b0_idx[0][0]] = denoised_b0s
        else:
            for i, idx in enumerate(b0_idx):
                denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_b0s[..., i])

        for i, idx in enumerate(dwi_idx):
            denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_dwi[..., i])

        if self.shift_intensity and not self.clip_negative_vals:
            for i in range(0, denoised_arr.shape[3]):
                shift = np.min(data[..., i]) - np.min(denoised_arr[..., i])
                denoised_arr[..., i] = denoised_arr[..., i] + shift

        elif self.clip_negative_vals and not self.shift_intensity:
            denoised_arr.clip(min=0, out=denoised_arr)

        return denoised_arr
