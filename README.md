# Patch2Self2 (P2S2)
Contains the submission code for CVPR 2024

The codebase has been divided into 3 parts:
- 1) models - contains the main P2S2 code.
- 2) notebooks - contains the jupyter notebooks to reproduce all the results from the paper.
- 3) matrix_sketching - contains a separate module for matrix sketching which is general purpose and can find applications beyond P2S2.

For the sake of simplicity and complete reproducibility, each example notebook has the P2S2 code initialized within it. The reason for doing so is because, we needed to extract different things such as regression coefficients, leverage scores for different iterations, etc. which were solely for analysis required in the paper. These tools may not be useful to a general purpose user who wants to denoise the data.

API of P2S2:
```
from models.patch2self2 import patch2self as p2s2
from dipy.io.image import load_nifti
import numpy as np
import matplotlib.pyplot as plt

data, affine = load_nifti('data.nii.gz')
bvals = np.loadtxt('bval')

denoised_data = p2s2(data, bvals, sketching_method='leverage_scores', sketch_size=50000)
```
All the data denoised in the paper were denoised in the above manner. We include the denoised datasets in supplementary material.

Example Notebooks:

1) `notebooks/lev_plots_maps_3D.ipynb`: contains the example plots of 3D leverage score maps. The leverage scores were computed as shown on the PPMI data in: `notebooks/leverage_score_plot.ipynb`

The leverage scores which is a 1D array by default was then reshaped to a 3D array and saved to a nifti file with the same affine as the actual data.

2) `notebooks/leverage_score_plot.ipynb`: Shows the leverage scores as a line plot visualization on the PPMI dataset.
3) `notebooks/CSD_Spherical_Harmonics_Comparison.ipynb`: Shows the fiber orientation distributions (FODs) obtained by fitting the Stanford HARDI data with the Constrained Spherical Deconvolution model from DIPY.
4) `notebooks/Ground_Truth_P2S_Comparison.ipynb`: Contains the comparison of P2S with P2S2 on simulated data with a sketch size of 10000. It also contains code to reproduce the scatter plots at different SNRs shown in the paper along with the corresponding $R^2$ scores shown in Table. 1 of the main text.
5) `notebooks/CC_prob_tracking_fbc.ipynb`: Contains the code to perform tracking and Fiber-to-Bundle Coherence quantification.
6) `notebooks/p2s_sketcking_comparisons_10iterations.ipynb`: Contains the comparison of different sketching methods present in the `matrix_sketching` module repeated 10 times to capture the variance of the different sketching methods.
7) `notebooks/SpeedUp_Timing_Comparisons.ipynb`: reproduces the plots for comparing speedup and the actual time in seconds in comparison to P2S.
