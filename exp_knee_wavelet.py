#! /usp/bin/env python
import matplotlib
matplotlib.use('tkagg')
import sigpy as sp
import mripy as mr
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Set parameters
accel = 8
lamda = 0.001
lamda_nlinv = 0
ksp_calib_width = 24
mps_ker_width = 12

ksp = sp.io.read_ra('data/knee/ksp.ra')
ksp /= abs(sp.util.rss(sp.util.ifftc(ksp, axes=(-1, -2)))).max()
num_coils = ksp.shape[0]

# Simulate undersampling in kspace
img_shape = ksp.shape[1:]
# mask = mr.samp.poisson(img_shape, accel, calib=[ksp_calib_width, ksp_calib_width],
#                        dtype=ksp.dtype)
mask = sp.io.read_ra('mask.ra').astype(ksp.dtype).reshape(ksp.shape[-2:])
print(mask.shape, ksp.shape)
sp.view.View(mask)

ksp_under = ksp * mask

# Estimate maps
mps_ker_shape = [num_coils, mps_ker_width, mps_ker_width]
ksp_calib_shape = [num_coils, ksp_calib_width, ksp_calib_width]

ksp_calib = sp.util.crop(ksp_under, ksp_calib_shape)

jsense_app = mr.app.JointSenseRecon(ksp_calib, mps_ker_shape, ksp.shape)
mps = jsense_app.run()

sp.view.View(mps)

# Generate kspace preconditioner
precond = mr.precond.sense_kspace_precond(mps, mask=mask)

sp.view.View(precond * mask)

# Initialize apps
wavelet_primal_app = mr.app.WaveletRecon(ksp_under, mps, lamda,
                                         save_iter_obj=True, save_iter_img=True)
wavelet_primal_dual_app = mr.app.WaveletPrimalDualRecon(ksp_under, mps, lamda,
                                                        save_iter_obj=True,
                                                        save_iter_img=True)
wavelet_precond_app = mr.app.WaveletPrimalDualRecon(ksp_under, mps, lamda,
                                                    precond=precond,
                                                    save_iter_obj=True,
                                                    save_iter_img=True)

# Run reconstructions
wavelet_primal_app.run()
wavelet_primal_dual_app.run()
wavelet_precond_app.run()

plt.figure(),
plt.semilogy(wavelet_primal_app.iter_obj)
plt.semilogy(wavelet_primal_dual_app.iter_obj)
plt.semilogy(wavelet_precond_app.iter_obj)
plt.legend(['FISTA Recon.', 'Primal Dual Recon.', 'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 1$-Wavelet Regularized Reconstruction")
plt.show()

sp.view.View(np.stack([wavelet_primal_app.iter_img,
                       wavelet_primal_dual_app.iter_img,
                       wavelet_precond_app.iter_img]))
