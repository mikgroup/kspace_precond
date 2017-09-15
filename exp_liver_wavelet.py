#! /usr/bin/env python
import sigpy as sp
import mripy as mr
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Set parameters
accel = 4
ksp_calib_width = 30
mps_ker_width = 12
lamda_nlinv = 0
lamda = 0.0003

# Read kspace
ksp = sp.io.read_ra('data/liver/ksp.ra')
coord = sp.io.read_ra('data/liver/coord.ra')

ksp /= abs(ksp).max()
num_coils = ksp.shape[0]

# Simulate undersampling in kspace
ksp_under = ksp[:, ::accel, :]
coord_under = coord[::accel, :, :]

# Estimate maps
mps_ker_shape = [num_coils, mps_ker_width, mps_ker_width]
calib_shape = [num_coils, ksp_under.shape[1], ksp_calib_width]
mps_shape = [ksp.shape[0]] + [int(coord[..., i].max() - coord[..., i].min())
                              for i in range(ksp.ndim - 1)]

ksp_calib = sp.util.crop(ksp_under, calib_shape)
coord_calib = sp.util.crop(coord_under, calib_shape[1:] + [2])


nlinv_app = mr.app.NonlinearInversionRecon(ksp_calib, mps_ker_shape, lamda_nlinv,
                                           coord=coord_calib)
img_ker, mps_ker = nlinv_app.run()
mps = nlinv_app.kernels_to_maps(img_ker, mps_ker, mps_shape)

# Generate kspace preconditioner
precond = mr.precond.sense_kspace_precond(mps, lamda, coord=coord_under)

# Initialize app
wavelet_primal_app = mr.app.WaveletRecon(ksp_under, mps, lamda, coord=coord_under,
                                         save_iter_obj=True, save_iter_img=True)
wavelet_primal_dual_app = mr.app.WaveletPrimalDualRecon(ksp_under, mps, lamda,
                                                        coord=coord_under,
                                                        save_iter_obj=True,
                                                        save_iter_img=True)
wavelet_precond_app = mr.app.WaveletPrimalDualRecon(ksp_under, mps, lamda, precond=precond,
                                                    coord=coord_under,
                                                    save_iter_obj=True,
                                                    save_iter_img=True)

# Perform reconstructions
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
