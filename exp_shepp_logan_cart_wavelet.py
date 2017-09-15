#! /usr/bin/env python
import sigpy as sp
import mripy as mr
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Set parameters
img_shape = [128, 128]
mps_shape = [8, 128, 128]
calib_shape = [20, 20]
accel = 8
max_iter = 100
lamda = 0.001

# Simulate kspace and image
img = mr.sim.shepp_logan(img_shape)
mps = mr.sim.birdcage_maps(mps_shape)

mask = mr.samp.poisson(img_shape, accel, calib=calib_shape)

A = mr.linop.Sense(mps, mask=mask)
ksp = A(img)

# Generate preconditioner

precond = mr.precond.sense_kspace_precond(mps, mask=mask)

sp.view.View(precond * mask)

# Initialize apps
wavelet_primal_app = mr.app.WaveletRecon(ksp, mps, lamda,
                                         save_iter_obj=True, save_iter_img=True)
wavelet_primal_dual_app = mr.app.WaveletPrimalDualRecon(ksp, mps, lamda,
                                                        save_iter_obj=True,
                                                        save_iter_img=True)
wavelet_precond_app = mr.app.WaveletPrimalDualRecon(ksp, mps, lamda,
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
