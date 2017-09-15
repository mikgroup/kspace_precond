#! /usr/bin/env python
import sigpy as sp
import mripy as mr
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Set parameters
accel = 8
lamda = 0.01
lamda_nlinv = 0
ksp_calib_width = 24
mps_ker_width = 12

ksp = sp.io.read_ra('data/knee/ksp.ra')
ksp /= abs(sp.util.rss(sp.util.ifftc(ksp, axes=(-1, -2)))).max()
num_coils = ksp.shape[0]

# Simulate undersampling in kspace
img_shape = ksp.shape[1:]
mask = mr.samp.poisson(img_shape, accel, calib=[ksp_calib_width, ksp_calib_width],
                       dtype=ksp.dtype)

ksp_under = ksp * mask

# Estimate maps
mps_ker_shape = [num_coils, mps_ker_width, mps_ker_width]
ksp_calib_shape = [num_coils, ksp_calib_width, ksp_calib_width]

ksp_calib = sp.util.crop(ksp_under, ksp_calib_shape)

nlinv_app = mr.app.NonlinearInversionRecon(ksp_calib, mps_ker_shape, lamda_nlinv)
img_ker, mps_ker = nlinv_app.run()
mps = nlinv_app.kernels_to_maps(img_ker, mps_ker, ksp.shape)

sp.view.View(mps)

# Generate kspace preconditioner
precond = mr.precond.sense_kspace_precond(mps, mask=mask)

sp.view.View(precond * mask)

# Initialize app
sense_primal_app = mr.app.SenseRecon(ksp_under, mps, lamda,
                                     save_iter_obj=True, save_iter_img=True)
sense_primal_dual_app = mr.app.SensePrimalDualRecon(ksp_under, mps, lamda,
                                                    save_iter_obj=True, save_iter_img=True)
sense_precond_app = mr.app.SensePrimalDualRecon(ksp_under, mps, lamda, precond=precond,
                                                save_iter_obj=True, save_iter_img=True)

# Run reconstructions
sense_primal_app.run()
sense_primal_dual_app.run()
sense_precond_app.run()

# Plot
plt.figure(),
plt.semilogy(sense_primal_app.iter_obj)
plt.semilogy(sense_primal_dual_app.iter_obj)
plt.semilogy(sense_precond_app.iter_obj)
plt.legend(['ConjGrad Primal Recon.', 'Primal Dual Recon.', 'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 2$ Regularized Reconstruction")
plt.show()

sp.view.View(np.stack([sense_primal_app.iter_img,
                       sense_primal_dual_app.iter_img,
                       sense_precond_app.iter_img]))
