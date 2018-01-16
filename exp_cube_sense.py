#! /usr/bin/env python
import matplotlib
matplotlib.use('tkagg')
import sigpy as sp
import mripy as mr
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Set parameters
lamda = 0.01
ksp_calib_width = 24
mps_ker_width = 12

ksp = sp.io.read_ra('data/cube/ksp.ra')
ksp /= abs(sp.util.rss(sp.util.ifftc(ksp, axes=(-1, -2)))).max()
num_coils = ksp.shape[0]

# Simulate undersampling in kspace
img_shape = ksp.shape[1:]
mask = sp.util.rss(ksp) > 1e-5
ksp *= mask
sp.view.View(mask)

# Estimate maps
mps_ker_shape = [num_coils, mps_ker_width, mps_ker_width]
ksp_calib_shape = [num_coils, ksp_calib_width, ksp_calib_width]

ksp_calib = sp.util.crop(ksp, ksp_calib_shape)

jsense_app = mr.app.JointSenseRecon(ksp_calib, mps_ker_shape, ksp.shape)
mps = jsense_app.run()

sp.view.View(mps)

# Generate kspace preconditioners
precond = mr.precond.sense_kspace_precond(mps, mask=mask)

sp.view.View(precond * mask)

# Initialize app
sense_app = mr.app.SenseRecon(ksp, mps, lamda,
                              save_iter_obj=True)
sense_primaldual_app = mr.app.SenseRecon(ksp, mps, lamda,
                                         alg_name='FirstOrderPrimalDual',
                                         save_iter_obj=True)
sense_precond_app = mr.app.SenseRecon(ksp, mps, lamda,
                                      alg_name='FirstOrderPrimalDual',
                                      dual_precond=precond,
                                      save_iter_obj=True)

# Run reconstructions
img = sense_app.run()
img_primaldual = sense_primaldual_app.run()
img_precond = sense_precond_app.run()

# Plot
plt.figure(),
plt.semilogy(sense_app.iter_obj)
plt.semilogy(sense_primaldual_app.iter_obj)
plt.semilogy(sense_precond_app.iter_obj)
plt.legend(['ConjGrad Primal Recon.', 'Primal Dual Recon.', 'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 2$ Regularized Reconstruction")
plt.show()

sp.view.View(np.stack([img, img_primaldual, img_precond]))
