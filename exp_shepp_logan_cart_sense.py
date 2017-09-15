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
calib_shape = [16, 16]
accel = 8
max_iter = 50
lamda = 0.001

# Simulate kspace and image
img = mr.sim.shepp_logan(img_shape)
mps = mr.sim.birdcage_maps(mps_shape)

mask = mr.samp.poisson(img_shape, accel, calib=calib_shape)

A = mr.linop.Sense(mps, mask=mask)
ksp = A(img)

# Generate preconditioner

precond = mr.precond.sense_kspace_precond(mps, mask=mask, lamda=lamda)

sp.view.View(precond * mask)

# Initialize app
sense_app = mr.app.SenseRecon(ksp, mps, lamda,
                              save_iter_obj=True, save_iter_img=True)
sense_primal_dual_app = mr.app.SensePrimalDualRecon(ksp, mps, lamda,
                                                    save_iter_obj=True, save_iter_img=True)
sense_precond_app = mr.app.SensePrimalDualRecon(ksp, mps, lamda, precond=precond,
                                                save_iter_obj=True, save_iter_img=True)

# Run reconstructions
img_rec = sense_app.run()
img_dual_rec = sense_primal_dual_app.run()
img_precond_rec = sense_precond_app.run()

plt.figure(),
plt.semilogy(sense_app.iter_obj)
plt.semilogy(sense_primal_dual_app.iter_obj)
plt.semilogy(sense_precond_app.iter_obj)
plt.legend(['ConjGrad Primal Recon.', 'Primal Dual Recon.', 'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 2$ Regularized Reconstruction")
plt.show()

sp.view.View(np.stack([sense_app.iter_img,
                       sense_primal_dual_app.iter_img,
                       sense_precond_app.iter_img]))
