#! /usr/bin/env python
import sigpy as sp
import mripy as mr
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

img_shape = [128, 128]
mps_shape = [8, 128, 128]
ksp_shape = [64, 128]
max_iter = 100
lamda = 0.003

img = mr.sim.shepp_logan(img_shape)
mps = mr.sim.birdcage_maps(mps_shape)
coord = mr.samp.radial(ksp_shape, img_shape)

A = mr.linop.Sense(mps, coord=coord)

ksp = A(img)

sp.view.View(mps)

# Generate preconditioner

precond = mr.precond.sense_kspace_precond(mps, coord=coord, lamda=lamda)

sp.view.View(precond)

# Initialize app
sense_primal_app = mr.app.SenseRecon(ksp, mps, lamda, coord=coord,
                                     save_iter_obj=True, save_iter_img=True)
sense_primal_dual_app = mr.app.SensePrimalDualRecon(ksp, mps, lamda, coord=coord,
                                                    save_iter_obj=True, save_iter_img=True)
sense_precond_app = mr.app.SensePrimalDualRecon(ksp, mps, lamda, precond=precond, coord=coord,
                                                save_iter_obj=True, save_iter_img=True)

# Perform reconstructions
sense_primal_app.run()
sense_primal_dual_app.run()
sense_precond_app.run()

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
