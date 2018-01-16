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
lamda = 0.1

# Read kspace
ksp = sp.io.read_ra('data/ute/ksp.ra')
coord = sp.io.read_ra('data/ute/coord.ra')

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

ksp_calib = sp.util.crop(ksp_under, calib_shape, center=False)
coord_calib = sp.util.crop(coord_under, calib_shape[1:] + [2], center=False)

nlinv_app = mr.app.NonlinearGradRecon(ksp_calib, mps_ker_shape, lamda_nlinv,
                                      coord=coord_calib)
img_ker, mps_ker = nlinv_app.run()
mps = nlinv_app.kernels_to_maps(img_ker, mps_ker, mps_shape)

sp.view.View(mps)

# Generate kspace preconditioner
precond = mr.precond.sense_kspace_precond(mps, coord=coord_under)

# Initialize app
sense_primal_app = mr.app.SenseRecon(ksp_under, mps, lamda, coord=coord_under,
                                     save_iter_obj=True, save_iter_img=True)
sense_primal_dual_app = mr.app.SensePrimalDualRecon(ksp_under, mps, lamda,
                                                    coord=coord_under,
                                                    save_iter_obj=True, save_iter_img=True)
sense_precond_app = mr.app.SensePrimalDualRecon(ksp_under, mps, lamda, precond=precond,
                                                coord=coord_under,
                                                save_iter_obj=True, save_iter_img=True)

# Perform reconstructions
img_primal = sense_primal_app.run()
img_primal_dual = sense_primal_dual_app.run()
img_precond = sense_precond_app.run()

plt.figure(),
plt.semilogy(sense_primal_app.iter_obj)
plt.semilogy(sense_primal_dual_app.iter_obj)
plt.semilogy(sense_precond_app.iter_obj)
plt.legend(['ConjGrad Primal Recon.',
            'Primal Dual Recon.',
            'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 2$ Regularized Reconstruction")
plt.show()

sp.view.View(np.stack([sense_primal_app.iter_img,
                       sense_primal_dual_app.iter_img,
                       sense_precond_app.iter_img]))
