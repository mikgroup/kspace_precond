#! /usr/bin/env python
import mrpy as mr
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

img_shape = [128, 128]
mps_shape = [8, 128, 128]
ksp_shape = [64, 128]
max_iter = 100
lamda = 0.001
lamda_linv = 0

img = mr.sim.shepp_logan(img_shape)
mps = mr.sim.birdcage_maps(mps_shape)
coord = mr.samp.radial(ksp_shape, img_shape)

A = mr.app.SENSE(mps, coord=coord)

ksp = A(img)

# Get maps
ker_shape = [8, 12, 12]
ksp_calib_shape = [8, 64, 32]
coord_calib_shape = [64, 32, 2]

ksp_calib = mr.util.crop(ksp, ksp_calib_shape, center=False)
coord_calib = mr.util.crop(coord, coord_calib_shape, center=False)

img_ker, mps_ker = mr.app.nlinv_recon(ksp_calib, ker_shape, lamda_linv,
                                      coord=coord_calib,
                                      max_iter=10, max_inner_iter=10)
mps = mr.app.kernels_to_maps(img_ker, mps_ker, mps_shape)

precond = mr.app.sense_precond(mps, coord=coord)


img_rec, costs_rec = mr.app.wavelet_recon(ksp, mps, lamda,
                                          coord=coord, wavelet='haar',
                                          max_iter=max_iter, output_costs=True)

img_drec, costs_drec = mr.app.wavelet_dual_recon(ksp, mps, lamda,
                                                 coord=coord, wavelet='haar',
                                                 max_iter=max_iter, output_costs=True)

img_prec, costs_prec = mr.app.wavelet_dual_recon(ksp, mps, lamda,
                                                 precond=precond,
                                                 coord=coord, wavelet='haar',
                                                 max_iter=max_iter, output_costs=True)

plt.figure(),
plt.semilogy(range(max_iter), costs_rec,
             range(max_iter), costs_drec,
             range(max_iter), costs_prec)
plt.legend(['FISTA', 'Primal Dual Recon.', 'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 1$-Wavelet Regularized Reconstruction")
plt.show()

mr.view.Viewer(np.stack([img_rec, img_drec, img_prec]))
