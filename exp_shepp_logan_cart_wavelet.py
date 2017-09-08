#! /usr/bin/env python
import sigrec as sr
import sigrec_mri as mr
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)

# Set parameters
img_shape = [128, 128]
mps_shape = [8, 128, 128]
calib_shape = [24, 24]
accel = 6
max_iter = 50
lamda = 0.001
lamda_nlinv = 0

# Simulate kspace and image
img = mr.sim.shepp_logan(img_shape)
mps = mr.sim.birdcage_maps(mps_shape)

mask = mr.samp.poisson(img_shape, accel, calib=calib_shape)

precond = mr.sense_precond(mps, mask=mask)

A = mr.SENSE(mps, mask=mask)

ksp = A(img)

# Estimate maps
ker_shape = [8, 12, 12]
ksp_calib_shape = [8, 24, 24]

ksp_calib = sr.util.crop(ksp, ksp_calib_shape)

img_ker, mps_ker = mr.nlinv_recon(ksp_calib, ker_shape, lamda_nlinv)
mps = mr.kernels_to_maps(img_ker, mps_ker, mps_shape)

# Perform reconstructions
img_rec, costs_rec = mr.wavelet_recon(ksp, mps, lamda,
                                      max_iter=max_iter, output_costs=True)

img_drec, costs_drec = mr.wavelet_dual_recon(ksp, mps, lamda,
                                             max_iter=max_iter, output_costs=True,
                                             sigma=0.1, tau=10.0)

img_prec, costs_prec = mr.wavelet_dual_recon(ksp, mps, lamda,
                                             precond=precond,
                                             max_iter=max_iter, output_costs=True,
                                             sigma=0.1, tau=10.0)

plt.figure(),
plt.semilogy(range(max_iter), costs_rec,
             range(max_iter), costs_drec,
             range(max_iter), costs_prec)
plt.legend(['FISTA', 'Primal Dual without Precond.', 'Primal Dual with Precond.'])
plt.title(r'$\ell 1$-wavelet regularized reconstruction')
plt.show()
