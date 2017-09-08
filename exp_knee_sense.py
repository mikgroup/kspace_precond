#! /usr/bin/env python
import sigrec as sr
import sigrec_mri as mr
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)

# Set parameters
accel = 8
max_iter = 50
lamda = 0.01
lamda_nlinv = 0

ksp = sr.io.read_ra('data/knee/ksp.ra')
ksp /= abs(sr.util.rss(sr.util.ifftc(ksp, axes=(-1, -2)))).max()

# Estimate maps
ker_shape = [ksp.shape[0], 12, 12]
calib_shape = [ksp.shape[0], 24, 24]
ksp_calib = sr.util.crop(ksp, calib_shape)
img_ker, mps_ker = mr.nlinv_recon(ksp_calib, ker_shape, lamda_nlinv)
mps = mr.kernels_to_maps(img_ker, mps_ker, ksp.shape)

sr.view.View(mps)

# Simulate undersampling in kspace
img_shape = mps.shape[1:]
mask = mr.samp.poisson(img_shape, accel, calib=calib_shape[1:],
                       dtype=ksp.dtype)

ksp_under = ksp * mask

# Generate kspace preconditioner
precond = mr.sense_precond(mps, mask=mask, lamda=lamda)

sr.view.View(precond)

# Perform reconstruction
img_rec, costs_rec = mr.sense_recon(ksp_under, mps, lamda,
                                      max_iter=max_iter, output_costs=True)

img_drec, costs_drec = mr.sense_dual_recon(ksp_under, mps, lamda,
                                             sigma=0.1, tau=10.0,
                                             max_iter=max_iter, output_costs=True)

img_prec, costs_prec = mr.sense_dual_recon(ksp_under, mps, lamda,
                                             precond=precond,
                                             sigma=0.1, tau=10.0,
                                             max_iter=max_iter, output_costs=True)

plt.figure(),
plt.semilogy(range(max_iter), costs_rec,
             range(max_iter), costs_drec,
             range(max_iter), costs_prec)
plt.legend(['FISTA', 'Primal Dual without Precond.', 'Primal Dual with Precond.'])
plt.title(r'$\ell 2$ regularized reconstruction')
plt.show()
