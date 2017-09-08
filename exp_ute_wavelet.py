#! /usr/bin/env python
import sigrec as sr
import sigrec_mri as mr
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

# Set parameters
accel = 4
max_iter = 50
calib_width = 30
mps_ker_width = 16
lamda_nlinv = 0
lamda = 0.0001

# Read kspace
ksp = sr.io.read_ra('data/ute/ksp.ra')
coord = sr.io.read_ra('data/ute/coord.ra')

ksp /= abs(ksp).max()
num_coils = ksp.shape[0]

# Simulate undersampling in kspace
ksp_under = ksp[:, ::accel, :]
coord_under = coord[::accel, :, :]

# Estimate maps
mps_ker_shape = [num_coils, mps_ker_width, mps_ker_width]
calib_shape = [num_coils, ksp_under.shape[1], calib_width]

ksp_calib = sr.util.crop(ksp_under, calib_shape, center=False)
coord_calib = sr.util.crop(coord_under, calib_shape[1:] + [2], center=False)

img_ker, mps_ker = mr.nlinv_recon(ksp_calib, mps_ker_shape, lamda_nlinv, coord=coord_calib)
mps_shape = [ksp.shape[0]] + [int(coord[..., i].max() - coord[..., i].min())
                              for i in range(ksp.ndim - 1)]

mps = mr.kernels_to_maps(img_ker, mps_ker, mps_shape)

sr.view.View(mps)

# Generate kspace preconditioner
precond = mr.sense_precond(mps, lamda, coord=coord_under)

sr.view.scatter(coord_under, abs(precond[0]))

# Perform reconstruction
img_rec, costs_rec = mr.wavelet_recon(ksp_under, mps, lamda,
                                      coord=coord_under,
                                      max_iter=max_iter, output_costs=True)

img_drec, costs_drec = mr.wavelet_dual_recon(ksp_under, mps, lamda,
                                             coord=coord_under,
                                             max_iter=max_iter, output_costs=True)

img_prec, costs_prec = mr.wavelet_dual_recon(ksp_under, mps, lamda,
                                             coord=coord_under,
                                             precond=precond,
                                             max_iter=max_iter, output_costs=True)

plt.figure(),
plt.semilogy(range(max_iter), costs_rec,
             range(max_iter), costs_drec,
             range(max_iter), costs_prec)
plt.legend(['SENSE Primal', 'SENSE Dual without Precond.', 'SENSE Dual with Precond.'])
plt.title(r'$\ell 1$-wavelet regularized reconstruction')
plt.show()

sr.view.View(img_prec)
