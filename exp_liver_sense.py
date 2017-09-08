#! /usr/bin/env python
import sigrec as sr
import sigrec_mri as mr
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

# Set parameters
accel = 4
max_iter = 10
calib_width = 30
ker_shape = [16, 16, 16]
lamda_nlinv = 0
lamda = 0.1

# Read kspace
ksp = sr.io.read_ra('data/liver/ksp.ra')
coord = sr.io.read_ra('data/liver/coord.ra')

ksp /= abs(ksp).max()

# Simulate undersampling in kspace
ksp_under = ksp[:, ::accel, :]
coord_under = coord[::accel, :, :]

# Estimate maps
calib_shape = [ksp_under.shape[0], ksp_under.shape[1], calib_width]
ksp_calib = sr.util.crop(ksp_under, calib_shape)
coord_calib = sr.util.crop(coord_under, calib_shape[1:] + [2])

img_ker, mps_ker = mr.nlinv_recon(ksp_calib, ker_shape, lamda_nlinv, coord=coord_calib)
mps_shape = [ksp.shape[0]] + [int(coord[..., i].max() - coord[..., i].min())
                              for i in range(ksp.ndim - 1)]
mps = mr.kernels_to_maps(img_ker, mps_ker, mps_shape)

sr.view.View(mps)

# Generate kspace preconditioner
precond = mr.sense_precond(mps, lamda, coord=coord)

sr.view.scatter(coord, abs(precond[0]))

# Perform reconstruction
img_rec, costs_rec = mr.sense_recon(ksp_under, mps, lamda,
                                    coord=coord,
                                    max_iter=max_iter, output_costs=True)

img_drec, costs_drec = mr.sense_dual_recon(ksp_under, mps, lamda,
                                           coord=coord,
                                           max_iter=max_iter, output_costs=True)

img_prec, costs_prec = mr.sense_dual_recon(ksp_under, mps, lamda,
                                           coord=coord,
                                           precond=precond,
                                           max_iter=max_iter, output_costs=True)

plt.figure(),
plt.semilogy(range(max_iter), costs_rec,
             range(max_iter), costs_drec,
             range(max_iter), costs_prec)
plt.legend(['SENSE Primal', 'SENSE Dual without Precond.', 'SENSE Dual with Precond.'])
plt.title(r'$\ell 1$-wavelet regularized reconstruction')
plt.show()
