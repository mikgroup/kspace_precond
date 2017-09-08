#! /usr/bin/env python
import mrpy as mr
import matplotlib.pyplot as plt
import numpy as np
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)

img_shape = [128, 128]
mps_shape = [8, 128, 128]
ksp_shape = [64, 128]
max_iter = 100
lamda = 0.003

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


mr.io.write_ra('ksp.ra', ksp_calib.reshape([8, 64, 32, 1]).astype(np.complex64))
coord2 = np.concatenate([coord_calib[..., ::-1], np.zeros([64, 32, 1])], axis=-1)
mr.io.write_ra('coord.ra', coord2.reshape([64, 32, 3]).astype(np.complex64))

subprocess.run(['bart', 'nufft', '-i', '-d128:128:1', 'coord.ra', 'ksp.ra', 'cimg.ra'])
subprocess.run(['bart', 'fft', '7', 'cimg.ra', 'cksp.ra'])
subprocess.run(['bart', 'ecalib', '-m1', 'cksp.ra', 'mps.ra'])

mps = np.squeeze(mr.io.read_ra('mps.ra').astype(np.complex))




# img_ker, mps_ker = mr.app.nlinv_recon(ksp_calib, ker_shape, coord=coord_calib,
#                                       max_iter=30, max_inner_iter=30)
# mps = mr.app.kernels_to_maps(img_ker, mps_ker, mps_shape)
# mps *= (abs(img) > 1e-3)

# mr.view.Viewer(np.stack([mps, mps_orig, mps - mps_orig]))

precond = mr.app.sense_precond(mps, coord=coord, lamda=lamda)


img_rec, costs_rec = mr.app.sense_recon(ksp, mps, lamda,
                                        coord=coord,
                                        max_iter=max_iter, output_costs=True)

img_drec, costs_drec = mr.app.sense_dual_recon(ksp, mps, lamda,
                                               coord=coord,
                                               max_iter=max_iter, output_costs=True)

img_prec, costs_prec = mr.app.sense_dual_recon(ksp, mps, lamda,
                                               precond=precond,
                                               coord=coord,
                                               max_iter=max_iter, output_costs=True)

plt.figure(),
plt.semilogy(range(max_iter), costs_rec,
             range(max_iter), costs_drec,
             range(max_iter), costs_prec)
plt.legend(['FISTA', 'Primal Dual Recon.', 'Primal Dual Recon. with Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 2$ Regularized Reconstruction")
plt.show()

mr.view.Viewer(np.stack([img_rec, img_drec, img_prec]))
