#! /usr/bin/env python
import matplotlib
matplotlib.use('tkagg')
import sigpy as sp
import mripy as mr
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Set parameters
ksp_calib_width = 1000
mps_ker_width = 12
lamda = 0.0003
max_iter = 30

# Read kspace
ksp = sp.io.read_ra('data/spiral/ksp.ra')
coord = sp.io.read_ra('data/spiral/coord.ra')

ksp /= abs(ksp).max()
num_coils = ksp.shape[0]

# Simulate undersampling in kspace
ksp_under = ksp
coord_under = coord

# Estimate maps
mps_ker_shape = [num_coils, mps_ker_width, mps_ker_width]
calib_shape = [num_coils, ksp_under.shape[1], ksp_calib_width]
mps_shape = [ksp.shape[0]] + [int(coord[..., i].max() - coord[..., i].min())
                              for i in range(ksp.ndim - 1)]
print(mps_shape)

ksp_calib = sp.util.crop(ksp_under, calib_shape, center=False)
coord_calib = sp.util.crop(coord_under, calib_shape[1:] + [2], center=False)


jsense_app = mr.app.JointSenseRecon(ksp_calib, mps_ker_shape, mps_shape,
                                    coord=coord_calib)
mps = jsense_app.run()

sp.view.View(mps)

# Generate kspace preconditioner
dual_precond = mr.precond.sense_kspace_precond(mps, lamda, coord=coord_under)

# Initialize app
wavelet_primal_app = mr.app.WaveletRecon(ksp_under, mps, lamda, coord=coord_under,
                                         save_iter_obj=True, save_iter_var=True,
                                         max_iter=max_iter)
wavelet_primal_dual_app = mr.app.WaveletRecon(ksp_under, mps, lamda,
                                              coord=coord_under,
                                              alg_name="FirstOrderPrimalDual",
                                              save_iter_obj=True,
                                              save_iter_var=True,
                                              max_iter=max_iter)
wavelet_precond_app = mr.app.WaveletRecon(ksp_under, mps, lamda, dual_precond=dual_precond,
                                          coord=coord_under,
                                          alg_name="FirstOrderPrimalDual",
                                          save_iter_obj=True,
                                          save_iter_var=True,
                                          max_iter=max_iter)

# Perform reconstructions
wavelet_primal_app.run()
wavelet_primal_dual_app.run()
wavelet_dcf_precond_app.run()
wavelet_precond_app.run()

plt.figure(),
plt.semilogy(wavelet_primal_app.iter_obj)
plt.semilogy(wavelet_primal_dual_app.iter_obj)
plt.semilogy(wavelet_dcf_precond_app.iter_obj)
plt.semilogy(wavelet_precond_app.iter_obj)
plt.legend(['FISTA Recon.',
            'Primal Dual Recon.',
            'Primal Dual Recon. with Proposed Precond'])
plt.ylabel('Objective Value')
plt.xlabel('Iteration')
plt.title(r"$\ell 1$-Wavelet Regularized Reconstruction")
plt.show()

sp.view.View(np.stack([wavelet_primal_app.iter_var,
                       wavelet_primal_dual_app.iter_var,
                       wavelet_precond_app.iter_var]))
sp.io.write_ra('spiral_wavelet_fista.ra', np.stack(wavelet_primal_app.iter_var))
sp.io.write_ra('spiral_wavelet_primal_dual.ra', np.stack(wavelet_primal_dual_app.iter_var))
sp.io.write_ra('spiral_wavelet_precond.ra', np.stack(wavelet_precond_app.iter_var))
