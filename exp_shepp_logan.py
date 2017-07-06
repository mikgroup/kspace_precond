#! /usr/bin/env python
import matplotlib
matplotlib.use('tkagg')
import mrpy as mr
import numpy as np

img_shape = [128, 128]
mps_shape = [8, 128, 128]
accel = 4

img = mr.sim.gen_shepp_logan(img_shape)
mps = mr.sim.gen_maps(mps_shape)

mask = mr.samp.poisson(img_shape, accel)

precond = mr.app.sense_precond.sense_precond(mps, mask=mask)

A = mr.app.sense_model.SENSE(mps, mask=mask)

ksp = A(img)

img_adj = A.H(ksp)
img_adj /= abs(img_adj).max()

img_precond = A.H(ksp / (precond + 1e-11))
img_precond /= abs(img_precond).max()

mr.view.Viewer(np.concatenate([img_adj, img_precond], axis=-1))
