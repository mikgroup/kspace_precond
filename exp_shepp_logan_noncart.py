#! /usr/bin/env python
import matplotlib
matplotlib.use('tkagg')
import mrpy as mr
import numpy as np
import matplotlib.pyplot as plt

img_shape = [128, 128]
mps_shape = [8, 128, 128]
ksp_shape = [128, 128]

img = mr.sim.gen_shepp_logan(img_shape)
mps = mr.sim.gen_maps(mps_shape)
coord = mr.samp.radial(ksp_shape, img_shape)

precond = mr.app.sense_precond.sense_precond(mps, coord=coord)

precond2 = mr.app.sense_precond.sense_precond(np.ones(mps_shape, dtype=np.complex),
                                              coord=coord)

_, dcf = np.mgrid[:128, :0.5:0.5 / 128]

A = mr.app.sense_model.SENSE(mps, coord=coord)

ksp = A(img)

img_adj = A.H(ksp)
img_adj /= abs(img_adj).max()

img_dcf = A.H(ksp * dcf)
img_dcf /= abs(img_dcf).max()

img_precond2 = A.H(ksp / (precond2 + 1e-11))
img_precond2 /= abs(img_precond2).max()

img_precond = A.H(ksp / (precond + 1e-11))
img_precond /= abs(img_precond).max()

mr.view.Viewer(np.concatenate([img, img_dcf, img_precond2, img_precond], axis=-1))
