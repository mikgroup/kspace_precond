#! /usr/bin/env python
import matplotlib
matplotlib.use('tkagg')
import mrpy as mr
import numpy as np
import matplotlib.pyplot as plt

img_shape = [128, 128]
mps_shape = [8, 128, 128]
ksp_shape = [256, 256]

img = mr.sim.gen_shepp_logan(img_shape)
mps = mr.sim.gen_maps(mps_shape)
coord = mr.samp.radial(ksp_shape, img_shape)

sigma = 0.1
noise = mr.util.randn(ksp_shape) * sigma

_, dcf = np.mgrid[:256, :0.5:0.5 / 256]

A = mr.app.sense_model.SENSE(mps, coord=coord)

ksp = A(img) + noise

img_rec = mr.app.sense_recon.sense_recon(ksp, mps, coord=coord)
img_rec /= abs(img_rec).max()

img_dcf = mr.app.sense_recon.sense_recon(ksp * dcf ** 0.5, mps, mask=dcf ** 0.5, coord=coord)
img_dcf /= abs(img_dcf).max()

mr.view.Viewer(np.concatenate([img_rec, img_dcf], axis=-1))
