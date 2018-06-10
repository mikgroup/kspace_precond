import numpy as np
import sigpy as sp
import sigpy.mri as mr


ksp_file = 'data/knee/ksp.npy'

ksp = np.load(ksp_file)
ksp /= sp.util.rss(sp.fft.ifft(ksp, axes=[-1, -2])).max()
weights = mr.samp.poisson(ksp.shape[1:], 8, calib=[24, 24])
ksp *= weights
coord = None
