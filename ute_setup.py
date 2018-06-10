import numpy as np


ksp_file = 'data/ute/ksp.npy'
coord_file = 'data/ute/coord.npy'

ksp = np.load(ksp_file)
coord = np.load(coord_file)
weights = 1
