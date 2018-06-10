import numpy as np


ksp_file = 'data/liver/ksp.npy'
coord_file = 'data/liver/coord.npy'

ksp = np.load(ksp_file)
coord = np.load(coord_file)
weights = 1
