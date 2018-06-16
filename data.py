import numpy as np
import sigpy as sp
import sigpy.mri as mr


def load(data):
    if data == 'ute':
        ksp = np.load('data/ute/ksp.npy')
        coord = np.load('data/ute/coord.npy')
        weights = 1
    elif data == 'ute3d':
        ksp = np.load('data/ute3d/ksp.npy')[:, :1000]
        coord = np.load('data/ute3d/coord.npy')[:1000]
        weights = 1
    elif data == 'liver':
        ksp = np.load('data/liver/ksp.npy')
        coord = np.load('data/liver/coord.npy')
        weights = 1
    elif data == 'spiral':
        ksp = np.load('data/spiral/ksp.npy')
        coord = np.load('data/spiral/coord.npy')
        weights = 1  
    elif data == 'knee':
        ksp = np.load('data/knee/ksp.npy')
        coord = None
        ksp /= sp.util.rss(sp.fft.ifft(ksp, axes=[-1, -2])).max()
        weights = mr.samp.poisson(ksp.shape[1:], 8, calib=[24, 24])
    else:
        raise Exception('Invalid data: {}.'.format(data))

    return ksp, coord, weights
