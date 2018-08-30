#! /usr/bin/env python
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
import numpy as np
import argparse


def l1_wavelet_recon(ksp, lamda, device=-1, coord=None, max_iter=100):

    device = sp.util.Device(device)
    xp = device.xp
    # Estimate sensitivity maps
    jsense_app = mr.app.JsenseRecon(ksp, coord=coord, device=device)
    mps = jsense_app.run()

    # Normalize data
    if coord is None:
        ndim = ksp.ndim - 1
        weights = sp.util.rss(ksp) > 0
        ksp /= sp.util.rss(sp.fft.ifft(ksp, axes=range(-ndim, 0))).max()
    else:
        weights = 1
        ksp /= sp.util.rss(sp.nufft.nufft_adjoint(ksp, coord)).max()    

    # Calculate preconditioner
    sigma = mr.precond.fourier_diag_precond(mps, weights=weights, coord=coord, device=device)

    # Create apps
    fista_app = mr.app.L1WaveletRecon(
        ksp, mps, lamda=lamda, coord=coord, device=device,
        max_iter=max_iter, save_objective_values=True)
    pdhg_app = mr.app.L1WaveletRecon(
        ksp, mps, lamda=lamda, coord=coord, max_iter=max_iter,
        alg_name='PrimalDualHybridGradient', device=device, save_objective_values=True)
    precond_pdhg_app = mr.app.L1WaveletRecon(
        ksp, mps, lamda=lamda, coord=coord, sigma=sigma, max_iter=max_iter,
        alg_name='PrimalDualHybridGradient', device=device, save_objective_values=True)

    # Run recons
    fista_app.run()
    pdhg_app.run()
    precond_pdhg_app.run()

    # Plot
    pl.Image(xp.stack([fista_app.img, pdhg_app.img, precond_pdhg_app.img]))
    plt.figure(),
    plt.loglog(fista_app.objective_values)
    plt.loglog(pdhg_app.objective_values)
    plt.loglog(precond_pdhg_app.objective_values)
    plt.legend(['FISTA',
                'Primal Dual Hybrid Gradient',
                'Primal Dual Hybrid Gradient with Fourier preconditioning'])
    plt.ylabel('Objective Value')
    plt.xlabel('Iteration')
    plt.title(r"$\ell 1$-Wavelet Regularized Reconstruction")
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--lamda', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--coord_file', type=str)
    parser.add_argument('ksp_file', type=str)
    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    if args.coord_file:
        coord = np.load(args.coord_file)
    else:
        coord = None

    l1_wavelet_recon(ksp, args.lamda, coord=coord, device=args.device, max_iter=args.max_iter)
