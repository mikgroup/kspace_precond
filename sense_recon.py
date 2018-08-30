#! /usr/bin/env python
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sense_recon(ksp, lamda, device=-1, coord=None, max_iter=100):

    device = sp.util.Device(device)
    xp = device.xp
    # Estimate sensitivity maps
    jsense_app = mr.app.JsenseRecon(ksp, coord=coord, device=device)
    mps = jsense_app.run()

    # Normalize data
    if coord is None:
        ndim = ksp.ndim - 1
        ksp /= sp.util.rss(sp.fft.ifftc(ksp, axes=range(-ndim, 0))).max()
    else:
        ksp /= sp.util.rss(sp.nufft.nufft_adjoint(ksp, coord)).max()        

    # Calculate preconditioner
    precond = mr.precond.fourier_diag_precond(mps, coord=coord, device=device)
    P = sp.linop.Multiply(ksp.shape, precond)

    # Create apps
    cg_app = mr.app.SenseRecon(
        ksp, mps, coord=coord, device=device,
        max_iter=max_iter, save_objective_values=True)
    pdhg_app = mr.app.SenseRecon(
        ksp, mps, coord=coord, max_iter=max_iter,
        alg_name='PrimalDualHybridGradient', device=device, save_objective_values=True)
    precond_pdhg_app = mr.app.SenseRecon(
        ksp, mps, coord=coord, sigma=precond, max_iter=max_iter,
        alg_name='PrimalDualHybridGradient', device=device, save_objective_values=True)

    # Run recons
    cg_app.run()
    pdhg_app.run()
    precond_pdhg_app.run()

    # Plot
    pl.Image(xp.stack([cg_app.img, pdhg_app.img, precond_pdhg_app.img]))
    plt.figure(),
    plt.loglog(cg_app.objective_values)
    plt.loglog(pdhg_app.objective_values)
    plt.loglog(precond_pdhg_app.objective_values)
    plt.legend(['Conjugate Gradient',
                'Primal Dual Hybrid Gradient',
                'Primal Dual Hybrid Gradient with Fourier preconditioning'])
    plt.ylabel('Objective Value')
    plt.xlabel('Iteration')
    plt.title(r"Sense Reconstruction")
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='L1 wavelet reconstruction.')

    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--lamda', type=float, default=0.001)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--coord_file', type=str)
    parser.add_argument('ksp_file', type=str)
    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    if args.coord_file:
        coord = np.load(args.coord_file)

    sense_recon(ksp, args.lamda, coord=coord, device=args.device, max_iter=args.max_iter)
