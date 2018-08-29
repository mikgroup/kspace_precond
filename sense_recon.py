#! /usr/bin/env python
import sigpy as sp
import sigpy.mri as mr
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sense_recon(ksp, lamda, device=-1, coord=None):

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
    D = sp.linop.Multiply(ksp.shape, precond)
    ksp_d = sp.util.move(ksp, device=device)
    def proxfc_D(alpha, x):
        with sp.util.Device(device):
            return (x - alpha * precond * ksp_d) / (1 + alpha * precond)

    # Create apps
    cg_app = mr.app.SenseRecon(
        ksp, mps, coord=coord, device=device, save_objective_values=True)
    pdhg_app = mr.app.SenseRecon(
        ksp, mps, coord=coord,
        alg_name='PrimalDualHybridGradient', device=device, save_objective_values=True)
    precond_pdhg_app = mr.app.SenseRecon(
        ksp, mps, coord=coord, D=D, proxfc_D=proxfc_D,
        alg_name='PrimalDualHybridGradient', device=device, save_objective_values=True)

    # Run recons
    cg_app.run()
    pdhg_app.run()
    precond_pdhg_app.run()

    # Plot
    plt.figure(),
    plt.semilogy(cg_app.objective_values)
    plt.semilogy(pdhg_app.objective_values)
    plt.semilogy(precond_pdhg_app.objective_values)
    plt.legend(['Conjugate Gradient',
                'Primal Dual Hybrid Gradient',
                'Primal Dual Hybrid Gradient with Fourier preconditioning'])
    plt.ylabel('Objective Value')
    plt.xlabel('Iteration')
    plt.title(r"$\ell 1$-Wavelet Regularized Reconstruction")
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='L1 wavelet reconstruction.')

    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--lamda', type=float, default=0.001, help='Regularization parameter.')
    parser.add_argument('--coord_file', type=str)
    parser.add_argument('ksp_file', type=str)
    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    if args.coord_file:
        coord = np.load(args.coord_file)

    sense_recon(ksp, args.lamda, coord=coord, device=args.device)
