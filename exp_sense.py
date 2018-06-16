#! /usr/bin/env python
import data
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot
import matplotlib.pyplot as plt
import logging
import numpy as np


def exp_sense(ksp, coord, weights, lamda, device):
    
    jsense_app = mr.app.JsenseRecon(ksp, coord=coord, device=device)
    mps = jsense_app.run()

    precond = mr.precond.sense_kspace_precond(
        mps, weights=weights, coord=coord, lamda=lamda, device=device)

    if coord is None:
        sp.plot.Image(precond, title='Preconditioner')
    else:
        sp.plot.Scatter(coord, precond, title='Preconditioner')

    primal_app = mr.app.SenseRecon(
        ksp, mps, lamda=lamda, coord=coord, device=device, save_objs=True)
    dual_app = mr.app.SenseRecon(
        ksp, mps, lamda=lamda, coord=coord,
        alg_name='DualConjugateGradient', device=device, save_objs=True)
    dual_precond_app = mr.app.SenseRecon(
        ksp, mps, lamda=lamda, coord=coord, dual_precond=precond,
        alg_name='DualConjugateGradient', device=device, save_objs=True)

    img_primal = primal_app.run()
    img_dual = dual_app.run()
    img_dual_precond = dual_precond_app.run()

    sp.plot.Image(np.stack([sp.util.move(img_primal),
                            sp.util.move(img_dual),
                            sp.util.move(img_dual_precond)]))

    plt.figure(),
    plt.semilogy(primal_app.objs)
    plt.semilogy(dual_app.objs)
    plt.semilogy(dual_precond_app.objs)
    plt.legend(['ConjGrad Recon.',
                'DualConjGrad Recon.',
                'DualConjGrad Recon. with Precond'])
    plt.ylabel('Objective Value')
    plt.xlabel('Iteration')
    plt.title(r"$\ell 2$ Regularized Reconstruction")
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SENSE Experiment.')

    parser.add_argument('--data', type=str, default='liver')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--lamda', type=float, default=0.1, help='Regularization.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    ksp, coord, weights = data.load(args.data)
    exp_sense(ksp, coord, weights, args.lamda, args.device)
