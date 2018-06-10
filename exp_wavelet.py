#! /usr/bin/env python

if __name__ == '__main__':
    import sigpy as sp
    import sigpy.mri as mr
    import sigpy.plot
    import matplotlib.pyplot as plt
    import logging
    import numpy as np
    import argparse
    
    parser = argparse.ArgumentParser(description='L1 Wavelet Experiment.')

    parser.add_argument('--data', type=str, default='ute')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--lamda', type=float, default=0.01, help='Regularization.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.data == 'ute':
        from ute_setup import *
    elif args.data == 'liver':
        from liver_setup import *
    elif args.data == 'knee':
        from knee_setup import *
    elif args.data == 'spiral':
        from spiral_setup import *        
    else:
        raise Exception('Invalid data: {}.'.format(args.data))

    jsense_app = mr.app.JsenseRecon(ksp, coord=coord, device=args.device)
    mps = jsense_app.run()

    precond = mr.precond.sense_kspace_precond(
        mps, weights=weights, coord=coord, device=args.device)

    if coord is None:
        sp.plot.Image(precond, title='Preconditioner')
    else:
        sp.plot.Scatter(coord, precond, title='Preconditioner')

    primal_app = mr.app.WaveletRecon(
        ksp, mps, lamda=args.lamda, coord=coord, device=args.device, save_objs=True)
    dual_app = mr.app.WaveletRecon(
        ksp, mps, lamda=args.lamda, coord=coord,
        alg_name='PrimalDualHybridGradient', device=args.device, save_objs=True)
    dual_precond_app = mr.app.WaveletRecon(
        ksp, mps, lamda=args.lamda, coord=coord, dual_precond=precond,
        alg_name='PrimalDualHybridGradient', device=args.device, save_objs=True)

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
    plt.legend(['ConjGrad Primal Recon.',
                'DualConjGrad Recon.',
                'DualConjGrad Recon. with Precond'])
    plt.ylabel('Objective Value')
    plt.xlabel('Iteration')
    plt.title(r"$\ell 2$ Regularized Reconstruction")
    plt.show()
