#! /usr/bin/env python
import data
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot
import matplotlib.pyplot as plt
import logging
import numpy as np


def exp_grid(ksp, coord, weights, device):

    sp.plot.Image(weights)
        
    jsense_app = mr.app.JsenseRecon(ksp, coord=coord, device=device)
    mps = jsense_app.run().asarray()
    sp.plot.Image(mps)
    
    jsense_thresh_app = mr.app.JsenseRecon(ksp, coord=coord, device=device, thresh=0.1)
    mps_thresh = jsense_thresh_app.run().asarray()
    sp.plot.Image(mps_thresh)

    ones = np.ones(mps.shape, mps.dtype) / mps.shape[0]**0.5
    precond = mr.precond.sense_kspace_precond(
        ones, weights=weights, coord=coord, device=device)
    
    sense_precond = mr.precond.sense_kspace_precond(
        mps, weights=weights, coord=coord, device=device)
    
    sense_precond_thresh = mr.precond.sense_kspace_precond(
        mps_thresh, weights=weights, coord=coord, device=device)

    device = sp.util.Device(device)
    xp = device.xp
    ksp = sp.util.move(ksp, device)
    mps = sp.util.move(mps, device)
    mps_thresh = sp.util.move(mps_thresh, device)
    with device:
        if coord is None:
            img_precond = xp.sum(xp.conj(mps) * sp.fft.ifft(ksp * precond, axes=[-1, -2]), axis=0)
            img_sense_precond = xp.sum(
                xp.conj(mps) * sp.fft.ifft(ksp * sense_precond, axes=[-1, -2]), axis=0)
            img_sense_precond_thresh = xp.sum(
                xp.conj(mps) * sp.fft.ifft(ksp * sense_precond_thresh, axes=[-1, -2]), axis=0)
        else:
            coord = sp.util.move(coord, device)
            img_precond = xp.sum(
                xp.conj(mps) * sp.nufft.nufft_adjoint(ksp, coord), axis=0)
            img_sense_precond = xp.sum(
                xp.conj(mps) * sp.nufft.nufft_adjoint(ksp * sense_precond, coord), axis=0)
            img_sense_precond_thresh = xp.sum(
                xp.conj(mps) * sp.nufft.nufft_adjoint(ksp * sense_precond_thresh, coord), axis=0)

            # print(ksp.shape)
            # dcf = sp.util.move(np.load('data/ute/dcf_for_4x.npy'), device)
            # print(dcf.shape)
            # img_dcf = xp.sum(
            #     xp.conj(mps_thresh) * sp.nufft.nufft_adjoint(ksp * dcf, coord), axis=0)

        sp.plot.Image(xp.stack([img_precond, img_sense_precond, img_sense_precond_thresh]))
        
    if coord is None:
        sp.plot.Image(precond, title='Preconditioner')
    else:
        sp.plot.Scatter(coord, precond, title='Preconditioner')
    
    if coord is None:
        sp.plot.Image(sense_precond, title='Preconditioner')
    else:
        sp.plot.Scatter(coord, sense_precond, title='Preconditioner')
        
    if coord is None:
        sp.plot.Image(sense_precond_thresh, title='Preconditioner')
    else:
        sp.plot.Scatter(coord, sense_precond_thresh, title='Preconditioner')



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Gridding Experiment.')

    parser.add_argument('--data', type=str, default='liver')
    parser.add_argument('--device', type=int, default=-1)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    ksp, coord, weights = data.load(args.data)
    exp_grid(ksp, coord, weights, args.device)
