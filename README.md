# fourier_precond

This repo contains scripts to produce experients with fourier preconditioning.

## Requirements

The [sigpy](https://github.com/mikgroup/sigpy.git) package is required, and can be installed via pip:
	
	pip install sigpy

## Example usage

	python l1_wavelet_recon.py data/ute/ksp.npy --coord_file data/ute/coord.npy
	python l1_wavelet_recon.py data/spiral/ksp.npy --coord_file data/spiral/coord.npy
