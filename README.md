# VaxNeRF

### [Paper]()

This is the official implementation of VaxNeRF (Voxel-Accelearated NeRF).

<div align="center">
<img src="https://user-images.githubusercontent.com/23403885/143438536-03946310-ca85-4b53-afb9-a118293eda1d.PNG" width="50%">
</div>


This codebase is implemented using [JAX](https://github.com/google/jax),
building on [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

VaxNeRF provides very fast training and slightly higher scores compared to original (Jax)NeRF!!

<div align="center">
<img src="https://user-images.githubusercontent.com/23403885/142990613-73889333-ec75-41f4-8c99-e568da7a6e1a.png" width="90%">
</div>

![fast](https://user-images.githubusercontent.com/23403885/143439496-5005831f-315c-48b9-a0d8-084be72470c3.PNG)


## Installation

Please see the README of [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).


## Quick start

### Training

```shell
# make a bounding volume voxel using Visual Hull
python visualhull.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --voxel_dir data/voxel_dil7/lego \
    --dilation 7 \
    --thresh 1. \
    --alpha_bkgd True

# train VaxNeRF
python train.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --voxel_dir data/voxel_dil7/lego \
    --train_dir logs/lego_vax_c800 \
    --num_coarse_samples 800 \
    --render_every 2500
```

### Evaluation

```shell
python eval.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --voxel_dir data/voxel_dil7/lego \
    --train_dir logs/lego_vax_c800 \
    --num_coarse_samples 800
```


## Try other NeRFs

**Original NeRF**

```shell
python train.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --train_dir logs/lego_c64f128 \
    --num_coarse_samples 64 \
    --num_fine_samples 128 \
    --render_every 2500
```

**VaxNeRF with hierarchical sampling**

```shell
# hierarchical sampling needs more dilated voxel
python visualhull.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --voxel_dir data/voxel_dil47/lego \
    --dilation 47 \
    --thresh 1. \
    --alpha_bkgd True

# train VaxNeRF
python train.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --voxel_dir data/voxel_dil47/lego \
    --train_dir logs/lego_vax_c64f128 \
    --num_coarse_samples 64 \
    --num_fine_samples 128 \
    --render_every 2500
```


## Option details

**Visual Hull**

- Use `--dilation 11` / `--dilation 51` for NSVF-Synthetic dataset for training VaxNeRF without / with hierarchical sampling.
- The following options were used for the `Lifestyle`, `Spaceship`, `Steamtrain` scenes (included in the NSVF dataset) because these datasets do not have alpha channel.
  - Lifestyle: `--thresh 0.95`, Spaceship: `--thresh 0.9`, Steamtrain: `--thresh 0.95`

**NeRFs**

- We used `--small_lr_at_first` option for **original** NeRF training on the `Robot` and `Spaceship` scenes to avoid local minimum. 


## Code modification from JaxNeRF

- You can see the main difference between (Jax)NeRF (`jaxnerf` branch) and VaxNeRF (`vaxnerf` branch) [here](https://github.com/naruya/VaxNeRF/compare/jaxnerf...vaxnerf)
- The `main` branch (derived from the `vaxnerf` branch) contains the following features.
  - Support for original NeRF
  - Support for VaxNeRF with hierarchical sampling
  - Support for the NSVF-Synthetic dataset
  - Visualization of number of sampling points evaluated by MLP (VaxNeRF)
  - Automatic choice of the number of sampling points to be evaluated (VaxNeRF)

## Citation

Please use the following bibtex for citations:

```
@article{
}
```

and also cite the original [NeRF](http://www.matthewtancik.com/nerf) paper and [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) implementation:

```
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}

@software{jaxnerf2020github,
  author = {Boyang Deng and Jonathan T. Barron and Pratul P. Srinivasan},
  title = {{JaxNeRF}: an efficient {JAX} implementation of {NeRF}},
  url = {https://github.com/google-research/google-research/tree/master/jaxnerf},
  version = {0.0},
  year = {2020},
}
```

## Acknowledgement
We'd like to express deep thanks to the inventors of [NeRF](http://www.matthewtancik.com/nerf) and [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

Have a good VaxNeRF'ed life!
