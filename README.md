# VaxNeRF

### [Paper](http://arxiv.org/abs/2111.13112) | [Google Colab](https://colab.research.google.com/drive/1Hf5-2eI_E7954iThqm4_5Jsa3YBNnvFn?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hf5-2eI_E7954iThqm4_5Jsa3YBNnvFn?usp=sharing)

This is the official implementation of VaxNeRF (Voxel-Accelearated NeRF).  
VaxNeRF provides very fast training and slightly higher scores compared to original (Jax)NeRF!!

<div align="center">
<img src="https://user-images.githubusercontent.com/23403885/143438536-03946310-ca85-4b53-afb9-a118293eda1d.PNG" width="50%">
</div>

### Updates!

- **December 26, 2021**
  - We Vax'ed MipNeRF!
  - We achieved roughly 28 times faster training in terms of producing the final accuracy of the original NeRF.
  - See -> https://github.com/naruya/mipnerf
- **December 2, 2021**
  - We Vax'ed PlenOctrees! You can train NeRF-SH about 5x faster.
  - https://github.com/naruya/plenoctree (out of maintenance)

## 

<table>
<td colspan="4" align="center"><b>Visual Hull (10sec)</b></td>
<tr>
<td><img src="https://i.gyazo.com/1342ac36ff033a8e74782cd91b7c4239.gif"></td>
<td><img src="https://i.gyazo.com/2cc6adc464f1e003cf3075342d9c8b18.gif"></td>
<td><img src="https://i.gyazo.com/eff9e2fb34dbbf9a258d96644804b510.gif"></td>
<td><img src="https://i.gyazo.com/9b8c7ec37df96a01c84b369564101047.gif"></td>
</tr>
<td colspan="4" align="center"><b>NeRF (10min)</b></td>
</tr>
<td><img src="https://i.gyazo.com/a1ce01462f9c74b978084694a757b290.gif"></td>
<td><img src="https://i.gyazo.com/7367778a2edb41b8ec422768da552b7c.gif"></td>
<td><img src="https://i.gyazo.com/03b161cc28fae6698617f835ad84dcfa.gif"></td>
<td><img src="https://i.gyazo.com/ae8eb3c89c597817b493fca9f1c4d028.gif"></td>
</tr>
<td colspan="4" align="center"><b>VaxNeRF (10min)</b></td>
</tr>
<td><img src="https://i.gyazo.com/e006344d4132fbb2d1a0865f10f54a3c.gif"></td>
<td><img src="https://i.gyazo.com/2e7cbf2d8f6cc201cb0f8a049448626b.gif"></td>
<td><img src="https://i.gyazo.com/283238f208cb461b4ea915a4a758f357.gif"></td>
<td><img src="https://i.gyazo.com/f6496fd576fefea2552f87f0a36b7894.gif"></td>
</table>

## 

<div align="center">
<img src="https://user-images.githubusercontent.com/23403885/147459802-8871f92f-923d-437c-a0a8-79b6077985ab.png" width="90%">
<br clear="left">
(The results of <a href="https://github.com/naruya/mipnerf">Vax-MipNeRF</a> are also included in this figure.)
</div>

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
    --alpha_bkgd

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
# small `num_xx_samples` needs more dilated voxel (see our paper)
python visualhull.py \
    --config configs/demo \
    --data_dir data/nerf_synthetic/lego \
    --voxel_dir data/voxel_dil47/lego \
    --dilation 47 \
    --thresh 1. \
    --alpha_bkgd

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
- The following options were used
- Since the `Lifestyle`, `Spaceship`, `Steamtrain` scenes (included in the NSVF dataset) do not have alpha channel, please use following options **and** remove `--alpha_bkgd` option.
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
@article{kondo2021vaxnerf,
  title={VaxNeRF: Revisiting the Classic for Voxel-Accelerated Neural Radiance Field},
  author={Kondo, Naruya and Ikeda, Yuya and Tagliasacchi, Andrea and Matsuo, Yutaka and Ochiai, Yoichi and Gu, Shixiang Shane},
  journal={arXiv preprint arXiv:2111.13112},
  year={2021}
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
