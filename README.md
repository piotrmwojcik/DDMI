# DDMI: Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations
**[Project Page](https://dogyunpark.github.io/ddmi) |
[Paper](https://arxiv.org/abs/2401.12517)**

Dogyun Park,
Sihyeon Kim,
Sojin Lee,
Hyunwoo J. Kim†.

This repository is an official implementation of the ICLR 2024 paper DDMI (Domain-Agnostic Latent Diffusion Models for Synthesizing High-Quality Implicit Neural Representations).

<div align="center">
  <img src="asset/main.png" width="900px" />
</div>

Data preparation and training codes for occupancy, NeRF, and video will be released soon.

## Setup
To install requirements, run:
```
git clone https://github.com/mlvlab/DDMI.git
cd DDMI
conda create -f requirements.yaml
conda activate ddmi
```

## Data preparation
### Image
We have utilized two datasets for our experiments: [AFHQ-V2](https://github.com/clovaai/stargan-v2) and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans). You may change the location of dataset by changing the directory of config files in `configs/`.

## Training
To train other signal domains, you may change the `domain` of config files in `configs/`. By default, the model's checkpoint will be stored in `./results`.
### First-stage training (D2C-VAE)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp d2c-vae --configs configs/d2c-vae/img.yaml
```

### Second-stage training (LDM)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 main.py --exp ldm --configs configs/ldm/img.yaml
```

## Generation
You can generate a signal from the pretrained model in `./results` by changing the `mode` of config files to `eval` from `train`, then run:
```
python main.py --exp ldm --configs configs/ldm/img.yaml
```



