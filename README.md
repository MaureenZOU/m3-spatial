# M3: 3D-Spatial Multimodal Memory
https://github.com/user-attachments/assets/d37ff5c9-d9f0-4a0a-88c0-f80e475176d7

## TODO
- [x] Installation
- [x] Demo
- [x] Dataset
- [x] Checkpoint
- [x] Training
- [x] Inference

## Installation
* Prepare Conda Environment
```sh
conda create --name gs python=3.10
conda activate gs
conda install -c conda-forge cudatoolkit=11.7
conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge gxx_linux-64=11.2.0
conda install -c conda-forge libxcrypt
pip install plyfile tqdm psutil setuptools mkl pandas
pip install --force-reinstall numpy==1.23.5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
* Download M3 and install submodules
```sh
git clone https://github.com/MaureenZOU/m3-spatial.git
cd submodules/diff-gaussian-rasterization && pip install -e .
cd submodules/diff-gaussian-rasterization2 && pip install -e .
```

* Download Grendel-GS in a separate folder and install submodules
```sh
git clone git@github.com:nyu-systems/Grendel-GS.git --recursive
cd submodules/gsplat && pip install -e .
cd submodules/simple-knn && pip install -e .
```

## Demo
```sh
sh run/app.sh
```
https://github.com/user-attachments/assets/42038d0b-8016-4f98-bf96-9cbae0c79708

## Dataset
* Download data for raw image
1. https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
2. http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

* Download data for embedding
| Train |
|-------|
| [Link](https://drive.google.com/file/d/17CU8H2kCXF5WL4qEdzE5lLO3T3iivLDn/view) |

* Feature Extraction
```sh
python3 -m lmm.clip.extract # CLIP feature
python3 -m lmm.siglip.extract # SigLIP feature
python3 -m lmm.dinov2.extract # DINOv2 feature
python3 -m lmm.llama.extract # LLaMA3 feature
python3 -m lmm.llamav.extract # LLaMAv feature
python3 -m lmm.seem.extract # SEEM feature
```

## Checkpoint
* We prepare trained M3 representation for two scenes train and geisel.

| Name       | size | link      |
|------------|-----|-----------------|
| train      | 2.04GB  | https://huggingface.co/xueyanz/M3-Train/resolve/main/train_ckpt.tar.gz   |
| geisel     | 1.04GB  | https://huggingface.co/xueyanz/M3-Train/resolve/main/geisel_ckpt.tar.gz  |

## Training
```sh
sh run/train.sh # single GPU
sh run/mtrain.sh # multi GPU
```

## Evaluation and Demo
```sh
sh run/eval.sh # single GPU evaluation
sh run/app.sh # run interactive demo
```
