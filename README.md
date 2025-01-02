# EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild
[![GitHub](https://img.shields.io/github/license/lym29/EasyHOI.svg?style=flat-square&color=df7e66)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2411.14280-gree.svg?style=flat-square)](https://arxiv.org/abs/2411.14280)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-blue?style=flat-square)](https://lym29.github.io/EasyHOI-page/)

<div style="text-align: center;">
  <img src="docs/teaser_gif/clip1.gif" alt="Description of GIF" style="width:100%;">
</div>
<div style="text-align: center;">
  <img src="docs/teaser_gif/clip2.gif" alt="Description of GIF" style="width:100%;">
</div>
<div style="text-align: center;">
  <img src="docs/teaser_gif/clip3.gif" alt="Description of GIF" style="width:100%;">
</div>
<div style="text-align: center;">
  <img src="docs/teaser_gif/clip4.gif" alt="Description of GIF" style="width:100%;">
</div>
<div style="text-align: center;">
  <img src="docs/teaser_gif/clip5.gif" alt="Description of GIF" style="width:100%;">
</div>

EasyHOI is a pipeline designed for reconstructing hand-object interactions from single-view images.

---
## ✅ TODO
- [x] Provide the code for utilizing the Tripo3D API to improve reconstruction quality - Completed on 2024-12-24.
- [x] Resolve issues in segmentation. - Completed on 2025-01-02
- [ ] Integrate the code execution environments into one.
- [ ] Complete a one-click demo.
---

## 📑 Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
    - [Initial Reconstruction of Hand and Object](#initial-reconstruction-of-the-hand-and-object)
    - [Prior-guided Optimization](#optimization)
3. [Acknowledgements](#acknowledgements)

## 🛠️ Installation
Download MANO models from the [official website](https://mano.is.tue.mpg.de/) and place the mano folder inside the ./assets directory. After setting up, the directory structure should look like this:
```
assets/
├── anchor/
├── mano/
│ ├──models/
│ ├──webuser/
│ ├──__init__.py
│ ├──__LICENSE.txt
├── contact_zones.pkl
├── mano_backface_ids.pkl
```

Create the environment for optimization:
```
conda create -n easyhoi python=3.9
conda activate easyhoi
conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda env update --file environment.yaml
```
Install pytorch3d follow the [official instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).


Install HaMeR and ViTPose:
```
cd third_party
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ./hamer
pip install -e .[all]
cd ../ViTPose
pip install -v -e .
```

Install chamfer_distance:
```
pip install git+'https://github.com/otaheri/chamfer_distance'
```

Install mano:
```
pip install git+'https://github.com/otaheri/MANO'
pip install git+'https://github.com/lixiny/manotorch'
```

Install nvdiffrast:
```
pip install git+'https://github.com/NVlabs/nvdiffrast.git'
```

### Additional Environments
Since I haven’t resolved the conflict between the environments yet, it’s necessary to create several virtual environments called afford_diff, lisa, and instantmesh. Please refer to the links below to set up these environments.

- afford_diff: https://github.com/NVlabs/affordance_diffusion/blob/master/docs/install.md

- lisa: https://github.com/dvlab-research/LISA
- instantmesh: https://github.com/TencentARC/InstantMesh?tab=readme-ov-file

Thanks to the authors of these wonderful projects. I will resolve the environment conflicts as soon as possible and provide a more user-friendly demo.


## 🚀  Usage

### Initial Reconstruction of the Hand and Object

Set the data directory by running the following command:

```
export DATA_DIR="./data"
```
Place your images in the $DATA_DIR/images folder. If you prefer a different path, ensure it contains a subfolder named images.

#### Step 1: Hand pose estimation, get hand mask from hamer
```
conda activate easyhoi
python preprocess/recon_hand.py --data_dir $DATA_DIR

```

#### Step 2: Segment hand mask and object mask from image before inpainting
```
export TRANSFORMERS_CACHE="/public/home/v-liuym/.cache/huggingface/hub"
conda activate lisa
CUDA_VISIBLE_DEVICES=0 python preprocess/lisa_ho_detect.py --seg_hand --skip --load_in_8bit --data_dir $DATA_DIR
CUDA_VISIBLE_DEVICES=0 python preprocess/lisa_ho_detect.py --skip --load_in_8bit --data_dir $DATA_DIR
```

#### Step 3: Inpaint
```
conda activate afford_diff
python preprocess/inpaint.py --data_dir $DATA_DIR --save_dir $DATA_DIR/obj_recon/ --img_folder images --inpaint --skip
```


#### Step 4: Segment inpainted obj get the inpainted mask 

```
conda activate easyhoi
python preprocess/seg_image.py --data_dir $DATA_DIR
```

#### Step 5: Reconstruct obj
##### Use InstantMesh
```
conda activate instantmesh
export HUGGINGFACE_HUB_CACHE="/public/home/v-liuym/.cache/huggingface/hub"
python preprocess/instantmesh_gen.py preprocess/configs/instant-mesh-large.yaml $DATA_DIR
```

##### Use Tripo3D

To use Tripo3D for reconstruction, you need to generate an API key following the instructions in the [Tripo AI Docs](https://platform.tripo3d.ai/docs/quick-start). Then replace the `api_key` in `preprocess/tripo3d_gen.py` with your own key. 
After updating the API key, execute the following command in your terminal:
```
python preprocess/tripo3d_gen.py --data_dir $DATA_DIR
```

#### Step 6: fix the object mesh, get watertight mesh
```
conda activate easyhoi
python preprocess/resample_mesh.py --data_dir $DATA_DIR [--resample]
```

---

### Optimization
```
conda activate easyhoi
python src/optim_easyhoi.py -cn optim_teaser

```

## 🙏 Acknowledgements

We would like to express our gratitude to the authors and contributors of the following projects:
- [HaMeR](https://github.com/geopavlakos/hamer/tree/main)
- [AffordanceDiffusion](https://github.com/NVlabs/affordance_diffusion/blob/master/docs/install.md)

- [LISA](https://github.com/dvlab-research/LISA)
- [InstantMesh](https://github.com/TencentARC/InstantMesh?tab=readme-ov-file)
- [IHOI](https://github.com/JudyYe/ihoi)
- [MOHO](https://github.com/ZhangCYG/MOHO)

## Citation
If you find our work useful, please consider citing us using the following BibTeX entry:
```
@article{liu2024easyhoi,
  title={EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild},
  author={Liu, Yumeng and Long, Xiaoxiao and Yang, Zemin and Liu, Yuan and Habermann, Marc and Theobalt, Christian and Ma, Yuexin and Wang, Wenping},
  journal={arXiv preprint arXiv:2411.14280},
  year={2024}
}
```