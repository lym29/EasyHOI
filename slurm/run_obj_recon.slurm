#!/bin/bash
#SBATCH --job-name=obj_recon
#SBATCH --output=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.out
#SBATCH --error=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=mayuexin,critical
#SBATCH --account=mayuexin-critical
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=30G

cd /public/home/v-liuym/open-source/EasyHOI
source /public/home/v-liuym/.bashrc
#export DATA_DIR="/storage/group/4dvlab/yumeng/HO3D_easyhoi/"
export DATA_DIR="/public/home/v-liuym/open-source/EasyHOI/data/"

conda activate myenv
date +"%Y-%m-%d %H:%M:%S"
python preprocess/seg_image.py --data_dir $DATA_DIR
date +"%Y-%m-%d %H:%M:%S"

export CUDA_PATH=/public/software/CUDA/cuda-11.8
export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=${CUDA_PATH}
conda activate instantmesh
export HUGGINGFACE_HUB_CACHE="/public/home/v-liuym/.cache/huggingface/hub"
export CUDA_VISIBLE_DEVICES=0
date +"%Y-%m-%d %H:%M:%S"
python preprocess/instantmesh_gen.py preprocess/configs/instant-mesh-large.yaml $DATA_DIR
date +"%Y-%m-%d %H:%M:%S"

conda activate synchoi
date +"%Y-%m-%d %H:%M:%S"
python preprocess/resample_mesh.py --data_dir $DATA_DIR
date +"%Y-%m-%d %H:%M:%S"
