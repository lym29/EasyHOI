#!/bin/bash
#SBATCH --job-name=hamer
#SBATCH --output=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.out
#SBATCH --error=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=critical
#SBATCH --account=mayuexin-critical
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=1:00:00


cd /public/home/v-liuym/open-source/EasyHOI
source /public/home/v-liuym/.bashrc
export CUDA_PATH=/public/software/CUDA/cuda-12.1
export PATH=${CUDA_PATH}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#export DATA_DIR="/storage/group/4dvlab/yumeng/HO3D_easyhoi/"
export DATA_DIR="/public/home/v-liuym/open-source/EasyHOI/data/"

conda activate synchoi
date +"%Y-%m-%d %H:%M:%S"
python preprocess/recon_hand.py --data_dir $DATA_DIR
date +"%Y-%m-%d %H:%M:%S"
