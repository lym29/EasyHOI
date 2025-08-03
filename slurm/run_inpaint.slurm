#!/bin/bash
#SBATCH --job-name=inpaint
#SBATCH --output=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.out
#SBATCH --error=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=mayuexin,critical
#SBATCH --account=mayuexin-critical
#SBATCH --gres=gpu:NVIDIAA40:1
#SBATCH --time=1:00:00
#SBATCH --mem=20G


cd /public/home/v-liuym/open-source/EasyHOI
source /public/home/v-liuym/.bashrc
#export DATA_DIR="/storage/group/4dvlab/yumeng/HO3D_easyhoi/"
export DATA_DIR="/public/home/v-liuym/open-source/EasyHOI/data/"

conda activate afford_diff
date +"%Y-%m-%d %H:%M:%S"
python preprocess/inpaint.py --data_dir $DATA_DIR --save_dir $DATA_DIR/obj_recon/ --img_folder images --inpaint --skip
date +"%Y-%m-%d %H:%M:%S"
