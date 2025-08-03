#!/bin/bash
#SBATCH --job-name=lisa_arctic
#SBATCH --output=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.out
#SBATCH --error=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=mayuexin,critical
#SBATCH --account=mayuexin-critical
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=1:00:00


cd /public/home/v-liuym/open-source/EasyHOI
source /public/home/v-liuym/.bashrc
#export DATA_DIR="/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/"
export DATA_DIR="/public/home/v-liuym/open-source/EasyHOI/data/"

conda activate llmseg
export TRANSFORMERS_CACHE="/public/home/v-liuym/.cache/huggingface/hub"

date +"%Y-%m-%d %H:%M:%S"
#python preprocess/lisa_ho_detect.py --seg_hand --skip --load_in_8bit --data_dir $DATA_DIR
python preprocess/lisa_ho_detect.py --skip --load_in_8bit --data_dir $DATA_DIR
date +"%Y-%m-%d %H:%M:%S"
