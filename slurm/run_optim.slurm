#!/bin/bash
#SBATCH --job-name=optim
#SBATCH --output=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.out
#SBATCH --error=/public/home/v-liuym/open-source/EasyHOI/slurm/log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=mayuexin,critical
#SBATCH --account=mayuexin-critical
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00


cd /public/home/v-liuym/open-source/EasyHOI
source /public/home/v-liuym/.bashrc

conda activate myenv
date +"%Y-%m-%d %H:%M:%S"
python src/optim_easyhoi.py -cn optim_teaser
date +"%Y-%m-%d %H:%M:%S"
