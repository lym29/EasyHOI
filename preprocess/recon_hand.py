import sys
sys.path.append("..")
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import open_clip
from open_clip import tokenizer
import os
import os.path as osp
import subprocess
import trimesh
import json
import pyrender
import argparse
sys.path.append('third_party/hamer')
from hamer.models import MANO
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'


class HandPoseEstim:
    def __init__(self) -> None:
        pass
    def run(self, img_dir=None, out_dir=None, mask_path=None, save_mesh=True):
        current_dir = os.getcwd()
        if img_dir is None:
            img_dir = osp.join(current_dir, "preprocess/collected_data/input")
        if out_dir is None:
            out_dir = osp.join(current_dir, "preprocess/collected_data/output/hamer")
        if mask_path is None:
            mask_path = osp.join(current_dir, "preprocess/collected_data/hand_mask/")
        cmd = (
            f'cd ./third_party/hamer && '
            f'python demo.py '
            f'--img_folder {img_dir} --out_folder {out_dir} --hand_mask_path {mask_path} '
            f'--batch_size=1 --side_view --full_frame '
        )
        if save_mesh:
            cmd += f'--save_mesh'
        
        print(cmd)
        completed_process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        print(completed_process.stdout)
        if completed_process.stderr:
            print(completed_process.stderr)
    
def test_hamer_mow():
    hpe = HandPoseEstim()
    # hpe.run()
    hpe.run(img_dir="/storage/group/4dvlab/datasets/mow/images",
            out_dir="/storage/group/4dvlab/datasets/mow/hamer",
            save_mesh=False)
    
def test_hamer_wild():
    hpe = HandPoseEstim()
    # hpe.run()
    hpe.run(img_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/images/",
            out_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/hamer/",
            save_mesh=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Provide the path to the data directory. The directory must contain a folder named 'images'.")
    
    args = parser.parse_args()
        
    data_dir = args.data_dir
    print(f"Received data dir: {data_dir}")
    
    hpe = HandPoseEstim()
    hpe.run(
        img_dir=os.path.join(data_dir, "images"),
        out_dir=os.path.join(data_dir, "hamer"),
        mask_path=os.path.join(data_dir, "obj_recon/hand_mask"),
        save_mesh=False
    )
    
    