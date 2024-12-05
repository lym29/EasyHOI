import json
import os
import os.path as osp
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageDraw
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from manotorch.manolayer import ManoLayer, MANOOutput
import trimesh
import random
import csv
import pandas as pd
import rootutils
import shutil
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import glob
import re
from src.utils import hand_utils, geom_utils
from src.utils import to_np

SUB_DIR = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    
]

def prepare_data_split(data_dir, out_dir):
    # data_dir = "/inspurfs/group/mayuexin/datasets/dexycb/raw/"
    # select frame 60 from dataset
    selected_frames = [60]
    file_list = []
    for frame in selected_frames:
        files = glob.glob(os.path.join(data_dir, SUB_DIR[0], '*', '*', 'color_{:06d}.jpg'.format(frame)))
        print(len(files))
        file_list += files
    print(len(file_list))
    
    csv_file = os.path.join(out_dir, "split", "test.csv")
    img_dir = os.path.join(out_dir, "images")

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_id', 'img_path'])
        
        for idx, image_path in tqdm(enumerate(file_list)):
            writer.writerow([idx, image_path])
            shutil.copy(image_path, os.path.join(img_dir, f"{idx}.jpg"))
    
    
if __name__ == "__main__":
    data_dir = "/inspurfs/group/mayuexin/datasets/dexycb/raw/"
    out_dir = "/storage/group/4dvlab/yumeng/DexYCB_easyhoi"
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "split"), exist_ok=True)
    prepare_data_split(data_dir, out_dir)