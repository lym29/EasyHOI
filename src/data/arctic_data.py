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
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ho_det_utils import (
    filter_object,
    parse_det,
    intersect_box,
    union_box
)
from src.data.base_data import BaseData, create_symbolic_links

def sample_images(data_dir, csv_file, output_csv):
    df = pd.read_csv(csv_file)
    data_dict = df.to_dict(orient='list')
    res = {'img_path': [], 'sid_seq_name': [], 'frame': []}
    img_path_dir = os.path.join(data_dir, "{}/{}/{:05d}.jpg")
    
    for i, seq_name in tqdm(enumerate(data_dict["sid_seq_name"])):
        start = data_dict["start"][i]
        end = data_dict["end"][i]
        sampled_frame = random.randint(start, end)
        sampled_view = random.randint(0, 7)
        img_path = img_path_dir.format(seq_name, sampled_view, sampled_frame)
        if os.path.exists(img_path):
            res['img_path'].append(img_path)
            res['sid_seq_name'].append(seq_name)
            res['frame'].append(sampled_frame)
                
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['img_id', 'img_path', 'sid_seq_name', 'frame'])  # Header of CSV
        for id, path in enumerate(res['img_path']):
            csvwriter.writerow([id, path, res['sid_seq_name'][id], res['frame'][id]])
            
            
    
if __name__ == "__main__":
    data_dir = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/"
    csv_file = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/split/stable_grasps_v3_frag_valid_min20.csv"
    output_file = osp.join(data_dir, 'split/test.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sample_images(os.path.join(data_dir, "arctic"), csv_file, output_file)
    df = pd.read_csv(output_file)
    data_dict = df.to_dict(orient='list')
    create_symbolic_links(data_dict["img_path"], os.path.join(data_dir, "images"))