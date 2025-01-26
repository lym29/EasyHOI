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
import glob
import re
from src.utils import hand_utils, geom_utils
from src.utils import to_np

ALL_CAT = [
    "apple",
    "banana",
    "binoculars",
    "bottle",
    "bowl",
    "cameras",
    "can",
    "cup",
    "cylinder_bottle",
    "donut",
    "eyeglasses",
    "flashlight",
    "fryingpan",
    "gamecontroller",
    "hammer",
    "headphones",
    "knife",
    "lightbulb",
    "lotion_pump",
    "mouse",
    "mug",
    "pen",
    "phone",
    "pincer",
    "power_drill",
    "scissors",
    "screwdriver",
    "squeezable",
    "stapler",
    "teapot",
    "toothbrush",
    "trigger_sprayer",
    "wineglass",
    "wrench",
]

CENTER_IDX = 9

def parse_seq_id(seq_id):
    parts = seq_id.split('_')
    if len(parts) == 3 or len(parts) == 4:
        obj_id = str(parts[0])
        intent_id = str(parts[1])
        if len(parts) == 3:
            subject_id = parts[2]
        else:
            subject_id = f"{parts[2]}_{parts[3]}"  # Combining C & D into one subject_id
        return (obj_id, intent_id, subject_id)
    else:
        return None

def sample_images(root_dir,output_csv,img_per_seq=1):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['img_id', 'obj_id', 'intent_id', 'subj_id', 'img_path'])  # Header of CSV
        img_num = 0
        for subdir, dirs, files in tqdm(os.walk(root_dir)):
            for dir in dirs:
                seq_id = parse_seq_id(dir)
                if seq_id is None:
                    continue
                full_dir_path = os.path.join(subdir, dir)
                for sequence_subdir, _, images in os.walk(full_dir_path):
                    image_files = [img for img in images if img.endswith('.png')]
                    if len(image_files) <= img_per_seq:
                        sampled_images = image_files  # Take all if fewer than img_per_seq
                    else:
                        sampled_images = random.sample(image_files, img_per_seq)
                    
                    # Write sampled image paths to CSV
                    for image in sampled_images:
                        full_image_path = os.path.join(sequence_subdir, image)
                        obj_id, intent_id, subject_id = seq_id
                        csvwriter.writerow([img_num, obj_id, intent_id, subject_id, full_image_path])
                        img_num += 1
                        
                        
def get_obj_path(oid, oakink_shape_dir, use_downsample=True, key="align"):
    meta_path = os.path.join(oakink_shape_dir, "metaV2")
    obj_suffix_path = "align_ds" if use_downsample else "align"
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    # virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))
    assert oid in real_meta
    if oid in real_meta:
        obj_name = real_meta[oid]["name"]
        obj_path = os.path.join(oakink_shape_dir, "OakInkObjectsV2")
    # else:
    #     obj_name = virtual_meta[oid]["name"]
    #     obj_path = os.path.join(oakink_shape_dir, "OakInkVirtualObjectsV2")
    obj_mesh_path = list(
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
    if len(obj_mesh_path) > 1:
        obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
    assert len(obj_mesh_path) == 1
    return obj_mesh_path[0]

def get_hand_parameter(path):
    pose = pickle.load(open(path, "rb"))
    return pose["pose"], pose["shape"], pose["tsl"]

def get_hand_path(oid, img_path, oakink_shape_dir):
    frame = img_path.split('/')[-1].split('.')[0].split('_')[-1]
    meta_path = os.path.join(oakink_shape_dir, "metaV2")
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    assert oid in real_meta
    obj_cat = None
    for cat in ALL_CAT:
        path = os.path.join(oakink_shape_dir, "oakink_shape_v2", cat, oid)
        if os.path.exists(path):
            obj_cat = cat
            break
    assert obj_cat is not None
    print(obj_cat)
    
    for subdir in os.listdir(path):
        source = open(os.path.join(path, subdir, "source.txt")).read()
        parts = source.split('/')
        extracted = '/'.join(parts[1:3])
        if extracted in img_path:
            hand_path = os.path.join(path, subdir, "hand_param.pkl")
            return hand_path
    
    return None
        
def get_gt(data_cfg):
    split_path = os.path.join(data_cfg.base_dir, "split", f"{data_cfg.split}.csv")
    df = pd.read_csv(split_path)
    df['img_id'] = df['img_id'].astype(str)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mano_layer = ManoLayer(side="right", center_idx=0).to(device)
    
    for index, row in tqdm(df.iterrows()):
        img_id = row["img_id"]
        obj_id = row["obj_id"]
        
        # if os.path.exists(os.path.join(data_cfg.base_dir, "gt_hoi", f"{img_id}_hand.ply")):
        #     continue
        
        hand_path = get_hand_path(obj_id, row["img_path"], 
                                oakink_shape_dir=os.path.join(data_cfg.orig_path, "shape"))
        if hand_path is None:
            print(hand_path)
            continue
        hand_pose, hand_shape, hand_tsl = get_hand_parameter(hand_path)
        mano_output: MANOOutput = mano_layer(torch.from_numpy(hand_pose[None,:]).to(device),
                                             torch.from_numpy(hand_shape[None,:]).to(device))
        center_joint = mano_output.center_joint.cpu().squeeze()
        print(center_joint)
        
        rot = torch.tensor(hand_pose[:3])[None]
        transl = torch.tensor(hand_tsl)[None] 
        hand_world_mat = geom_utils.axis_angle_t_to_matrix(rot, transl)
        wTh = torch.linalg.inv(hand_world_mat[0])
        
        hand_verts = to_np(mano_output.verts.squeeze()) + hand_tsl[None, :]
        hand_mesh = trimesh.Trimesh(hand_verts, mano_layer.get_mano_closed_faces())
        hand_mesh.apply_transform(to_np(wTh))
        hand_mesh.vertices = hand_mesh.vertices + to_np(center_joint)
        hand_mesh.export(os.path.join(data_cfg.base_dir, "gt_hoi", f"{img_id}_hand.ply"))
        
        hand_joints = to_np(mano_output.joints.squeeze()) + hand_tsl[None, :]
        hand_joints = trimesh.transform_points(hand_joints, to_np(wTh))
        hand_joints = hand_joints + to_np(center_joint)
        np.save(os.path.join(data_cfg.base_dir, "gt_hoi", f"{img_id}_hand_joints.npy"),
                hand_joints)
        
        obj_path = get_obj_path(obj_id,oakink_shape_dir=os.path.join(data_cfg.orig_path, "shape"))
        obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
        bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        obj_trimesh.vertices = obj_trimesh.vertices - bbox_center 
        obj_trimesh.apply_transform(to_np(wTh))
        obj_trimesh.vertices = obj_trimesh.vertices + to_np(center_joint)
        obj_trimesh.export(os.path.join(data_cfg.base_dir, "gt_hoi", f"{img_id}_obj.ply"))
                        
                        
if __name__ == '__main__':
    img_dir = "/inspurfs/group/mayuexin/datasets/OakInk/image/stream_release_v2/"
    output_csv = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/split/test.csv"
    
    if not os.path.exists(output_csv):
        sample_images(img_dir, output_csv)
        
    data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi"
    split_file = osp.join(data_dir, 'split/test.csv')
    df = pd.read_csv(split_file)
    data_dict = df.to_dict(orient='list')
    # create_symbolic_links(data_dict["img_path"], os.path.join(data_dir, "images"))