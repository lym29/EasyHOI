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
import tqdm
from manotorch.manolayer import ManoLayer, MANOOutput
import trimesh
import shutil
import pandas as pd
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ho_det_utils import (
    compute_iou,
    mask_to_bbox
)

def create_symbolic_links(image_path_list, link_dir):
    # Create the directory for links if it does not exist
    os.makedirs(link_dir, exist_ok=True)
    
    for id, path in enumerate(image_path_list):
        _, file_extension = os.path.splitext(path)
        file_format = file_extension[1:]
        link_path = os.path.join(link_dir, f"{id}.{file_format}")
        
        try:
            os.symlink(path, link_path)
        except FileExistsError:
            print(f"Link already exists for {path}")
        except OSError as e:
            print(f"Error creating link for {path}: {e}")

def extract_img(img_path_list, img_dir, img_name_list = None):
    shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)
    if img_name_list is None:
        img_name_list = list(range(len(img_path_list)))
    for id, src_path in enumerate(img_path_list):
        _, file_extension = os.path.splitext(src_path)
        file_format = file_extension[1:]
        tgt_path = osp.join(img_dir, f"{img_name_list[id]}.{file_format}")
        shutil.copy(src_path, tgt_path)
    
class BaseData(Dataset):
    def __init__(self, data_dir, split="evaluation"):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        
        self.hodet_dir = osp.join(self.data_dir, 'hand_obj_det', '{}.pt')
        self.hand_mask_dir = osp.join(self.data_dir, 'obj_recon/hand_mask', '{}.png')
        self.hamer_dir = osp.join(self.data_dir, 'hamer/{}.pt')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mano_layer = ManoLayer(side="right", flat_hand_mean=False).to(self.device)
        self.default_betas = torch.zeros([1, 10], dtype=torch.float).to(self.device)
        self.mano_layer.eval()
        
        self.for_inpaint = False
        
        self.load_annos()
        
    def load_annos(self):
        self.annos = []
        
    def load_hamer_info(self, image_id, hand_bbox, is_right):
        if not osp.exists(self.hamer_dir.format(image_id)):
            return None
        hamer_info = torch.load(self.hamer_dir.format(image_id))
        boxes = hamer_info["boxes"]
        hamer_is_right = hamer_info["is_right"]
        
        max_iou = 0
        best_id = None
        for i in range(len(boxes)):
            if hamer_is_right[i] != is_right:
                continue
            iou = compute_iou(hand_bbox, boxes[i])
            if iou > max_iou:
                best_id = i
                max_iou = iou
                
        if best_id is None:
            return None
        
        info = {"id": best_id}
        for key in hamer_info:
            if key in ["batch_size", "mano_params"]:
                continue
            info[key] = hamer_info[key][best_id]
        
        return info
    
    def get_hamer_id(self, image_id, hand_bbox):
        pass
        
            
    def set_for_inpaint(self, save_dir, save_index="inpaint"):
        self.save_hoi = osp.join(save_dir, save_index,'glide_hoi/{}.png')
        os.makedirs(osp.dirname(self.save_hoi), exist_ok=True)
        self.save_obj = osp.join(save_dir, save_index,'glide_obj/{}.png')
        os.makedirs(osp.dirname(self.save_obj), exist_ok=True)
        self.save_mask = osp.join(save_dir, save_index,'det_mask/{}.png')
        os.makedirs(osp.dirname(self.save_mask), exist_ok=True)
        self.save_box = osp.join(save_dir, save_index,'hoi_box/{}.json')
        os.makedirs(osp.dirname(self.save_box), exist_ok=True)
        
        self.error = osp.join(save_dir, save_index, 'errors/{}.txt')
        os.makedirs(osp.dirname(self.error), exist_ok=True)
        self.for_inpaint = True
        
    def get_det_results(self, image_id, key):
        if key not in ["object", "hand"]:
            raise ValueError(f"Invalid item: {key}. Allowed items are {key}.")
        det_res = torch.load(self.hodet_dir.format(image_id))[key]
        
        return det_res
    
    def parse_det(self, det_res):
        # score_max = np.argmax(det_res[:, 4])
        res = {
            "bbox": det_res[:, :4],
            "score": det_res[:, 4],
            "is_right": det_res[:, -1], #(0: l, 1: r)
            "state": det_res[:, 5] #{0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
        }
        return res
        
    
    def enlarge_box_np(self, boxes, scale_factor=2.5):
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        size = boxes[:, 2:] - boxes[:, :2]
        new_size = size * scale_factor
        new_box = np.concatenate((center - new_size / 2, center + new_size / 2), axis=-1)
        return new_box    
    
    def convert_to_np(self, data):
        if isinstance(data, dict):
            return {key: self.convert_to_np(value) for key, value in data.items()}
        elif isinstance(data, list):
            return np.array(data)
        else:
            return data
        
    
    def __len__(self):
        return len(self.annos)
    
    
class ImgData(BaseData):
    def __init__(self, data_dir, split="test"):
        super().__init__(data_dir, split)
        
        self.data_dir = data_dir
        self.split = split
        
        self.hodet_dir = osp.join(self.data_dir, 'hand_obj_det', '{}.pt')
        self.hand_mask_dir = os.path.join(self.data_dir, 'obj_recon/hand_mask', '{}.png')
        self.obj_mask_dir = os.path.join(self.data_dir, 'obj_recon/obj_mask', '{}.png')
        
        
        self.for_inpaint=False
        
            
    def load_annos(self):
        self.annos = [ ]
        img_dir = os.path.join(self.data_dir, "images")
        prompt = "Remove the hand from the object and restore the object to its original appearance. Remove all the fingers."
        for file in os.listdir(img_dir):
            if not file.endswith(("png", "jpg")):
                continue
            self.annos.append({'image_path': os.path.join(img_dir, file),
                            'img_id':file.split('.')[0],
                            'prompt':prompt
                            })
        
    def __len__(self):
        return len(self.annos)
    
    def __getitem__(self, idx):
        image_path = self.annos[idx]['image_path']
        
        try:
            hoi_image = Image.open(image_path)  
            W, H = hoi_image.size
        except Exception as e:
            print(f"Failed to open the image: {e}")
        # print("load image: ", image_id)
        
        image_id = self.annos[idx]['img_id']
        
        if not self.for_inpaint:
            res = {
                'image_id': image_id,
                'image': hoi_image,
            }
        else:
            
            hand_mask_path = self.hand_mask_dir.format(image_id)
            obj_mask_path = self.obj_mask_dir.format(image_id)
            if not (osp.exists(hand_mask_path) and osp.exists(obj_mask_path)): 
                return None
            
            hand_mask = np.array(Image.open(hand_mask_path).convert('L'))# already processed
            obj_mask = np.array(Image.open(obj_mask_path).convert('L'))
            mask = (hand_mask > 0) & ~(obj_mask > 0) # Pixels in hand_mask but not in obj_mask
            # Convert the boolean array back to an image
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            
            box = mask_to_bbox(np.array(obj_mask), rate=3)
            x,y,w,h = box
            hoi_image = hoi_image.crop([x,y,x+w,y+h])
            mask = mask.crop([x,y,x+w,y+h])
            
            inp_file = self.save_box.format(image_id)
            if not osp.exists(inp_file): json.dump(box.tolist(), open(inp_file, 'w'))

            inp_file = self.save_hoi.format(image_id)
            if not osp.exists(inp_file): hoi_image.save(inp_file)

            inp_file = self.save_mask.format(image_id)
            if not osp.exists(inp_file): mask.save(inp_file)
            res = {
                # for inpainting
                'image_id': image_id,
                'inp_file': self.save_hoi.format(image_id),
                'out_file': self.save_obj.format(image_id),
                'mask_file': self.save_mask.format(image_id),
                'prompt': self.annos[idx]['prompt'],
            }
        
        return res