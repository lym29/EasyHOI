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
import random
import pandas as pd
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ho_det_utils import (
    filter_object,
    parse_det,
    intersect_box,
    union_box,
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
        
        # self.load_annos()
        # self.load_annos_simple()
        
    def load_annos_old(self):
        
        split_file = osp.join(self.data_dir, 'split', f'{self.split}.csv')
        df = pd.read_csv(split_file)
        data_dict = df.to_dict(orient='list')
        
        dtype = [
            ('img_id', 'U10'),  # Unicode string of max length 10
            ('img_obj_id', 'O'),  # Object, to accommodate arrays of various lengths
            ('img_hand_id', 'O'),
            ('hamer_info', 'O')
        ]
        filtered_file = osp.join(self.data_dir,f"{self.split}_filtered.npy")
        
        if osp.exists(filtered_file):
            img_id_list = np.load(filtered_file, allow_pickle=True)
            self.img_id_list = [(d['img_id'], 
                                 d['img_obj_id'], 
                                 d['img_hand_id'], 
                                 d['hamer_info']) for d in img_id_list]
        else:
            self.img_id_list = []
            for img_id, _ in enumerate(tqdm(data_dict['img_path'])):
                
                obj_dets = self.get_det_results(image_id=f"{img_id}", key="object")
                hand_dets = self.get_det_results(image_id=f"{img_id}", key="hand")
                res = filter_object(obj_dets, hand_dets)
                if res is None:
                    continue
                img_obj_id, img_hand_id = res
                hand_boxes = parse_det(hand_dets[img_hand_id])['bbox']
                is_right = parse_det(hand_dets[img_hand_id])['is_right']
                hamer_info = []
                for i in range(hand_boxes.shape[0]):
                    hamer_info.append(self.load_hamer_info(img_id, hand_boxes[i], is_right[i]))
                self.img_id_list.append((img_id, img_obj_id, img_hand_id, hamer_info))
                
            np.save(filtered_file, np.array(self.img_id_list, dtype=dtype))
            
        self.annos = [ ]
        
        # self.img_id_list = random.sample(self.img_id_list, 500)
        
        for item in tqdm(self.img_id_list):
            img_id, img_obj_id, img_hand_id, hamer_info = item
            path = data_dict['img_path'][int(img_id)]
            self.annos.append({'image_path': path,
                               'img_id':img_id,
                               'img_obj_id': img_obj_id,
                               'img_hand_id': img_hand_id,
                               'hamer_info': hamer_info
                                })
            
    def load_annos(self):
        self.annos = [ ]
        img_id_list = [i for i in range(30)] + [i for i in range(30, 400, 10)] + [i for i in range(400, 1400, 100)]
        
        split_file = osp.join(self.data_dir, 'split', f'{self.split}.csv')
        # if os.path.exists(split_file):
        if False:
            df = pd.read_csv(split_file)
            data_dict = df.to_dict(orient='list')
            for id, _ in enumerate(tqdm(data_dict['img_path'])):
                path = data_dict['img_path'][id]
                img_id = data_dict['img_id'][id]
                if img_id not in img_id_list:
                    continue
                if 'sid_seq_name' in data_dict:
                    obj_category = data_dict['sid_seq_name'][id].split('/')[1].split('_')[0]
                    prompt = obj_category
                else:
                    prompt = "Remove the hand from the object and restore the object to its original appearance. Ensure that no human skin or fingers are visible."
                print(prompt)
                
                self.annos.append({'image_path': path,
                                'img_id':img_id,
                                'prompt':prompt
                                })
        else:
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
            
            mask_path = self.hand_mask_dir.format(image_id)
            obj_mask_path = self.obj_mask_dir.format(image_id)
            if not (osp.exists(mask_path) and osp.exists(obj_mask_path)): 
                return None
            
            mask = Image.open(mask_path).convert('L')# already processed
            obj_mask = Image.open(obj_mask_path).convert('L')
            
            
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
                'inp_file': self.save_hoi.format(image_id),
                'out_file': self.save_obj.format(image_id),
                'mask_file': self.save_mask.format(image_id),
                'prompt': self.annos[idx]['prompt'],
            }
        
        return res