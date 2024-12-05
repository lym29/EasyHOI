import json
import os
import os.path as osp
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageDraw
import imageio
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import image_utils

class Obman(Dataset):
    def __init__(self, data_dir, save_dir, split='test', save_index='obman_glide'):
        self.data_dir = osp.join(data_dir, 'obman', split)
        
        image_dir = osp.join(self.data_dir, 'rgb')
        self.idx_list = [int(f.rstrip('.jpg')) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        self.image_dir = osp.join(self.data_dir, 'rgb', '{:08d}.jpg')
        self.mask_dir = osp.join(self.data_dir, 'segm', '{:08d}.png')
        
        
        self.save_hoi = osp.join(save_dir, save_index, split,'glide_hoi/{:08d}.png')
        os.makedirs(osp.dirname(self.save_hoi), exist_ok=True)
        self.save_obj = osp.join(save_dir, save_index, split,'glide_obj/{:08d}.png')
        os.makedirs(osp.dirname(self.save_obj), exist_ok=True)
        self.save_mask = osp.join(save_dir, save_index, split,'det_mask/{:08d}.png')
        os.makedirs(osp.dirname(self.save_mask), exist_ok=True)
        self.save_box = osp.join(save_dir, save_index, split,'hoi_box/{:08d}.json')
        os.makedirs(osp.dirname(self.save_box), exist_ok=True)
        
        self.error = osp.join(save_dir, save_index, 'errors/{}.txt')
        os.makedirs(osp.dirname(self.error), exist_ok=True)
        
    def get_image(self, index):
        return Image.open(self.image_dir.format(index))   
        
    def get_obj_mask(self, index):
        """R channel"""
        mask_file = self.mask_dir.format(index)
        if osp.exists(mask_file):
            mask = np.array(Image.open(mask_file))[..., 0]
            mask = (mask > 0) * 255
        else:
            mask = np.ones_like(np.array(self.get_image(index))[..., 0])
            mask = (mask > 0) * 255
        return mask
    
    def get_hand_mask(self, index):
        """B channel or from hA?"""
        mask_file = self.mask_dir.format(index)
        if osp.exists(mask_file):
            mask = np.array(Image.open(mask_file))[..., 1]
            mask = (mask > 0) * 255
        else:
            mask = np.ones_like(np.array(self.get_image(index))[..., 2])
            mask = (mask > 0) * 255
        return mask
    
    def get_bbox(self, obj_mask, hand_mask, H, W):
        bbox = image_utils.mask_to_bbox((obj_mask + hand_mask) > 0)
        bbox = image_utils.square_bbox(bbox, pad=0.8)
        bbox = image_utils.intersect_box(bbox, np.array([0,0,W-1,H-1]))
        bbox = image_utils.square_bbox_no_black(bbox, Ymax=H, Xmax=W,)
        hoi_box = bbox
        return hoi_box
    
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        index = self.idx_list[idx]
        hoi_image = Image.open(self.image_dir.format(index))  

        obj_mask = self.get_obj_mask(index)
        hand_mask = self.get_hand_mask(index)
        not_hand_mask = (np.array(hand_mask) < 255) * 255
        mask = Image.fromarray(not_hand_mask.astype(np.uint8))
        
        W, H = hoi_image.size
        hoi_box = self.get_bbox(obj_mask, hand_mask, H, W)
        
        inp_file = self.save_box.format(index)
        if not osp.exists(inp_file): json.dump(hoi_box.tolist(), open(inp_file, 'w'))

        inp_file = self.save_hoi.format(index)
        if not osp.exists(inp_file): imageio.imwrite(inp_file, hoi_image)

        inp_file = self.save_mask.format(index)
        if not osp.exists(inp_file): imageio.imwrite(inp_file, mask)
        
        out = {
            'inp_file': self.save_hoi.format(index),
            'out_file': self.save_obj.format(index),
            'mask_file': self.save_mask.format(index),
            
            'prompt': "Remove the hand from the object and restore the object to its original appearance. Remove all the fingers.",
        }
        
        return out