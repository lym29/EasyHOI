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
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ho_det_utils import (
    filter_object,
    parse_det,
    intersect_box,
    union_box
)
from src.data.base_data import BaseData

class HO3D(BaseData):
    def __init__(self, data_dir, split="evaluation"):
        super().__init__(data_dir, split)
        
        self.data_dir = data_dir
        self.split = split
        
        self.image_dir = osp.join(self.data_dir,"HO3D",split, '{}/rgb/{}.jpg')
        
        # self.mmdet_dir = osp.join(self.data_dir, 'mmdetection', 'preds', '{}.json')
        self.hodet_dir = osp.join(self.data_dir, 'hand_obj_det', '{}.pt')
        self.hand_mask_dir = os.path.join(self.data_dir, 'obj_recon/hand_mask', '{}.png')
        
        self.for_inpaint=False
        
        self.load_annos()
        
    def load_annos(self):
        dtype = [
            ('seq_id', 'U10'),  # Unicode string of max length 10
            ('frame', 'U10'),
            ('img_obj_id', 'O'),  # Object, to accommodate arrays of various lengths
            ('img_hand_id', 'O')
        ]
        split_file = np.loadtxt(osp.join(self.data_dir,"HO3D",f"{self.split}.txt"), dtype=str)
        total_list = [s.split("/") for s in split_file]
        
        filtered_file = osp.join(self.data_dir,f"{self.split}_filtered.npy")
        if osp.exists(filtered_file):
            img_id_list = np.load(filtered_file, allow_pickle=True)
            self.img_id_list = [(d['seq_id'], d['frame'], d['img_obj_id'], d['img_hand_id']) for d in img_id_list]
        else:
            self.img_id_list = []
            for item in tqdm.tqdm(total_list):
                seq_id, frame = item
                
                obj_dets = self.get_det_results(image_id=f"{seq_id}_{frame}", key="object")
                hand_dets = self.get_det_results(image_id=f"{seq_id}_{frame}", key="hand")
                res = filter_object(obj_dets, hand_dets)
                if res is None:
                    continue
                img_obj_id, img_hand_id = res
                self.img_id_list.append((seq_id, frame, img_obj_id, img_hand_id))
                
            np.save(filtered_file, np.array(self.img_id_list, dtype=dtype))
            
        self.annos = [ ]
        
        # self.img_id_list = random.sample(self.img_id_list, 500)
        
        for item in tqdm.tqdm(self.img_id_list):
            seq_id, frame, img_obj_id, img_hand_id = item
            # obj_dets = self.get_det_results(image_id=f"{seq_id}_{frame}", key="object")
            # hand_dets = self.get_det_results(image_id=f"{seq_id}_{frame}", key="hand")
            self.annos.append({'image_id': (seq_id, frame),
                               'img_obj_id': img_obj_id,
                               'img_hand_id': img_hand_id,
                                # 'obj_dets': parse_det(obj_dets[img_obj_id, :]),
                                # 'hand_dets': parse_det(hand_dets[img_hand_id, :])
                                })
    
    def __len__(self):
        return len(self.annos)
    
    def __getitem__(self, idx):
        seq_id, frame = self.annos[idx]['image_id']
        
        try:
            hoi_image = Image.open(self.image_dir.format(seq_id, frame))  
        except Exception as e:
            print(f"Failed to open the image: {e}")
        # print("load image: ", image_id)
        
        image_id = f"{seq_id}_{frame}"
        obj_dets = self.get_det_results(image_id=f"{seq_id}_{frame}", key="object")
        hand_dets = self.get_det_results(image_id=f"{seq_id}_{frame}", key="hand")
        self.annos[idx]['obj_dets'] = parse_det(obj_dets[self.annos[idx]['img_obj_id'], :])
        self.annos[idx]['hand_dets'] = parse_det(hand_dets[self.annos[idx]['img_hand_id'], :])
        
        
        if not self.for_inpaint:
            hand_boxes = self.annos[idx]['hand_dets']['bbox']
            obj_boxes = self.annos[idx]['obj_dets']['bbox']
            
            hoi_boxes = [b for b in obj_boxes] + [b for b in hand_boxes]
            hoi_boxes = union_box(*hoi_boxes)
            hoi_score = self.annos[idx]['hand_dets']['score'] + self.annos[idx]['obj_dets']['score']
            res = {
                'image_id': image_id,
                'image': hoi_image,
                'hand_boxes': hand_boxes,
                'obj_boxes': obj_boxes,
                'hoi_boxes': hoi_boxes,
                'hoi_score': hoi_score,
            }
        else:
            mask = Image.open(self.hand_mask_dir.format(image_id)).convert('L')# already processed
            obj_boxes = self.annos[idx]['obj_dets']['bbox']
            # hoi_boxes = get_bounding_box_np(obj_boxes, hand_boxes)
            box = union_box(*(b for b in obj_boxes))
            box = np.array(box, dtype=int)
            hoi_image = hoi_image.crop(box)
            mask = mask.crop(box)
            
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
                'prompt': "Remove the hand from the object and restore the object to its original appearance. Ensure that no human skin or fingers are visible.",
                # 'prompt': "a white background"
            }
        
        return res