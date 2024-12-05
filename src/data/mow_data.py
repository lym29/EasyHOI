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
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ho_det_utils import (
    filter_object,
    parse_det,
    intersect_box,
    union_box
)

def center_vertices(vertices, faces, flip_y=True):
    """Centroid-align vertices."""
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces


def proj3d(points, cam):
    p2d = points.cpu() @ cam.cpu().transpose(-1, -2)
    p2d = p2d[..., :2] / p2d[..., 2:3]
    return p2d


def compute_iou(boxA, boxB):
    # Determine coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the IoU by dividing the intersection area by the union area
    iou = interArea / (boxAArea + boxBArea - interArea)
    
    return iou

def check_contain(boxA, boxB):
    # Check if the top-left corner of BoxB is within BoxA
    # and the bottom-right corner of BoxB is also within BoxA
    return (boxA[0] <= boxB[0] and boxA[1] <= boxB[1] and
            boxA[2] >= boxB[2] and boxA[3] >= boxB[3])
    
def box_distance_2d(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate horizontal distance
    if x1_max < x2_min:
        dx = x2_min - x1_max
    elif x2_max < x1_min:
        dx = x1_min - x2_max
    else:
        dx = 0
    
    # Calculate vertical distance
    if y1_max < y2_min:
        dy = y2_min - y1_max
    elif y2_max < y1_min:
        dy = y1_min - y2_max
    else:
        dy = 0
    
    # Calculate Euclidean distance between the closest edges
    return (dx**2 + dy**2)**0.5
    
def get_bounding_box_np(boxA, boxB):
    top_left = np.minimum(boxA[:, :2], boxB[:, :2])
    bottom_right = np.maximum(boxA[:, 2:], boxB[:, 2:])
    
    # Concatenate the results into a single array
    return np.concatenate((top_left, bottom_right), axis=-1)

def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]


    

class MOW(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        self.image_dir = osp.join(self.data_dir, 'images', '{}.jpg')
        self.hamer_dir = osp.join(self.data_dir, 'hamer', '{}.pt')
        
        # self.mmdet_dir = osp.join(self.data_dir, 'mmdetection', 'preds', '{}.json')
        self.hodet_dir = osp.join(self.data_dir, 'hand_obj_det', '{}.pt')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mano_layer = ManoLayer(side="right", flat_hand_mean=False).to(self.device)
        self.default_betas = torch.zeros([1, 10], dtype=torch.float).to(self.device)
        self.mano_layer.eval()
        
        self.load_annos(folder_name='images')
        self.filter_annos()
        
    def load_annos(self, folder_name):
        self.annos = [ ]
        for f in os.listdir(osp.join(self.data_dir, folder_name)):
            image_id, ext = os.path.splitext(f)
            if ext.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
            obj_dets = self.get_det_results(image_id, "object")
            hand_dets = self.get_det_results(image_id, "hand")
            res = filter_object(obj_dets, hand_dets)
            if res is None:
                continue
            img_obj_id, img_hand_id = res
            
            self.annos.append({'image_id': image_id,
                                'obj_dets': parse_det(obj_dets[img_obj_id, :]),
                                'hand_dets':parse_det(hand_dets[img_hand_id, :])
                                })
        
    def filter_annos(self):
        res = []
        for anno in self.annos:
            image_id = anno['image_id']
            if os.path.exists(self.hamer_dir.format(image_id)): # check wether there is a hand  
                res.append(anno)
        self.annos = res
        
    def get_det_results(self, image_id, key):
        if key not in ["object", "hand"]:
            raise ValueError(f"Invalid item: {key}. Allowed items are {key}.")
        det_res = torch.load(self.hodet_dir.format(image_id))[key]
        
        return det_res
        
    def transform_pts(self, pts, transl, rot=None, scale=None):
        if rot is not None:
            pts = torch.matmul(pts, rot)
        pts += transl
        if scale is not None:
            pts *= scale
        return pts
    
    def load_obj(self, path, normalization=True):
        mesh = trimesh.load_mesh(path)
        verts = mesh.vertices
        
        # normalize into a unit cube centered zero, define in the MOW official code
        if normalization:
            verts -= np.min(verts, axis=0)
            verts /= np.abs(verts).max()
            verts *= 2
            verts -= np.max(verts, axis=0) / 2
            
        mesh.vertices = verts

        return mesh
        
    def get_gt_mesh(self, idx):
        anno = self.annos[idx]
        image_id = anno['image_id']
        hoi_image = Image.open(self.image_dir.format(image_id))  
        obj_mesh = self.load_obj(self.model_dir.format(image_id))
        
        W, H = hoi_image.size

        mano_pose = torch.tensor(anno['hand_pose']).unsqueeze(0).to(self.device)
        transl = torch.tensor(anno['trans']).unsqueeze(0).to(self.device)
        mano_output: MANOOutput = self.mano_layer(mano_pose, self.default_betas)
        hand_verts = mano_output.verts + transl
        
        hand_verts = hand_verts.detach().squeeze(0).to(self.device)
        hand_faces = self.mano_layer.get_mano_closed_faces()
        hand_verts = self.transform_pts(
            hand_verts,
            transl=torch.tensor(anno['hand_t']).reshape((1, 3)).to(self.device),
            rot=torch.FloatTensor(anno['hand_R']).reshape((3, 3)).to(self.device),
            scale=anno['hand_s'],
        )
        
        
        obj_verts = torch.FloatTensor(obj_mesh.vertices).to(self.device)
        obj_faces = torch.tensor(obj_mesh.faces).to(self.device)
        obj_verts, obj_faces = center_vertices(obj_verts, obj_faces)
        obj_verts = self.transform_pts(
            obj_verts,
            transl=torch.tensor(anno['t']).reshape((1, 3)).to(self.device),
            rot=torch.tensor(anno['R']).reshape((3, 3)).to(self.device),
            scale=anno['s']
        )
        
        hand_mesh = trimesh.Trimesh(vertices=hand_verts.cpu().numpy(),
                                    faces = hand_faces.cpu().numpy())
        obj_mesh = trimesh.Trimesh(vertices=obj_verts.cpu().numpy(),
                                   faces=obj_faces.cpu().numpy())
        (hand_mesh+obj_mesh).export("./test_hoi_mesh.ply")
        
        return hand_mesh, obj_mesh    
    
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
    
    def __getitem__(self, idx):
        image_id = self.annos[idx]['image_id']
        
        try:
            hoi_image = Image.open(self.image_dir.format(image_id))  
        except Exception as e:
            print(f"Failed to open the image: {e}")
        # print("load image: ", image_id)
        
        # gt_hand, gt_obj = self.get_gt_mesh(idx)
        # hamer_info = torch.load(self.hamer_dir.format(image_id))
        # hand_boxes = hamer_info['boxes'] #[(x_min, y_min, x_max, y_max), ...]
        
        hand_boxes = self.annos[idx]['hand_dets']['bbox']
        obj_boxes = self.annos[idx]['obj_dets']['bbox']
        hoi_boxes = get_bounding_box_np(obj_boxes, hand_boxes)
        hoi_score = self.annos[idx]['hand_dets']['score'] + self.annos[idx]['obj_dets']['score']
        
        res = {
            'image_id': image_id,
            'image': hoi_image,
            'hand_boxes': hand_boxes,
            'obj_boxes': obj_boxes,
            'hoi_boxes': hoi_boxes,
            'hoi_score': hoi_score
            # 'gt_hand_verts': gt_hand.vertices,
            # 'gt_obj_verts': gt_obj.vertices,
            # 'gt_obj_faces': gt_obj.faces
        }
        return res
    
    
class ImagesForInpaint(MOW):
    def __init__(self, data_dir, save_dir, img_folder, save_index='inpaint'):
        super(ImagesForInpaint,self).__init__(data_dir)
        
        self.image_dir = osp.join(self.data_dir, img_folder, '{}.jpg')
        self.mask_dir = os.path.join(data_dir, 'obj_recon/hand_mask', '{}.png')
        mask_dir = osp.join(self.data_dir, 'obj_recon/hand_mask')
        # self.idx_list = [f.rstrip('.png') for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        
        
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
        
    def __getitem__(self, idx):
        image_id = self.annos[idx]['image_id']
        
        hoi_image = Image.open(self.image_dir.format(image_id)) 
        W,H = hoi_image.size
        mask = Image.open(self.mask_dir.format(image_id)).convert('L')# already processed
        
        hand_boxes = self.annos[idx]['hand_dets']['bbox']
        obj_boxes = self.annos[idx]['obj_dets']['bbox']
        # hoi_boxes = get_bounding_box_np(obj_boxes, hand_boxes)
        box = union_box(*(b for b in obj_boxes))
        hoi_image = hoi_image.crop(box)
        mask = mask.crop(box)
        
        
        
        
        inp_file = self.save_box.format(image_id)
        if not osp.exists(inp_file): json.dump(box.tolist(), open(inp_file, 'w'))

        inp_file = self.save_hoi.format(image_id)
        if not osp.exists(inp_file): hoi_image.save(inp_file)

        inp_file = self.save_mask.format(image_id)
        if not osp.exists(inp_file): mask.save(inp_file)
        
        res = {
            'inp_file': self.save_hoi.format(image_id),
            'out_file': self.save_obj.format(image_id),
            'mask_file': self.save_mask.format(image_id),
            'prompt': "Remove the hand from the object and restore the object to its original appearance. Ensure that no human skin or fingers are visible.",
            # 'prompt': "a white background"
        }
        return res

class ImagesForSegHand(MOW):
    def __init__(self, data_dir, data_index='images_rm_bg'):
        super(ImagesForSegHand,self).__init__(data_dir)
        self.data_dir = data_dir
        self.image_dir = osp.join(self.data_dir, data_index, '{}.png')
        self.hamer_dir = osp.join(self.data_dir, 'hamer', '{}.pt')
        self.annos = [ ]
        for f in os.listdir(osp.join(self.data_dir, data_index)):
            base, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.png', '.jpeg']:
                self.annos.append({'image_id': base})
        self.filter_annos()
        
    def __len__(self):
        return len(self.annos)
        
    def __getitem__(self, idx):
        image_id = self.annos[idx]['image_id']
        
        try:
            hoi_image = Image.open(self.image_dir.format(image_id)).convert('RGB')  
            W,H = hoi_image.size
        except Exception as e:
            print(f"Failed to open the image: {e}")
        
        
        hamer_info = torch.load(self.hamer_dir.format(image_id))
        hand_boxes = hamer_info['boxes'] #[(x_min, y_min, x_max, y_max), ...]
        hand_keypts = hamer_info['keypts']
        
        hand_boxes = self.enlarge_box_np(hand_boxes, scale_factor=1)
        for i in range(hand_boxes.shape[0]):
            hand_boxes[i] = intersect_box(hand_boxes[i], np.array([0,0,W-1,H-1]))
        
        res = {
            'image_id': image_id,
            'image': hoi_image,
            'hand_boxes': hand_boxes,
            'hand_keypts': hand_keypts
        }
        return res