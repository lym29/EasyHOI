import sys
sys.path.append("..")
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os
import os.path as osp
import subprocess

class ObjectRecon:
    def __init__(self) -> None:
        pass
    def run(self, img_file, out_dir):
        # EXPORT_VIDEO = "true"
        # EXPORT_MESH = "true"

        INFER_CONFIG = "./configs/infer-b.yaml"
        
        current_dir = os.getcwd()
        MODEL_NAME = osp.join(current_dir, "preprocess/pretrained/openlrm-mix-base-1.1")
        MESH_DUMP = osp.join(out_dir)
        VIDEO_DUMP = MESH_DUMP
        IMAGE_INPUT = osp.join(img_file)
        # IMAGE_INPUT = "./assets/sample_input/owl.png"
        
        cmd = (
            f'cd ./third_party/OpenLRM && '
            f'python -m openlrm.launch infer.lrm '
            f'--infer {INFER_CONFIG} '
            f'model_name={MODEL_NAME} '
            f'image_input={IMAGE_INPUT} '
            f'mesh_dump={MESH_DUMP} '
            f'video_dump={VIDEO_DUMP} '
            f'export_video=true '
            f'export_mesh=true'
        )
        
        print(cmd)
        completed_process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        print(completed_process.stdout)
        if completed_process.stderr:
            print(completed_process.stderr)


    @staticmethod
    def crop_obj_img(img, mask, label=1):
        # given an image and a mask, crop the image to the mask
        img[mask!=label] = 255
        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            raise ValueError("The mask is empty.")
        print(coords.min(axis=0))
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        extend_x = (x1 - x0) // 10
        extend_y = (y1 - y0) // 10
        x0, x1 = max(0, x0-extend_x), min(mask.shape[0], x1+extend_x)
        y0, y1 = max(0, y0-extend_y), min(mask.shape[1], y1+extend_y)

        # Crop the image using the mask's bounding box
        cropped_image = img[x0:x1, y0:y1]
        cropped_mask = mask[x0:x1, y0:y1]
        return cropped_image, cropped_mask
    
    @staticmethod
    def get_color_map(N=256):
        """
        Return the color (R, G, B) of each label index.
        """
        
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        return cmap
    
    @staticmethod
    def parse_2Dmask_img(mask_img, N=10):
        """
        mask_img: RGB image, shape = (H, W, 3)
        N: number of labels (including background)

        return: pixel labels, shape = (H, W)
        """

        color_map = ObjectRecon.get_color_map(N=N)

        H, W = mask_img.shape[:2]
        labels = np.zeros((H, W)).astype(np.uint8)

        for i in range(N):
            c = color_map[i]
            valid = (mask_img[..., 0] == c[0]) & (mask_img[..., 1] == c[1]) & (mask_img[..., 2] == c[2])
            labels[valid] = i
        
        return labels
    
def test_openlrm_self():
    obj_recon = ObjectRecon()
    img_dir = "/storage/group/4dvlab/datasets/mow/obj_recon/input_for_lrm/"
    out_dir = "/storage/group/4dvlab/datasets/mow/obj_recon/results/openlrm"
    
    for image_id in os.listdir(img_dir):
        folder = os.path.join(img_dir, image_id)
        if not os.path.isdir(folder):
            continue
        
        out_path = os.path.join(out_dir, image_id)
        os.makedirs(out_path, exist_ok=True)
        
        for filename in os.listdir(folder):
            if not filename.endswith((".png", "jpg", "jpeg")):
                continue
            file = osp.join(folder, filename)
            obj_recon.run(file, out_path)
    

if __name__ == "__main__":
    test_openlrm_self()
    