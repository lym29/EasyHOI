# use conda env mmdetection

import torch
from mmdet.apis import DetInferencer
from rich.pretty import pprint
import os
import argparse

class Detector:
    def __init__(self) -> None:
        self.inferencer = self.load_model()

    def load_model(self):
        # Choose to use a config
        model_name = 'rtmdet_tiny_8xb32-300e_coco'
        # Setup a checkpoint file to load
        checkpoint = './preprocess/pretrained/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

        # Set the device to be used for evaluation
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        

        # Initialize the DetInferencer
        inferencer = DetInferencer(model_name, checkpoint, device)
        
        return inferencer

    def detect(self, img_path, out_dir):
        # Use the detector to do inference
        self.inferencer(img_path, out_dir=out_dir, no_save_pred=False)
    
    
if __name__ == "__main__":
    det = Detector()
    det.load_model()
    
    parser = argparse.ArgumentParser(description='Object detection.')
    parser.add_argument('dataset_type', type=str, help='Type of the dataset to process')
    parser.add_argument('--before', action='store_true', help='before inpaint')
    
    args = parser.parse_args()
    dataset_type = args.dataset_type
    print(f"Received dataset type: {dataset_type}")
    
    # Example of further usage
    if dataset_type == "mow":
        data_dir = "/storage/group/4dvlab/datasets/mow/images_rm_bg"
        out_root = "/storage/group/4dvlab/datasets/mow/mmdetection"
    elif dataset_type == "in_the_wild":
        if args.before:
            data_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/images_rm_bg"
            out_root="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/mmdetection"
        else:
            data_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/obj_recon/mow_glide/glide_obj/"
            out_root="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/mmdetection"
        
    else:
        print("Unknown dataset type.")
    
    
    
    for fname in os.listdir(data_dir):
        if not fname.endswith(('jpg', 'jpeg','png')):
            continue
        img_path = os.path.join(data_dir, fname)
        # out_dir = os.path.join(out_root, fname.split('.')[0])
        det.detect(img_path, out_root)