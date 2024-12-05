import sys
import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

def parse_args(args):
    parser = argparse.ArgumentParser(description="Vis Segmentation")
    parser.add_argument("--dataset", default="arctic")
    return parser.parse_args(args)

def vis_inpaint_mask(args):
    args = parse_args(args)
    if args.dataset == "arctic":
        data_dir = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/"
    elif args.dataset == "oakink":
        data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/"
    elif args.dataset == "in_the_wild":
        data_dir = "/storage/group/4dvlab/yumeng/InTheWild_easyhoi/"
        
    # image_dir = os.path.join(data_dir, "obj_recon/inpaint/glide_obj")
    image_dir= os.path.join(data_dir, "obj_recon/input_for_lrm/")
    bbox_dir = os.path.join(data_dir, "obj_recon/inpaint/hoi_box")
    mask_dir = os.path.join(data_dir, "obj_recon/inpaint_mask")
    out_dir = os.path.join(data_dir, "obj_recon/inpaint_mask_vis")
    
    os.makedirs(out_dir, exist_ok=True)
    
    name = "hand_image-10"
    
    # original_image = np.array(Image.open(os.path.join(image_dir, name + ".png")))
    original_image = np.array(Image.open(os.path.join(image_dir, name, "full.png")))
    
    mask_image = np.array(Image.open(os.path.join(mask_dir, name + ".png")).convert('L'))
    with open(os.path.join(bbox_dir, name + ".json")) as f:
        bbox = json.load(f)
        bbox = [int(num) for num in bbox]
    x,y,w,h = bbox
    print(x,y,w,h)
    mask_image = mask_image[y:y+h, x:x+w]
    
    mask = mask_image>0
    save_img = original_image.copy()
    save_img[mask] = (
            original_image * 0.5
            + mask[:, :, None].astype(np.uint8) * np.array([0, 255, 0]) * 0.5
    )[mask]
    
    Image.fromarray(save_img).save(os.path.join(out_dir, name + ".png"))

def main(args):
    args = parse_args(args)
    if args.dataset == "arctic":
        data_dir = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/"
    elif args.dataset == "oakink":
        data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/"
    
    image_dir = os.path.join(data_dir, "images")
    obj_mask_dir = os.path.join(data_dir, "obj_recon/obj_mask")
    out_dir = os.path.join(data_dir, "obj_recon/obj_mask_vis")
    
    os.makedirs(out_dir, exist_ok=True)
    
    for fn in tqdm(os.listdir(image_dir)):
        if fn.endswith("jpg"):
            mask_path = os.path.join(obj_mask_dir, fn.replace(".jpg", ".png"))
        elif fn.endswith("png"):
            mask_path = os.path.join(obj_mask_dir, fn)
        else:
            continue
        
        if not os.path.exists(mask_path):
            continue
        
        
        original_image = np.array(Image.open(os.path.join(image_dir, fn)))
        Image.fromarray(original_image).save(os.path.join(out_dir, "orig"+fn))
        
        mask_image = np.array(Image.open(mask_path).convert('L'))
        
        
        mask = mask_image>0
        save_img = original_image.copy()
        save_img[mask] = (
                original_image * 0.5
                + mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[mask]
        
        Image.fromarray(save_img).save(os.path.join(out_dir, fn))
        
    
if __name__ == "__main__":
    """
    python preprocess/vis_seg_results.py --dataset arctic
    """
    # main(sys.argv[1:])
    vis_inpaint_mask(sys.argv[1:])