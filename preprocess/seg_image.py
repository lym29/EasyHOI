# use conda env synchoi

import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import json
from tqdm import tqdm, trange
from PIL import Image, ImageDraw

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from preprocess.utils import image_utils
from src.data import (
    MOW, 
    HO3D, 
    ImgData
)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
def crop_image_and_mask(masks, bounding_box, image_size):
    w, h = image_size
    x1, y1, x2, y2 = np.array(bounding_box, dtype=np.int32)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w-1), min(y2, h-1)
    cropped_masks=[]
    for m in masks:
        cropped_m = m[y1:y2, x1:x2]
        cropped_masks.append(cropped_m)

    return cropped_masks

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    img = (img * 255).astype(np.uint8)
    return img

class SAM_wrapper:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.predictor = self.load_model()

    def load_model(self):
        sam = sam_model_registry["vit_h"](checkpoint="preprocess/pretrained/sam_vit_h_4b8939.pth")
        sam.to(self.device)
        predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        return predictor
        
    def compute_inpaint_mask(self, img, img_fn, 
                             mask_dir="preprocess/collected_data/mask"):
        # threshold on white
        # Define lower and uppper limits
        lower = np.array([240, 240, 240])
        upper = np.array([255, 255, 255])
        
        # load orig mask 
        orig_mask = cv2.imread(os.path.join(mask_dir, f"{img_fn}_obj.png"))
        thresh = cv2.inRange(img, lower, upper)
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[thresh==0] = 255
        mask[orig_mask>0] = 255
        # apply mask to image
        # result = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(os.path.join(self.out_dir, f"{img_fn}.png"), mask)
        
    def vis_mask(self, img, input_box, mask, out_path):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')
        plt.savefig(fname=out_path)
        
        # draw = ImageDraw.Draw(img)
        # box = input_box
        # draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=3)
        

    def get_mask_with_bbox(self, input_box):
        x, y, w, h = input_box
        box_for_sam = np.array([x, y, x+w, y+h])
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_for_sam[None, :],
            multimask_output=False,
        )
        return masks
    
    def get_mask_with_pt_and_bbox(self, pt, input_box):
        x, y, w, h = input_box
        box_for_sam = np.array([x, y, x+w, y+h])
        masks, _, _ = self.predictor.predict(
            point_coords=pt,
            point_labels=np.ones(pt.shape[0]),
            box=box_for_sam[None, :],
            multimask_output=False,
        )
        return masks
    
def seg_before_inpaint(data_dir, save_dir):
    
    for folder in ['hand_mask','hand_rm_vis']:
        folder = os.path.join(save_dir, folder)
        os.makedirs(folder, exist_ok=True)
    ds = ImgData(data_dir)
    print("total num", len(ds) )
    
    hand_mask_dir = os.path.join(save_dir, 'hand_mask', '{}.png')
    hand_rm_dir = os.path.join(save_dir, 'hand_rm_vis', '{}.png')
    
    seg = SAM_wrapper()
    
    # for data in tqdm(mow_data):
    for n in trange(len(ds)):
        data = ds[n]
        img_id = data['image_id']
        
        img = np.array(data['image'])
        W, H = data['image'].size
        hand_boxes = data['hand_boxes']
        hamer_info = data['hamer_info']
        seg.predictor.set_image(img)
        
        hand_mask = np.zeros((H, W), dtype=bool)
        img_rm_hand = Image.fromarray(img)
        draw = ImageDraw.Draw(img_rm_hand)
        
        for i in range(hand_boxes.shape[0]):
            if hamer_info[i] is None:
                partial_hand_mask = seg.get_mask_with_bbox(hand_boxes[i])[0]
            else:
                keypts = hamer_info[i]["keypts"][0:1]
                partial_hand_mask = seg.get_mask_with_pt_and_bbox(keypts, hand_boxes[i])[0]
                
                x, y = keypts[0]
                radius = 5
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
            hand_mask[partial_hand_mask==True] = True
            draw.rectangle(hand_boxes[i], outline="red", width=2)
            
        img_rm_hand = np.array(img_rm_hand)
        img_rm_hand[hand_mask, :] = np.zeros(3, dtype=np.uint8)
        try:
            img_rm_hand = Image.fromarray(img_rm_hand)
            img_rm_hand.save(hand_rm_dir.format(img_id))
        except Exception as e:
            print(f"Failed to save the image: {e}")
            
        # Convert boolean mask: True (black, 0), False (white, 255)
        hand_mask = np.where(hand_mask, 0, 255).astype(np.uint8)
        try:
            hand_mask = Image.fromarray(hand_mask, mode='L')  # 'L' mode is for grayscale
            hand_mask.save(hand_mask_dir.format(img_id))
        except Exception as e:
            print(f"Failed to save the image: {e}")
        
            
            
    print("finished")

def seg_after_inpaint(data_dir, save_dir):
    for folder in ['inpaint_mask', 'input_for_lrm']: 
        folder = os.path.join(save_dir, folder)
        os.makedirs(folder, exist_ok=True)
        
    
    ds = ImgData(data_dir)
    # input
    inpainted_dir = os.path.join(data_dir, 'obj_recon/inpaint/glide_obj')
    img_list = [file.rstrip('.png') for file in os.listdir(inpainted_dir) if file.endswith('.png')]
    inpainted_dir = os.path.join(inpainted_dir,  '{}.png')
    
    denoised_dir = os.path.join(data_dir, 'obj_recon/inpaint/denoised_obj',  '{}.png')
    hoi_box_dir = os.path.join(data_dir, 'obj_recon/inpaint/hoi_box',  '{}.json')
    obj_mask_dir = os.path.join(data_dir, 'obj_recon/obj_mask', '{}.png')
    hand_mask_dir = os.path.join(data_dir, 'obj_recon/hand_mask', '{}.png')
    
    print(len(img_list))
    
    # output
    lrm_dir = os.path.join(save_dir, 'input_for_lrm', '{}/{}.png')
    inpaint_mask_dir = os.path.join(save_dir, 'inpaint_mask', '{}.png')
    
    seg = SAM_wrapper()
    
    for n in trange(len(ds)):
        data = ds[n]
        image_id = data['image_id']
        img = data['image'] # original image
        orig_W, orig_H = img.size
        
        if not os.path.exists(inpainted_dir.format(image_id)):
            continue
        if not os.path.exists(obj_mask_dir.format(image_id)):
            continue
        
        with open(hoi_box_dir.format(image_id), 'r') as file:
            hoi_box = json.load(file)
        hoi_box = [int(item) for item in hoi_box]
        x, y, w, h = hoi_box

        inpainted_img = cv2.cvtColor(cv2.imread(inpainted_dir.format(image_id)), cv2.COLOR_BGR2RGB)
        denoised_img = cv2.cvtColor(cv2.imread(denoised_dir.format(image_id)), cv2.COLOR_BGR2RGB)
        
        
        obj_mask = cv2.cvtColor(cv2.imread(obj_mask_dir.format(image_id)), cv2.COLOR_BGR2GRAY)
        hand_mask = cv2.cvtColor(cv2.imread(hand_mask_dir.format(image_id)), cv2.COLOR_BGR2GRAY)
        
        # Crop the masks using the bounding box
        obj_mask = obj_mask[y:y+h, x:x+w]
        hand_mask = hand_mask[y:y+h, x:x+w]

        obj_box, point_list, edges_dilated = image_utils.mask_to_sam_prompt(obj_mask, hand_mask) 
        
        os.makedirs(os.path.join(save_dir, 'input_for_lrm', str(image_id)), exist_ok=True)
        Image.fromarray(edges_dilated).save(lrm_dir.format(image_id, "hand_edges"))
        if obj_box is None:
            print("no obj detected: ", image_id)
            continue   
        
        seg.predictor.set_image(denoised_img)
        
        if len(point_list) == 0:
            inpaint_mask = seg.get_mask_with_bbox(obj_box)
        else:
            inpaint_mask = seg.get_mask_with_pt_and_bbox(np.stack(point_list, axis=0), obj_box)
        
        inpaint_mask = np.any(inpaint_mask, axis=0)
        inpaint_mask = inpaint_mask | (obj_mask > 0)
        
        
        # ** export for vis mask and box/pts prompt **
        if max(w, h) < 500:
            thickness = 1
        else:
            thickness = 2
        vis_image = image_utils.draw_masked_image_with_labels(inpainted_img, inpaint_mask, 
                                                              obj_box, 
                                                              point_list,
                                                              thickness=thickness
                                                            )
        Image.fromarray(vis_image).save(lrm_dir.format(image_id, "vis_mask"))
        
        # ** export for filled mask **
        inpaint_mask = image_utils.fill_mask(inpaint_mask)
        vis_image = image_utils.draw_masked_image_with_labels(inpainted_img, inpaint_mask, 
                                                              obj_box, 
                                                              point_list,
                                                              thickness=thickness
                                                            )
        Image.fromarray(vis_image).save(lrm_dir.format(image_id, "vis_mask_filled"))
        
        # ** export segemented object image for reconstruction **
        # ** presereve inpaint_mask region, set background pixels to white
        inpainted_img[inpaint_mask==0] = np.array([255,255,255], dtype=np.uint8)
        Image.fromarray(inpainted_img).save(lrm_dir.format(image_id, "full"))
        
        # create the inpaint mask that match the original size
        new_inpaint_mask = np.zeros([orig_H, orig_W])
        new_inpaint_mask[y:y+h, x:x+w] = inpaint_mask
        new_inpaint_mask = np.where(new_inpaint_mask, 255, 0).astype(np.uint8)
        new_inpaint_mask = Image.fromarray(new_inpaint_mask, mode='L')  # 'L' mode is for grayscale
        new_inpaint_mask.save(inpaint_mask_dir.format(image_id))
        
            
    print("finished")
    
def seg_after_inpaint_graphcut(data_dir, save_dir, datatype):
    for folder in ['inpaint_mask', 'input_for_lrm']: 
        # obj mask is compute in optim code, use the diff of inpaint mask and hand mask
        folder = os.path.join(save_dir, folder)
        os.makedirs(folder, exist_ok=True)
        
    if datatype == "mow" or datatype == "in_the_wild":
        ds = MOW(data_dir)
    elif datatype == "ho3d":
        ds = HO3D(data_dir)
    elif datatype in ["oakink", "arctic"]:
        ds = ImgData(data_dir)
    
    # input
    inpainted_dir = os.path.join(data_dir, 'obj_recon/inpaint/glide_obj')
    img_list = [file.rstrip('.png') for file in os.listdir(inpainted_dir) if file.endswith('.png')]
    inpainted_dir = os.path.join(inpainted_dir,  '{}.png')
    denoised_dir = os.path.join(data_dir, 'obj_recon/inpaint/denoised_obj',  '{}.png')
    obj_mask_dir = os.path.join(data_dir, 'obj_recon/obj_mask', '{}.png')
    
    print(len(img_list))
    
    # output
    lrm_dir = os.path.join(save_dir, 'input_for_lrm', '{}/{}.png')
    inpaint_mask_dir = os.path.join(save_dir, 'inpaint_mask', '{}.png')
    
    for n in trange(len(ds)):
        data = ds[n]
        image_id = data['image_id']
        img = data['image'] # original image
        W, H = img.size
        
        if not os.path.exists(inpainted_dir.format(image_id)):
            continue
        
        inpainted_img = cv2.cvtColor(cv2.imread(inpainted_dir.format(image_id)), cv2.COLOR_BGR2RGB)
        inpainted_img = cv2.resize(inpainted_img, (W, H), interpolation=cv2.INTER_LINEAR)
        
        denoised_img = cv2.cvtColor(cv2.imread(denoised_dir.format(image_id)), cv2.COLOR_BGR2RGB)
        denoised_img = cv2.resize(denoised_img, (W, H), interpolation=cv2.INTER_LINEAR)
        
        partial_mask = cv2.cvtColor(cv2.imread(obj_mask_dir.format(image_id)), cv2.COLOR_BGR2GRAY)
        os.makedirs(os.path.join(save_dir, 'input_for_lrm', str(image_id)), exist_ok=True)
        
        mask = np.zeros(inpainted_img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        mask[partial_mask > 0] = 1

        # 应用Graph Cut算法
        (mask, bgd_model, fgd_model)=cv2.grabCut(denoised_img, mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)

        # 将结果转换为二值mask
        full_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
        # full_mask = np.where((mask == cv2.GC_BGD), 0, 1).astype('uint8')
        
        
        inpainted_img[full_mask==0] = np.array([255,255,255])
        Image.fromarray(inpainted_img).save(lrm_dir.format(image_id, "full"))
        
        inpaint_mask = np.where(full_mask, 255, 0).astype(np.uint8)
        inpaint_mask = Image.fromarray(inpaint_mask, mode='L')  # 'L' mode is for grayscale
        try:
            inpaint_mask.save(inpaint_mask_dir.format(image_id))
        except Exception as e:
            print(f"Failed to save the image: {e}")
            
    print("finished")
    
def vertical_concatenate(images):
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
        
    return new_im
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Provide the path to the data directory. The directory must contain a folder named 'images'.")
    parser.add_argument('--before', action='store_true', help='before inpaint')
    parser.add_argument('--vis', action='store_true', help='output vis')
    
    
    args = parser.parse_args()
        
    data_dir = args.data_dir
    save_dir = os.path.join(data_dir, "obj_recon")
    print(f"Saved to: {save_dir}")
            
    if args.before:
        print("segment hand before inpainting")
        seg_before_inpaint(data_dir, save_dir)
    else:
        print("segment object after inpainting")
        seg_after_inpaint(data_dir, save_dir)
        