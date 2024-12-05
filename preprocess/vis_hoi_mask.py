import cv2
import numpy as np
import os

def color_masks(mask_obj, mask_hand):
    # Ensure masks are binary and have the same shape
    mask_obj = mask_obj.astype(bool)
    mask_hand = mask_hand.astype(bool)
    assert mask_obj.shape == mask_hand.shape, "Masks must have the same shape"

    # Create a white RGB image as the background
    height, width = mask_obj.shape
    result = np.full((height, width, 3), 255, dtype=np.uint8)  # White background

    # Color the intersection 
    intersection = mask_obj & mask_hand
    result[intersection] = [115, 225, 220]  # yellow

    # Color the rest of mask object green
    rest_of_a = mask_obj & ~intersection
    result[rest_of_a] = [50, 160, 50]  # Green
    
    # Color the rest of mask hand red
    rest_of_b = mask_hand & ~intersection
    result[rest_of_b] = [65, 65, 170]  # Red

    

    return result

def read_mask(file_path):
    # Read the image in grayscale mode
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise FileNotFoundError(f"Could not read the mask file: {file_path}")
    
    # Binarize the mask (just in case it's not already binary)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Convert to boolean
    return mask.astype(bool)

def vis_hoi_mask():
    # File paths for mask A and mask B
    mask_obj_path = "/storage/group/4dvlab/yumeng/Teaser_easyhoi/obj_recon/inpaint_mask/0.png"
    mask_hand_path = "/storage/group/4dvlab/yumeng/Teaser_easyhoi/obj_recon/hand_mask/0.png"
    out_dir = "/storage/data/v-liuym/EasyHOI_results/paper_vis/"

    try:
        # Read masks from files
        mask_obj = read_mask(mask_obj_path)
        mask_hand = read_mask(mask_hand_path)
        mask_hand = ~mask_hand

        # Color the masks
        colored_result = color_masks(mask_obj, mask_hand)

        # Optionally, save the result
        cv2.imwrite(os.path.join(out_dir, "colored_hoi.png"), colored_result)
        
        print("Process completed successfully.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except AssertionError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def redraw_lisa_result():
    data_dir = "/storage/group/4dvlab/yumeng/InTheWild_easyhoi/LISA_output/"
    img_name = "najib-kalil-f01uTP4gX2c-unsplash"

if __name__ == "__main__":
    vis_hoi_mask()
    