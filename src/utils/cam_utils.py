import numpy as np
import torch
import torch.nn.functional as F
import json
from typing import Dict
from pyrender.camera import IntrinsicsCamera
from PIL import Image, ExifTags
import math

def batched_cam(cam, batch_size):
    cam_ext, cam_int = cam["extrinsics"], cam["intrinsic"]
    cam_ext = cam_ext.unsqueeze(0).expand(batch_size,-1,-1)
    cam_int = cam_int.unsqueeze(0).expand(batch_size,-1,-1)
    return cam_ext, cam_int
    

def verts_transfer_cam(src_verts:torch.Tensor, 
                       src_cam:Dict[str, torch.Tensor], 
                       tgt_cam:Dict[str, torch.Tensor]):
    """
    src_verts: torch.Tensor [batch_size, V, 3]
    """
    B,_,_ = src_verts.shape
    src_cam_ext = src_cam["extrinsics"].unsqueeze(0).expand(B,-1,-1)
    tgt_cam_ext = tgt_cam["extrinsics"].unsqueeze(0).expand(B,-1,-1)
    new_verts = (src_verts - src_cam_ext[:,:,-1]) @ src_cam_ext[:,:,:3]
    new_verts = new_verts @ tgt_cam_ext[:, :, :3].mT + tgt_cam_ext[:, :,-1] 
    return new_verts

def project_hand_2D(verts:torch.Tensor, cam:Dict[str, torch.Tensor]):
    B,_,_ = verts.shape
    cam_ext, cam_int = batched_cam(cam, B)
    verts = (verts - cam_ext[:,:,-1]) @ cam_ext[:,:,:3]
    verts = verts @ cam_int.mT #homogeneous coordinates
    verts_2D = verts[:,:2] / verts[:,-1:] 
    return verts_2D
    

def load_cam(file_path, device):
    with open(file_path, 'r') as json_file:
        cam = json.load(json_file)

    cam['extrinsics'] = torch.tensor(cam['extrinsics']).to(device)
    cam['intrinsic'] = torch.tensor([[cam['fx'], 0, cam['cx']],
                                     [0, cam['fy'], cam['cy']],
                                     [0, 0, 1]]).to(device)
    return cam

def get_projection(cam, width, height):
    fx, fy = cam['fx'] * width, cam['fy'] * height
    cx, cy = cam['cx'] * width, cam['cy'] * height
    intri_cam = IntrinsicsCamera(fx,fy,cx,cy)
    proj_mat = intri_cam.get_projection_matrix(width, height)
    return proj_mat

def correct_image_orientation(img):
    # Get image EXIF data
    exif = img._getexif()

    # Get the orientation tag code (274 is the standard code for orientation)
    orientation_key = next((key for key, val in ExifTags.TAGS.items() if val == 'Orientation'), None)

    # If the image has EXIF data and contains the orientation tag
    if exif and orientation_key in exif:
        orientation = exif[orientation_key]

        # Apply the necessary rotations
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
        elif orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 7:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)

    return img

def resize_frame(image:Image, block_size=8, max_size=1024):
    w, h = image.size
    scaling_factor = min(max_size / w, max_size / h)
    if scaling_factor < 1:
        w = int(w * scaling_factor)
        h = int(h * scaling_factor)
    new_w = (w // block_size) * block_size
    new_h = (h // block_size) * block_size
    image = image.resize((new_w, new_h))
    return image, (new_w, new_h)


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    device = camera_position.device
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32).to(device)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1).to(device)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1).to(device)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

def calc_orig_cam_params(D_crop,fov_crop, W_orig, H_orig, crop_bbox):
    """
    Calculate the original camera distance and FoV based on the cropped image.

    Parameters:
    - D_crop: Camera distance for the cropped image (float).
    - W_orig: Width of the original image (int or float).
    - H_orig: Height of the original image (int or float).
    - crop_bbox: Bounding box of the cropped area in the cropped image (x_min, y_min, x_max, y_max).
    - fov_crop: Field of view of the cropped image (in degrees, float).

    Returns:
    - D_orig: Camera distance for the original image (float).
    - fov_orig: Field of view for the original image (in degrees, float).
    """
    # Extract bbox dimensions from the cropped image
    x_min, y_min, bbox_width, bbox_height = crop_bbox
    x_max = bbox_width + x_min
    y_max = bbox_height + y_min

    # Calculate scaling factors for width and height
    scale_w = W_orig / bbox_width
    scale_h = H_orig / bbox_height

    # Convert fov_crop from degrees to radians
    fov_crop_rad = math.radians(fov_crop)

    # Compute fov_orig based on the dominant scaling factor
    scale = (scale_w + scale_h) / 2
    fov_orig_rad = 2 * math.atan(math.tan(fov_crop_rad / 2) * scale)
    fov_orig = math.degrees(fov_orig_rad)

    # Compute D_orig using the relationship between fov_orig and fov_crop
    D_orig = D_crop * (math.tan(fov_orig_rad / 2) / math.tan(fov_crop_rad / 2))

    return D_orig, fov_orig

def adjust_principal_point(crop_bbox, w, h):
    """
    Adjust the principal point (cx, cy) in normalized coordinates based on the crop box.

    Parameters:
    - crop_bbox: List of 4 values [x_min, y_min, x_max, y_max] representing the crop box in the original image.
    - w: Width of the original image (float or int).
    - h: Height of the original image (float or int).

    Returns:
    - cx: Adjusted x-coordinate of the principal point (normalized, float).
    - cy: Adjusted y-coordinate of the principal point (normalized, float).
    """
    # Calculate the normalized crop box center
    x_min, y_min, bbox_width, bbox_height = crop_bbox
    x_max = bbox_width + x_min
    y_max = bbox_height + y_min

    cx_crop = (x_min + x_max) / 2 / w  # Normalize x-center
    cy_crop = (y_min + y_max) / 2 / h  # Normalize y-center

    # Return adjusted principal point
    return cx_crop, cy_crop