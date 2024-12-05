import os
import os.path as osp
from pathlib import Path
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import igl
import trimesh
from PIL import Image
import torch
from tqdm import trange, tqdm
import numpy as np
from mesh_to_sdf import mesh_to_voxels
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from src.models.hoi_optim_module import HOI_Sync


from src.utils.cam_utils import (
    load_cam,
    get_projection,
    correct_image_orientation,
    resize_frame,
    center_looking_at_camera_pose
)
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def image_process(input_image, hand_mask, obj_mask, inpaint_mask):
    input_image = correct_image_orientation(input_image)
    hand_mask = correct_image_orientation(hand_mask).convert('L') # grayscale
    obj_mask = correct_image_orientation(obj_mask).convert('L')
    inpaint_mask = correct_image_orientation(inpaint_mask).convert('L')
    
    input_image, new_size = resize_frame(input_image)
    
    hand_mask = hand_mask.resize(new_size)
    obj_mask = obj_mask.resize(new_size)
    inpaint_mask = inpaint_mask.resize(new_size)
    
    hand_mask, obj_mask, inpaint_mask = np.array(hand_mask), np.array(obj_mask), np.array(inpaint_mask)
    
    return input_image, hand_mask, obj_mask, inpaint_mask
        
def get_obj_cam(device, w, h):
    # param from instantmesh
    DEFAULT_DIST = 4.5
    DEFAULT_FOV = 30.0
    focal_length = 0.5 / np.tan(np.deg2rad(DEFAULT_FOV) * 0.5)
    # cam_pose = torch.FloatTensor([DEFAULT_DIST, 0, 0]).cuda()
    cam_pose = torch.FloatTensor([DEFAULT_DIST, 0, 0]).cuda()
    
    
    obj_cam = {'fx':focal_length, 'fy': focal_length, 'cx': 0.5, 'cy': 0.5}
    obj_cam["extrinsics"] = center_looking_at_camera_pose(cam_pose).to(device)
    obj_cam["projection"] = torch.FloatTensor(get_projection(obj_cam, width=w, height=h)).to(device)
    
    return obj_cam

def get_obj_cam_tripo(device, w, h):
    # param from instantmesh
    DEFAULT_DIST = 3.5
    DEFAULT_FOV = 30.0
    focal_length = 0.5 / np.tan(np.deg2rad(DEFAULT_FOV) * 0.5)
    cam_pose = torch.FloatTensor([DEFAULT_DIST, 0, 0]).cuda()
    
    ratio = w/h
    print("ratio: ", ratio)
    
    obj_cam = {'fx':focal_length, 'fy': focal_length*ratio, 'cx': 0.5, 'cy': 0.5}
    obj_cam["extrinsics"] = center_looking_at_camera_pose(cam_pose).to(device)
    obj_cam["projection"] = torch.FloatTensor(get_projection(obj_cam, width=w, height=h)).to(device)
    
    return obj_cam

def load_hamer_info(file_path):
    if not osp.exists(file_path):
        return None
    hamer_info = torch.load(file_path)
    boxes = hamer_info["boxes"]
    mano_params = hamer_info["mano_params"]
    for key, item in mano_params.items():
        mano_params[key] = torch.tensor(item)
    fullpose = torch.cat([mano_params["global_orient"], mano_params["hand_pose"]], dim=1)
    mano_params['fullpose'] = matrix_to_axis_angle(fullpose).reshape(-1, 16*3) #[B, 16* 3]
    
    info_list = []
    for i in range(len(boxes)):
        info = {"id": i}
        info["mano_params"] = {key:mano_params[key][i:i+1] for key in mano_params }
        
        for key in hamer_info:
            if key in ["batch_size", "mano_params"]:
                continue
            info[key] = hamer_info[key][i]
        info_list.append(info)
        
    return info_list

def try_until_success(func, max_attempts=5, exception_to_check=Exception, verbose=True, **kwargs):
    """
    Tries to execute a function until it succeeds or reaches the maximum attempts.

    Args:
        func (callable): The function to execute.
        max_attempts (int, optional): Maximum number of attempts. Defaults to 5.
        exception_to_check (Exception, optional): The type of exception to catch. 
                                                  Defaults to Exception (catches all exceptions).
        verbose (bool, optional): Whether to print messages about attempts. Defaults to True.

    Returns:
        The result of the function if successful, otherwise None.
    """

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"Attempt {attempt}/{max_attempts}...")

        try:
            random.seed(attempt)
            np.random.seed(attempt)
            result = func(**kwargs)  # Try executing the function
            return result     # Return the result if successful
        except exception_to_check as e:
            if verbose:
                print(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                if verbose:
                    print(f"Retrying ...")

    if verbose:
        print(f"Function failed after {max_attempts} attempts.")
    return None  # Return None if all attempts fail

def load_data_single(cfg: DictConfig, file, hand_id, is_tripo = False):
    # resize the input image, because the resolution for nvdiffrastmust must be [<=2048, <=2048]
    try:
        img_path = osp.join(cfg.input_dir, file)
        input_image = Image.open(img_path)
    except:
        img_path = osp.join(cfg.input_dir, file.replace(".png", ".jpg"))
        input_image = Image.open(img_path)
        
        
    img_fn = file.split(".")[0]
    
    if not os.path.exists(osp.join(cfg.inpaint_dir, f"{img_fn}.png")):
        print("inpaint image not exist:", osp.join(cfg.inpaint_dir, f"{img_fn}.png"))
        return None
    
    
    hand_mask = Image.open(osp.join(cfg.hand_mask_dir, f"{img_fn}.png"))
    obj_mask = Image.open(osp.join(cfg.obj_mask_dir, f"{img_fn}.png"))
    inpaint_mask = Image.open(osp.join(cfg.inpaint_dir, f"{img_fn}.png"))
    origin_w, origin_h = input_image.width, input_image.height
    
    input_image, hand_mask, obj_mask, inpaint_mask = image_process(input_image, hand_mask, obj_mask, inpaint_mask)
    
    w,h = input_image.width, input_image.height
    
    hand_cam_file = osp.join(cfg.hand_dir, f"{img_fn}_cam.json")
    obj_mesh_path = osp.join(cfg.obj_dir,img_fn, "fixed.obj")
    
    if not os.path.exists(obj_mesh_path):
        print("obj mesh not exist:", obj_mesh_path)
        return None
    obj_mesh = trimesh.load(obj_mesh_path)
    object_colors = obj_mesh.visual.vertex_colors
    
    """ load hand info """
    info = torch.load(osp.join(cfg.hand_dir, f"{img_fn}.pt"))
    
    hand_info = {'mano_params': {}}
    for key in info['mano_params']:
        if hand_id >= info['mano_params'][key].shape[0]:
            return None
        hand_info['mano_params'][key] = info['mano_params'][key][hand_id]
            
    hand_info.update({key: info[key][hand_id] 
                        for key in info 
                        if key not in ['batch_size','mano_params']})
    
    print("right hand" if int(hand_info['is_right'].item()) else "left hand")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mano_params = hand_info["mano_params"]
    for key in mano_params:
        mano_params[key] = torch.tensor(mano_params[key]).to(device).unsqueeze(0)
        
    """Adjust the cam params to fit object into hand cam coordinates"""
    
    hand_cam = load_cam(hand_cam_file, device=device)
    hand_cam["projection"] = torch.tensor(get_projection(hand_cam, origin_w, origin_h)).float().to(device)
    
    if is_tripo:
        obj_cam = get_obj_cam_tripo(device, origin_w, origin_h)
    else:
        obj_cam = get_obj_cam(device, origin_w, origin_h)
        
    obj_verts = torch.tensor(obj_mesh.vertices).float().cuda()
    obj_verts_backup = torch.tensor(obj_mesh.vertices).float().cuda()
    obj_faces = obj_mesh.faces
    if not is_tripo:
        # mirror reflection w.r.t. xy plane
        rot1 = torch.tensor([[1,0,0],
                            [0,1,0],
                            [0,0,-1]], dtype=torch.float, requires_grad=False).to(obj_verts.device)
        
        obj_verts = obj_verts @ rot1.T
        obj_faces = np.array(obj_mesh.faces[:, ::-1])
    else:
        rot1 = torch.tensor([[1,0,0],
                            [0,0,-1],
                            [0,1,0]], dtype=torch.float, requires_grad=False).to(obj_verts.device)
        
        obj_verts = obj_verts @ rot1.T
        print("Done tripo trans")
        # obj_faces = np.array(obj_mesh.faces[:, ::-1])
        
    obj_mesh = trimesh.Trimesh(obj_verts.clone().cpu().numpy(), obj_faces)
        
    obj_sdf_origin = obj_mesh.bounding_box.centroid.copy()
    obj_sdf_scale = 2.0 / np.max(obj_mesh.bounding_box.extents)
    
    obj_sdf_path = osp.join(cfg.obj_dir,img_fn, "sdf.npy")
    if os.path.exists(obj_sdf_path):
        obj_sdf_voxel = np.load(obj_sdf_path, allow_pickle=True)
        if obj_sdf_voxel.size == 1 and obj_sdf_voxel.item() is None:
            obj_sdf_voxel = None
    else:
        obj_sdf_voxel = try_until_success(
                                    mesh_to_voxels,
                                    verbose=False,
                                    mesh = obj_mesh,
                                    voxel_resolution=64, 
                                    check_result=True, 
                                    surface_point_method="sample",
                                    sample_point_count=500000,
                                )
        np.save(obj_sdf_path, obj_sdf_voxel)
        
    if obj_sdf_voxel is None:
        print("obj sdf is None: ", img_fn)
        return None
        
    obj_sdf = {"origin": torch.FloatTensor(obj_sdf_origin).cuda(),
               "scale": torch.FloatTensor([obj_sdf_scale]).cuda(),
               "voxel": torch.FloatTensor(obj_sdf_voxel).cuda()}
    
    
    ret = {
            "name": img_fn,
            "img_path": img_path,
            "resolution":[h,w],
            "image": np.array(input_image),
            "hand_mask": torch.tensor(hand_mask == 0).cuda(), # the hand mask for inpaint has zero for hand region
            "obj_mask": torch.tensor(obj_mask > 0).cuda(),
            "inpaint_mask": torch.tensor(inpaint_mask > 0).cuda(),
            "hand_cam": hand_cam,
            "obj_cam": obj_cam,
            "mano_params": mano_params,
            "object_verts": obj_verts.unsqueeze(0),
            "object_faces": torch.LongTensor(obj_mesh.faces).cuda(),
            "object_colors": object_colors,
            "object_sdf": obj_sdf,
            "cam_transl": torch.tensor(hand_info["cam_transl"]).unsqueeze(0).float().cuda(),
            "is_right": hand_info["is_right"],
            "mesh_path": osp.join(cfg.obj_dir,img_fn, "fixed.obj")
        }
    return ret
    

@hydra.main(version_base=None, config_path="./configs", config_name="optim_notip_arctic")
def main(cfg : DictConfig) -> None:
    # include_list = ["0", "2"]
    # include_list = ["jakub-zerdzicki-Y1YJm1iibTg-unsplash"]
    include_list = ["601715679424_", "591715679424_", "581715679423_", "541715679420_",
                    "521715679419_", "491715679416_", "461715679415_", "441715679413_",
                    "431715679413_"]
    
    exp_cfg = OmegaConf.create(cfg['experiments'])
    data_cfg = OmegaConf.create(cfg['data']) 
    # print(cfg)   
    if "is_tripo" in cfg:
        is_tripo = True
    else:
        is_tripo = False
        
    # print(cfg['out_dir'])
    os.makedirs(cfg['out_dir'], exist_ok=True)
    
    exp_cfg['out_dir'] = cfg['out_dir']
    exp_cfg['log_dir'] = cfg['log_dir']
    if "hand_scale" in cfg:
        exp_cfg['hand_scale'] = cfg['hand_scale']
        
    
    filtered_file = osp.join(data_cfg.base_dir,f"{data_cfg.split}_filtered.npy")
    if os.path.exists(filtered_file):
        img_id_list = np.load(filtered_file, allow_pickle=True)
        hamer_info_list = [d['hamer_info'] for d in img_id_list]
        img_id_list = [d['img_id'] for d in img_id_list]
    else:
        img_id_list = []
        hamer_info_list = []
        for file in os.listdir(data_cfg.input_dir):
            print(file)
            if not file.endswith(("png", "jpg")):
                continue
            img_id = file.split('.')[0]
            info = load_hamer_info(os.path.join(data_cfg.hand_dir, f"{img_id}.pt"))
            if info == None or len(info) == 0:
                print("No Hamer Info!")
                continue
            img_id_list.append(img_id)
            hamer_info_list.append(info)
    
    
    # img_id_list = ["1", "2", "3", "4", "5", "6", "8"]
    # img_id_list = ["0", "1", "2", "3", "4"]
    # img_id_list = ["priscilla-du-preez-YtUMg1gw_pI-unsplash","recha-oktaviani-5tYUk7sZzqc-unsplash"]
    # img_id_list = [i for i in range(30)] + [i for i in range(30, 1000, 10)]
    # img_id_list = ["230", "310", "330", "160"]
    # img_id_list = ["39", "111", "587", "360"]
    # img_id_list = ["hand_image-21", "hand_image-6", "hand_image-16"]
    # img_id_list = ["2"]
    
    for i in trange(len(img_id_list)):
        img_fn = img_id_list[i]
        file = img_fn + ".png"
        hand_infos = hamer_info_list[i]
        hand_id = None
        
        path = osp.join(exp_cfg.out_dir, f"after_{img_fn}.ply")
        lock_file = osp.join(exp_cfg.out_dir, f"after_{img_fn}.lock")
        if osp.exists(path) or osp.exists(lock_file):
            continue
        
        progress_bar = tqdm(total=exp_cfg['iteration'], desc="Processing")
        hoi_sync = HOI_Sync(cfg=exp_cfg, progress_bar=progress_bar)
        
        """ Find the best match hand """
        min_iou = 1000
        for item in hand_infos:
            data_item = load_data_single(data_cfg, file, item["id"], is_tripo)
            if data_item is None:
                print("data_item is None")
                break
            hoi_sync.get_data(data_item)
            hand_iou, obj_iou = hoi_sync.get_hamer_hand_mask()
            print(item["id"], hand_iou, obj_iou)
            if hand_iou is None or obj_iou is None:
                iou = None
            else:
                iou = hand_iou + obj_iou
                
            if iou is not None and iou < min_iou:
                min_iou = iou
                hand_id = item["id"]
            
        if hand_id is None:
            continue
        
        data_item = load_data_single(data_cfg, file, hand_id, is_tripo)
        if data_item is None:
            with open(lock_file, 'w') as f:
                f.write("Failed to construct a SDF!")
            continue
        
        print(data_item["name"])
        
        with open(lock_file, 'w') as f:
            pass  # This creates an empty file
        
        
        hoi_sync.get_data(data_item)
        
        print("get_hamer_hand_mask")
        hoi_sync.get_hamer_hand_mask()
        print("optim_obj_cam")
        hoi_sync.export_for_eval(prefix="before_camsetup")
        succ = hoi_sync.optim_obj_cam()
        if succ is False:
            continue
        hoi_sync.export(prefix="init")
        hoi_sync.export_for_eval(prefix="init")
        
        print("run_handpose, global")
        hoi_sync.run_handpose()
        hoi_sync.export(prefix="after_global")
        hoi_sync.export_for_eval(prefix="after_global")
        
        print("run_handpose, not global")
        hoi_sync.run_handpose(global_only=False)
        hoi_sync.export(prefix="after")
        hoi_sync.export_for_retarget()
        hoi_sync.export_for_eval(prefix="final")
        
        os.remove(lock_file)
        
    
if __name__ == "__main__":
    main()