import os
import os.path as osp

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

from src.models.hoi_optim_module import HOI_Sync

from src.utils import (
    load_cam,
    get_projection,
    correct_image_orientation,
    resize_frame,
)

def image_process(input_image, hand_mask, obj_mask, inpaint_mask):
    input_image = correct_image_orientation(input_image)
    hand_mask = correct_image_orientation(hand_mask).convert('L') # grayscale
    obj_mask = correct_image_orientation(obj_mask).convert('L')
    inpaint_mask = correct_image_orientation(inpaint_mask).convert('L')
    
    input_image, new_size = resize_frame(input_image)
    print(new_size)
    hand_mask = hand_mask.resize(new_size)
    obj_mask = obj_mask.resize(new_size)
    inpaint_mask = inpaint_mask.resize(new_size)
    
    hand_mask, obj_mask, inpaint_mask = np.array(hand_mask), np.array(obj_mask), np.array(inpaint_mask)
    
    return input_image, hand_mask, obj_mask, inpaint_mask

def load_data(cfg: DictConfig, include_list: list):
    ret_list = []
    for file in os.listdir(cfg.input_dir):
        if not file.endswith((".png", "jpg", "jpeg")):
            continue
        img_fn = file.split(".")[0]
        if len(include_list)>0 and img_fn not in include_list:
            continue
        print(img_fn)
        
        # resize the input image, because the resolution for nvdiffrastmust must be [<=2048, <=2048]
        input_image = Image.open(osp.join(cfg.input_dir, file))
        hand_mask = Image.open(osp.join(cfg.mask_dir, f"{img_fn}_hand.png"))
        obj_mask = Image.open(osp.join(cfg.mask_dir, f"{img_fn}_obj.png"))
        inpaint_mask = Image.open(osp.join(cfg.inpaint_dir, f"{img_fn}.png"))
        origin_w, origin_h = input_image.width, input_image.height
        
        input_image, hand_mask, obj_mask, inpaint_mask = image_process(input_image, hand_mask, obj_mask, inpaint_mask)
        
        w,h = input_image.width, input_image.height
        
        obj_cam_file = osp.join(cfg.obj_dir, f"{img_fn}_cam.json")
        hand_cam_file = osp.join(cfg.hand_dir, f"{img_fn}_cam.json")
        
        # obj_mesh = trimesh.load(osp.join(cfg.obj_dir, f"{img_fn}.ply"))
        simp_path = osp.join(cfg.obj_dir,img_fn, "simplified.obj")
        # if osp.exists(simp_path):
        #     obj_mesh = trimesh.load(simp_path)
        # else:
        if True:
            obj_mesh = trimesh.load(osp.join(cfg.obj_dir,img_fn, "fixed.obj"))
            target_faces = 50000
            if obj_mesh.faces.shape[0] > target_faces:
                success, v_decimated, f_decimated, _, _= igl.decimate(np.array(obj_mesh.vertices), 
                                                        np.array(obj_mesh.faces), 
                                                        target_faces)
                if not success:
                    print("Mesh simplication error")
                else:
                    print("v_decimated.shape: ", v_decimated.shape)
                    print("f_decimated.shape: ", f_decimated.shape)
                    obj_mesh = trimesh.Trimesh(vertices=v_decimated, faces=f_decimated)
            obj_mesh.export(simp_path)
        
        max_valid = 0
        max_n = 0
        info_list = []
        for n in range(2):
            path = osp.join(cfg.hand_dir, f"{img_fn}_{n}.pt")
            if not osp.exists(path):
                break
            info = torch.load(path)
            info_list.append(info)
            if info['valid'] > max_valid:
                max_valid = info['valid']
                max_n = n
        hand_info = info_list[max_n]
        print("right hand" if int(hand_info['is_right'].item()) else "left hand")
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mano_params = hand_info["mano_params"]
        for key in mano_params:
            mano_params[key] = torch.tensor(mano_params[key]).to(device).unsqueeze(0)
            print(key, mano_params[key].shape)
        
        hand_cam = load_cam(hand_cam_file, device=device)
        hand_cam["projection"] = torch.tensor(get_projection(hand_cam, origin_w, origin_h)).float().to(device)
        
        if os.path.exists(obj_cam_file):
            obj_cam = load_cam(obj_cam_file, device=device)
            obj_cam["projection"] = torch.tensor(get_projection(obj_cam, origin_w, origin_h)).float().to(device)
        else:
            obj_cam = None
        
        
        obj_verts = torch.tensor(obj_mesh.vertices).unsqueeze(0).float().cuda()
        
        # Reflection transformation w.r.t. x-axis
        rot1 = torch.tensor([[1,0,0],
                            [0,-1,0],
                            [0,0,-1]], dtype=torch.float, requires_grad=False).to(obj_verts.device)
        # Rotate 90 degrees clockwise around the y-axis.
        rot2 = torch.tensor([[0,0,1],
                             [0,1,0],
                            [-1,0,0]], dtype=torch.float, requires_grad=False).to(obj_verts.device)
        obj_verts = obj_verts @ (rot2 @ rot1).T
        obj_verts = obj_verts - obj_verts.mean(dim=1, keepdim=True)
        obj_verts = obj_verts * 0.1
        # mirror transformation
        
        ret = {
                "name": img_fn,
                "resolution":[h,w],
                "image": input_image,
                "hand_mask": torch.tensor(hand_mask > 0).cuda(),
                "obj_mask": torch.tensor(obj_mask > 0).cuda(),
                "inpaint_mask": torch.tensor(inpaint_mask > 0).cuda(),
                "hand_cam": hand_cam,
                "obj_cam": obj_cam,
                "mano_params": mano_params,
                "object_verts": obj_verts,
                "object_faces": torch.tensor(obj_mesh.faces).unsqueeze(0).cuda(),
                "cam_transl": torch.tensor(hand_info["cam_transl"]).unsqueeze(0).float().cuda(),
                "is_right": hand_info["is_right"],
                "mesh_path": osp.join(cfg.obj_dir,img_fn, "fixed.obj")
            }
        ret_list.append(ret)
    return ret_list

@hydra.main(version_base=None, config_path="./configs", config_name="optim")
def main(cfg : DictConfig) -> None:
    # include_list = ["0", "2"]
    include_list = []
    
    exp_cfg = OmegaConf.create(cfg['experiments'])
    data_cfg = OmegaConf.create(cfg['data'])    
    
    data_list = load_data(data_cfg, include_list)
    
    for data_item in data_list:
        print(data_item["name"])
        progress_bar = tqdm(total=exp_cfg['iteration'], desc="Processing")
        hoi_sync = HOI_Sync(cfg=exp_cfg, progress_bar=progress_bar)
        hoi_sync.get_data(data_item)
        hoi_sync.run()
        hoi_sync.export()
        hoi_sync.run_handpose()
        hoi_sync.export(prefix="after")
        hoi_sync.export_for_sim()
        print(hoi_sync.global_params)
        
    
if __name__ == "__main__":
    main()