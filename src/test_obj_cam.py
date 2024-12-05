import nvdiffrast.torch as dr
import os
import trimesh
import torch
import numpy as np
import time
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.cam_utils import get_projection, center_looking_at_camera_pose
from src.utils.losses import DROTLossFunction
from src.utils.geom_utils import matrix_to_axis_angle_t, axis_angle_t_to_matrix



# set for nvdiff
glctx = dr.RasterizeCudaContext()

def optimize_projections(mesh, c2ws:torch.Tensor, projections:torch.Tensor, mask_gt:torch.Tensor):
    print(mask_gt.shape)
    H, W= mask_gt.shape
    resolution = (W, H)
    params = {
        "boost": 3,
        "alpha": 0.98,
        "loss": "l1", 
        "step_size": 2e-3,
        "optimizer": torch.optim.Adam, 
        "remesh": [50,100,150], 
        "steps": 50,
    }
    
    task = {
        "resolution": resolution,
        "matching":{"matcher":"Sinkhorn","matching_interval":5,"matching_weight":1.0,"rgb_loss_weight":1.0},
    }
    opt_time = params.get("time", -1) # Optimization time (in minutes)
    steps = params.get("steps", 100) # Number of optimization steps (ignored if time > 0)
    step_size = params.get("step_size", 0.01) # Step size
    optimizer = params.get("optimizer", torch.optim.Adam) # Which optimizer to use
    device = projections.device
    
    projections_origin = projections.clone()
    projections_residual = torch.nn.Parameter(torch.zeros((4, 4), device=device, dtype=projections_origin.dtype))
    projections_mask = torch.tensor([
        [1., 0., 0., 1.], 
        [0., 1., 0., 1.], 
        [0., 0., 0., 0.], 
        [0., 0., 0., 0.],
    ]).to(projections_origin)
    
    # c2ws_origin = c2ws.clone()
    c2ws_r_orig, c2ws_t_orig, c2ws_s_orig = matrix_to_axis_angle_t(c2ws)
    c2ws_residual = torch.nn.Parameter(torch.zeros(6, device=device, dtype=c2ws.dtype))
    
    opt = optimizer([projections_residual, c2ws_residual], lr=step_size)
    
    # Set values for time and step count
    if opt_time > 0:
        steps = -1
    it = 0
    t0 = time.perf_counter()
    t = t0
    opt_time *= 60
    
    if params["loss"] == "l1":
        loss_func = torch.nn.L1Loss()
    elif params["loss"] == "OptimalTransport":
        loss_func = DROTLossFunction(
                debug=False,
                resolution=task["resolution"],
                settings=task["matching"],
                device=device,
                renderer=renderer,
                num_views=task["view"]["num"],
                logger=None)
    elif params["loss"] == "l2":
        loss_func = torch.nn.MSELoss()
        
    for i in range(500):
        projections = projections_origin + projections_residual * projections_mask
        c2ws_r = c2ws_r_orig + c2ws_residual[:3]
        c2ws_t = c2ws_t_orig + c2ws_residual[3:]
        c2ws = axis_angle_t_to_matrix(c2ws_r, c2ws_t, c2ws_s_orig)
                
        mask_opt = renderer(mesh, projections, c2ws, resolution)
        loss = loss_func(mask_opt, mask_gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    print(projections_residual.grad, c2ws_residual.grad)
        
    return projections.detach(), c2ws.detach()


def diffrender_proj(verts, cam):
    ones = torch.ones(1, verts.shape[1], 1).to(verts.device)
    pos = torch.cat((verts, ones), dim=2).float() # augumented pos
    
    if cam["extrinsics"].shape[0] == 3:
        view_matrix = torch.cat([cam["extrinsics"], torch.tensor([[0,0,0,1]], device=pos.device)], dim=0)
    else:
        view_matrix = cam["extrinsics"]
    view_matrix = torch.inverse(view_matrix)
    proj_matrix = torch.FloatTensor(cam["projection"]).to(verts.device)
    
    mat = (proj_matrix @ view_matrix).unsqueeze(0)
    # mat = proj_matrix.unsqueeze(0)
    pos_clip = pos @ mat.mT
    
    return pos_clip

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

def get_obj_cam():
    # param from instantmesh
    DEFAULT_DIST = 4.5
    DEFAULT_FOV = 30.0
    focal_length = 0.5 / np.tan(np.deg2rad(DEFAULT_FOV) * 0.5)
    # cam_pose = torch.FloatTensor([DEFAULT_DIST, 0, 0]).cuda()
    cam_pose = torch.FloatTensor([DEFAULT_DIST, 0, 0]).cuda()
    
    
    obj_cam = {'fx':focal_length, 'fy': focal_length, 'cx': 0.5, 'cy': 0.5}
    obj_cam["extrinsics"] = center_looking_at_camera_pose(cam_pose)
    obj_cam["projection"] = get_projection(obj_cam, width=1000, height=1000)
    
    return obj_cam

def renderer(mesh, projection, c2ws, resolution):
    device = projection.device
    verts = torch.tensor(mesh.vertices).unsqueeze(0).to(device)
    tri = torch.tensor(mesh.faces).int().to(device)
    
    ones = torch.ones(1, verts.shape[1], 1).to(device)
    pos = torch.cat((verts, ones), dim=2).float() # augumented pos
    
    view_matrix = torch.inverse(c2ws)
    
    mat = (projection @ view_matrix).unsqueeze(0)
    # mat = proj_matrix.unsqueeze(0)
    pos_clip = pos @ mat.mT
    
    color_obj = torch.FloatTensor([0, 1, 0]).repeat(verts.shape[1], 1)
    color_obj = color_obj.unsqueeze(0).to(device)
    
    rast, _ = dr.rasterize(glctx, pos_clip, tri, resolution)
    out, _ = dr.interpolate(color_obj, rast, tri)
    out = dr.antialias(out, rast, pos_clip, tri)
    # img = torch.flip(out[0], dims=[0]) # Flip vertically.
    img = out[0]
    mask = img[..., 1] # green channel
    
    return mask
    

def test_render_obj(data_dir, id_list, out_dir):
    image_dir = os.path.join(data_dir, "images")
    model_dir = os.path.join(data_dir, "obj_recon/results/instantmesh/instant-mesh-large/")
    mesh_path = model_dir + "meshes/{}/full.obj"

    for img_id in id_list:
        orig_img = Image.open(os.path.join(image_dir, f"{img_id}.jpg"))
        obj_mesh = trimesh.load(mesh_path.format(img_id))
        obj_cam = get_obj_cam()
        
        projection = torch.FloatTensor(obj_cam["projection"]).cuda()
        c2ws = obj_cam["extrinsics"].cuda()
        mask = renderer(obj_mesh, projection, c2ws, orig_img.size)
        
        # vis for check, can be commented
        print(mask.shape)
        mask = transforms.ToPILImage()(mask)
        mask.save(os.path.join(out_dir, f"{img_id}.png"))
        
def test_optim_obj(data_dir, id_list, out_dir):
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "obj_recon/inpaint_mask")
    model_dir = os.path.join(data_dir, "obj_recon/results/instantmesh/instant-mesh-large/")
    mesh_path = model_dir + "meshes/{}/full.obj"

    for img_id in id_list:
        orig_img = Image.open(os.path.join(image_dir, f"{img_id}.jpg"))
        gt_mask = Image.open(os.path.join(mask_dir, f"{img_id}.png")).convert("L")
        gt_mask.save(os.path.join(out_dir, f"{img_id}_gt.png"))
        
        gt_mask = transforms.ToTensor()(gt_mask).squeeze()
        gt_mask = (gt_mask>0.5).float().cuda()
        
        obj_mesh = trimesh.load(mesh_path.format(img_id))
        obj_cam = get_obj_cam()
        
        projection = torch.FloatTensor(obj_cam["projection"]).cuda()
        c2ws = obj_cam["extrinsics"].cuda()
        
        projection, c2ws = optimize_projections(obj_mesh, c2ws, projection, gt_mask)
        
        
        # vis for check, can be commented
        mask = renderer(obj_mesh, projection, c2ws, orig_img.size)
        print(mask.shape)
        mask = transforms.ToPILImage()(mask)
        mask.save(os.path.join(out_dir, f"{img_id}_optimized.png"))
        
        

if __name__ == "__main__":
    data_dir = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/"
    id_list = ["0", "1", "2", "3", "4"]
    out_dir = "./output/test_obj_cam"
    os.makedirs(out_dir, exist_ok=True)
    test_render_obj(data_dir, id_list, out_dir)
    # test_optim_obj(data_dir, id_list, out_dir)
    
