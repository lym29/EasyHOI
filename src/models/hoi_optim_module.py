from typing import Any, Dict, Tuple
from yacs.config import CfgNode
import pickle
import json
import os
import os.path as osp
import sys
import time
from PIL import Image
import trimesh
import torch
from torch import optim, nn, utils, Tensor
from torchvision import transforms

from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops.knn import knn_gather, knn_points

from chamfer_distance import ChamferDistance
from geomloss import SamplesLoss

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# sys.path.append('third_party/hamer')

from src.utils.losses import (
    compute_obj_contact_loss, 
    soft_iou_loss, 
    compute_nonzero_distance_2d,
    chamfer_dist_loss, 
    anatomy_loss, 
    compute_penetr_loss, 
    compute_h2o_sdf_loss,
    # DROTLossFunction,
    compute_sinkhorn_loss,
    compute_sinkhorn_loss_rgb,
    compute_depth_loss,
    compute_obj_contact,
    compute_hand_contact,
    icp_with_scale,
    statistical_outlier_removal,
    moment_based_comparison
)

from src.utils.cam_utils import verts_transfer_cam, center_looking_at_camera_pose, get_projection
from src.utils.mesh_utils import render_mesh, pc_to_sphere_mesh
from src.utils import geom_utils, hand_utils, image_utils


from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.axislayer import AxisAdaptiveLayer, AxisLayerFK
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.anchorlayer import AnchorLayer

import nvdiffrast.torch as dr

ToPILImage = transforms.ToPILImage()

class HOI_Sync:
    def __init__(self, cfg:CfgNode, progress_bar):
        super().__init__()
        # Instantiate MANO model
        self.cfg = cfg
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        # self.mano = MANO(**mano_cfg).cuda()
        self.tip_ids = [745, 317, 444, 556, 673]
        # self.mano.eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        """ for optim """
        self.L1Loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.sinkhorn_loss = SamplesLoss('sinkhorn')
        self.axisFK = AxisLayerFK(mano_assets_root="assets/mano").to(self.device)
        self.anchor_layer = AnchorLayer(anchor_root="assets/anchor").to(self.device)
        self.anatomyLoss = AnatomyConstraintLossEE().to(self.device)
        self.anatomyLoss.setup()
        # self.chd_loss = ChamferDistance()
        self.loss_weights = {k: float(v) for k, v in dict(cfg.weights).items()}
        self.param_dim = {'hand':[('scale',1), ('transl',3)],
                          'obj': [('scale',1), ('transl',3), ('orient',3)]}
        if "hand_scale" not in cfg:
            self.global_params = {
                'hand': torch.FloatTensor([5, 0, 0, 0]).to(self.device),
                'obj': torch.FloatTensor([1, 0,0,0, 0,0,0]).to(self.device),
            }
        else:
            self.global_params = {
                'hand': torch.FloatTensor([cfg.hand_scale, 0, 0, 0]).to(self.device),
                'obj': torch.FloatTensor([1, 0,0,0, 0,0,0]).to(self.device),
            }
            
        
        for key in self.global_params:
            self.global_params[key].requires_grad_(True)
        
        """ for log """
        self.global_step = 0
        self.progress_bar = progress_bar
        
        """ for render """
        self.glctx = dr.RasterizeCudaContext()
        
        """ for obj cam optim """
        self.phi_center = None
        self.phi_range = None
        
        """ for mano template """
        self.mano_layer = ManoLayer(side="right").to(self.device)
        with open("assets/mano_backface_ids.pkl", "rb") as f:
            self.hand_backface_ids = pickle.load(f)
            
        contact_zone = np.load("assets/contact_zones.pkl", allow_pickle=True)['contact_zones']
        self.hand_contact_zone = []
        for key in contact_zone:
            self.hand_contact_zone += contact_zone[key]
        
        
        """ for export """
        self.vis_mid_results = True
        if self.vis_mid_results:
            os.makedirs(osp.join(self.cfg.out_dir, "./midresult/test_hand_obj_cam"), exist_ok=True)
            os.makedirs(osp.join(self.cfg.out_dir, "./midresult/test_obj_cam"), exist_ok=True)
            os.makedirs(osp.join(self.cfg.out_dir, "./midresult/test_hand_cam"), exist_ok=True)
            
            
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        os.makedirs(osp.join(self.cfg.out_dir, "render"), exist_ok=True)
        os.makedirs(osp.join(self.cfg.out_dir, "eval"), exist_ok=True)
        os.makedirs(osp.join(self.cfg.out_dir, "vis"), exist_ok=True)
        os.makedirs(osp.join(self.cfg.out_dir, "retarget"), exist_ok=True)
        os.makedirs(osp.join(self.cfg.out_dir, "contact"), exist_ok=True)

    
    def get_params_for(self, option):
        key = option
        res = {}
        offset = 0
        for pair in self.param_dim[option]:
            name, dim = pair
            res[name] = self.global_params[key][offset:offset+dim]
            offset += dim
        return res
        
    def get_data(self, data_item, **kwarg):
        self.data = data_item
        self.mano_params = data_item["mano_params"]
        
        self.hand_faces = self.mano_layer.get_mano_closed_faces().to(self.device)
        if not self.data["is_right"]:
            self.hand_faces = self.hand_faces[:,[0,2,1]] # faces for left hand
        
        fullpose = torch.cat([self.mano_params["global_orient"], self.mano_params["hand_pose"]], dim=1)
        self.mano_params['fullpose'] = matrix_to_axis_angle(fullpose).reshape(-1, 16*3) #[B, 16* 3]
        
        
    def log(self, value_dict, step, log_dir="./logs/optim", tag="optim"):
        output = ""
        for key in value_dict:
            if isinstance(value_dict[key], torch.Tensor):
                output += f"{key}:{value_dict[key].item():.4e};"
            else:
                output += f"{key}:0;"
        self.progress_bar.set_description(output)
        self.progress_bar.update(step)
    
    def get_mano_output(self):
        mano_params = self.mano_params
        fullpose = mano_params['fullpose']
        betas = mano_params['betas']
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        return mano_output
    
    def diffrender_proj(self, pos, cam):
        ones = torch.ones(1, pos.shape[1], 1).to(pos.device)
        pos = torch.cat((pos, ones), dim=2).float() # augumented pos
        
        view_matrix = torch.cat([cam["extrinsics"], torch.tensor([[0,0,0,1]], device=pos.device)], dim=0)
        view_matrix = torch.inverse(view_matrix)
        proj_matrix = cam["projection"]
        
        mat = (proj_matrix @ view_matrix).unsqueeze(0)
        # mat = proj_matrix.unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        return pos_clip
    
    def get_hamer_hand_mask(self, enlargement=20):
        mano_output = self.get_mano_output()
        hand_verts = self.get_hand_for_handcam(mano_output.verts, scale=1., transl=torch.zeros(3, device=self.device))
        
        pos_clip = self.diffrender_proj(hand_verts, self.data["hand_cam"])
        tri = self.hand_faces.squeeze().int()
        
        color = torch.tensor([1, 0, 0]).repeat(hand_verts.shape[1], 1)
        
        color = color.unsqueeze(0).float().to(hand_verts.device)
        
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution=self.data["resolution"])
        out, _ = dr.interpolate(color, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)
        img = torch.flip(out[0], dims=[0]) # Flip vertically.
        
        self.data["hamer_hand_mask"] = img[...,0].detach()
        
        tgt_mask = self.data["hamer_hand_mask"]
        # Get the indices of non-zero elements
        indices = torch.nonzero(tgt_mask, as_tuple=False)
        if len(indices) == 0:
            return None, None
        min_row, min_col = indices.min(dim=0)[0]
        max_row, max_col = indices.max(dim=0)[0]

        # Enlarge the bounding box
        min_row = max(0, min_row - enlargement)
        min_col = max(0, min_col - enlargement)
        max_row = min(tgt_mask.shape[0] - 1, max_row + enlargement)
        max_col = min(tgt_mask.shape[1] - 1, max_col + enlargement)

        # Zero out pixels outside the enlarged bounding box in the hand mask
        modified_hand_mask = torch.zeros_like(self.data["hand_mask"])
        modified_hand_mask[min_row:max_row+1, min_col:max_col+1] = self.data["hand_mask"][min_row:max_row+1, min_col:max_col+1]
        self.data["hand_mask"] = modified_hand_mask
        
        hand_iou = soft_iou_loss(self.data["hamer_hand_mask"], self.data["hand_mask"])
        o2h_dist = compute_nonzero_distance_2d(self.data["hamer_hand_mask"], 
                                              self.data["inpaint_mask"])
        
        if self.vis_mid_results:
            name = self.data['name']
            
            mask = self.data["hamer_hand_mask"].cpu().numpy()
            mask = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            Image.fromarray(mask).save(osp.join(self.cfg.out_dir, f"midresult/test_hand_cam/{name}_hamer.png"))
            
            mask = self.data["hand_mask"].cpu().numpy()
            mask = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            Image.fromarray(mask).save(osp.join(self.cfg.out_dir, f"midresult/test_hand_cam/{name}_seg.png"))
            
        return hand_iou.item(), o2h_dist.item()
            
    
    def render_hoi_image(self, 
                    hand_verts, hand_faces, 
                    obj_verts, obj_faces,
                    resolution,
                    hand_color=None,
                    ):
        vtx_offset = hand_verts.shape[1]
        verts = torch.cat([hand_verts, obj_verts], dim=1)
        tri = torch.cat([hand_faces, obj_faces + vtx_offset], dim=0).int()
        # tri = hand_faces
        if hand_color is None:
            col_hand = torch.tensor([1, 0, 0]).repeat(hand_verts.shape[1], 1)
        else:
            col_hand = hand_color
        col_obj = torch.tensor([0, 1, 0]).repeat(obj_verts.shape[1], 1)
        color = torch.cat((col_hand, col_obj), dim=0).to(self.device) # color for each vertex
        color = color.unsqueeze(0).float()
        
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        if c2ws.shape[0] == 3:
            c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        
        img = self.renderer(verts, tri, color, projections, c2ws, resolution)
        
        return img
    
    def render_hand_image(self, hand_verts):
        color_hand = torch.FloatTensor([1, 0, 0]).repeat(hand_verts.shape[1], 1)
        color_hand = color_hand.unsqueeze(0).to(hand_verts.device)
        
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        
        if c2ws.shape[0] == 3:
            c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        
        img = self.renderer(hand_verts, self.hand_faces.squeeze().int(), color_hand, projections, c2ws, self.data["resolution"])
        return img
        
    
    def uniform_sample_objcam(self, center=None, phi_range=None, n_phi=10):
        # param from instantmesh
        DEFAULT_DIST = 4.5
        
        if center is None:
            center = torch.pi/2
        if phi_range is None:
            phi_range = torch.pi/6
        
        phi = torch.linspace(center - phi_range/2, center + phi_range/2, n_phi, 
                            device=self.device, dtype=torch.float32)
        theta = torch.FloatTensor([0]).to(self.device)
        theta = torch.linspace( -phi_range, phi_range, n_phi, 
                            device=self.device, dtype=torch.float32)
        
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        
        directions = torch.stack((x, y, z), dim=1)
        cam_pose = DEFAULT_DIST * directions
        
        c2ws = center_looking_at_camera_pose(cam_pose)
        return c2ws, phi, phi_range
    
    def find_c2ws_init(self, verts, tri, color_obj, projections, resolution, gt_obj_mask, max_depth=3):
        phi_center = self.phi_center
        phi_range = self.phi_range
        for depth in range(max_depth):
            c2ws_list, phi_list, phi_range = self.uniform_sample_objcam(phi_center, phi_range, n_phi=10)
            min_metric = torch.inf
            best_id = 0
            # gt_obj_mask = gt_obj_mask.cpu().numpy().astype(bool)
            
            for i in range(c2ws_list.shape[0]):
                c2ws = c2ws_list[i]
                img = self.renderer(verts, tri, color_obj, projections, c2ws, resolution=resolution)
                mask_opt = img[..., 1] # green channel
                if mask_opt.sum() > 0:
                    with torch.no_grad():
                        metric = compute_sinkhorn_loss(mask_opt, gt_obj_mask)
                else:
                    metric = torch.inf
                
                if metric < min_metric:
                    best_id = i
                    min_metric = metric
            print("best_id: ", best_id, "phi: ", phi_list[best_id])
            c2ws = c2ws_list[best_id]
            phi_center = phi_list[best_id]
            phi_range = phi_range / 2
            
        self.phi_center = phi_center
        # self.phi_range = phi_range
            
        return c2ws
        
    
    def optim_obj_cam(self):
        
        gt_obj_mask = self.data["inpaint_mask"].float()
        # gt_obj_mask = self.data["obj_mask"].float()
        print(gt_obj_mask.shape)
        resolution = gt_obj_mask.shape
        params = {
            "boost": 3,
            "alpha": 0.98,
            "loss": "IoU", #"sinkhorn", 
            "step_size": 1e-2,
            "optimizer": torch.optim.Adam, 
            "remesh": [50,100,150], 
        }
        
        verts = self.data["object_verts"]
        tri = self.data["object_faces"].int()
        color_obj = torch.FloatTensor([0, 1, 0]).repeat(verts.shape[1], 1)
        color_obj = color_obj.unsqueeze(0).to(self.device)
        
        step_size = params.get("step_size") # Step size
        optimizer = params.get("optimizer", torch.optim.Adam) # Which optimizer to use
        
        
        projections = self.data["obj_cam"]["projection"]
        
        c2ws = self.data["obj_cam"]["extrinsics"]
        
        
        device = projections.device
        projections_origin = projections.clone()
        projections_residual = torch.nn.Parameter(torch.zeros((4, 4), device=device, dtype=projections_origin.dtype))
        # projections_residual = torch.nn.Parameter(torch.zeros((1), device=device, dtype=projections_origin.dtype))
        projections_mask = torch.tensor([
                [1., 0., 0., 0.], 
                [0., 1., 0., 0.], 
                [0., 0., 0., 0.], 
                [0., 0., 0., 0.]
            ], device=device, dtype=projections.dtype)
        
        # c2ws_origin = c2ws.clone()
        c2ws_residual = torch.nn.Parameter(torch.zeros(6, device=device, dtype=c2ws.dtype))
        opt = optimizer([projections_residual, c2ws_residual], lr=step_size)
        
        if params["loss"] == "l1":
            loss_func = torch.nn.L1Loss()
        elif params["loss"] == "l2":
            loss_func = torch.nn.MSELoss()
        elif params["loss"] == "IoU":
            loss_func = soft_iou_loss
        else:
            loss_func = compute_sinkhorn_loss
            
            
        print("obj_iteration: ", self.cfg['obj_iteration'])
        
        c2ws_r_orig, c2ws_t_orig, c2ws_s_orig = geom_utils.matrix_to_axis_angle_t(c2ws)
        
        
        # c2ws = self.find_c2ws_init(verts, tri, color_obj, projections, resolution, gt_obj_mask)
        # c2ws_r_orig, c2ws_t_orig, c2ws_s_orig = geom_utils.matrix_to_axis_angle_t(c2ws)
            
        for i in range(self.cfg['obj_iteration']):
            projections = projections_origin + projections_residual * projections_mask
            c2ws_r = c2ws_r_orig + c2ws_residual[:3] * 0.1 # control the step of rot
            c2ws_t = c2ws_t_orig + c2ws_residual[3:]
            c2ws = geom_utils.axis_angle_t_to_matrix(c2ws_r, c2ws_t, c2ws_s_orig)

            img = self.renderer(verts, tri, color_obj, projections, c2ws, resolution=resolution)
            mask_opt = img[..., 1] # green channel
            
            if i==0:
                mask_init = mask_opt.clone()
                
            # if not torch.any(mask_opt>0):
            #     return False
            
            iou_loss = loss_func(mask_opt, gt_obj_mask)
            
            if iou_loss > 0.9:
                sinkhorn_loss = compute_sinkhorn_loss(mask_opt.contiguous(), gt_obj_mask.contiguous())
                loss = sinkhorn_loss + iou_loss
            else:
                sinkhorn_loss = 0
                loss = iou_loss
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            self.log({
                "sinkhorn loss": sinkhorn_loss,
                "total loss": loss
                }, step=i
            )
            
            if loss.item() < 0.2:
                break
        
        self.data["obj_cam"]["projection"] = projections.detach()
        self.data["obj_cam"]["extrinsics"] = c2ws.detach()[:3, :] # (3, 4)
        
        object_depth, object_rast = self.depth_peel(verts, tri, projections, c2ws, resolution,
                                            znear=0.1, zfar=100)  
        self.object_depth = object_depth.squeeze().detach() # [num_layers, H, W]
        self.object_rast = object_rast.squeeze().detach() # [num_layers, H, W, 4]
        
        obj_mesh = trimesh.Trimesh(vertices=verts.squeeze().cpu().numpy(), 
                                   faces=tri.squeeze().cpu().numpy())
        normals = torch.tensor(obj_mesh.vertex_normals).float().to(self.device)
        
        
        hoi_mask = self.data["hamer_hand_mask"].bool() & self.data["inpaint_mask"]
        front_mask = (hoi_mask & self.data["inpaint_mask"] & (~self.data["obj_mask"])).int()
        obj_pts_front, obj_contact_normals_front, contact_mask_front = compute_obj_contact(
            side='obj_front',
            mask=front_mask,
            verts=verts.squeeze(),
            faces=tri.squeeze(),
            normals=normals,
            rast=self.object_rast
        )
        
            
        back_mask = (hoi_mask & self.data["inpaint_mask"] & self.data["hamer_hand_mask"].bool() & (~self.data["hand_mask"])).int()
        obj_pts_back, obj_contact_normals_back, contact_mask_back = compute_obj_contact(
            side='obj_back',
            mask=back_mask,
            verts=verts.squeeze(),
            faces=tri.squeeze(),
            normals=normals,
            rast=self.object_rast
        )
        
        if obj_pts_front is not None and obj_pts_back is not None:
            obj_pts = torch.concat([obj_pts_front, obj_pts_back], dim=0)
            obj_contact_normals = torch.concat([obj_contact_normals_front, obj_contact_normals_back], dim=0)
        else:
            obj_pts = None
            obj_contact_normals = None
            
        self.obj_contact = {'front': obj_pts_front, 
                            'back': obj_pts_back,
                            'both': obj_pts}
        self.obj_contact_normals = {'front': obj_contact_normals_front, 
                                    'back': obj_contact_normals_back,
                                    'both': obj_contact_normals}
        
        if self.vis_mid_results:
            # vis for check, can be commented
            img_id = self.data["name"]
            depth = self.object_depth.unsqueeze(1) # [N, 1, H, W]
            image_utils.save_depth(depth, 
                                fname=os.path.join(self.cfg.out_dir, f"midresult/test_obj_cam/{img_id}_depth"),
                                text_list=["layer_0", "layer_1", "layer_2", "layer_3"],
                                )
            
            mask = transforms.ToPILImage()(mask_opt)
            mask.save(os.path.join(self.cfg.out_dir, f"midresult/test_obj_cam/{img_id}_optimized.png"))
            mask = transforms.ToPILImage()(mask_init)
            mask.save(os.path.join(self.cfg.out_dir, f"midresult/test_obj_cam/{img_id}_init.png"))
            gt_obj_mask = transforms.ToPILImage()(gt_obj_mask)
            gt_obj_mask.save(os.path.join(self.cfg.out_dir, f"midresult/test_obj_cam/{img_id}_gt.png"))
            
            contact_mask_img = ToPILImage(front_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.cfg.out_dir, "contact", f"{img_id}_obj_mask_front.png"))
            
            contact_mask_img = ToPILImage(back_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.cfg.out_dir, "contact", f"{img_id}_obj_mask_back.png"))
            
            # output the object mesh with contact point
            verts = verts.squeeze().cpu().numpy()
            
            if obj_pts_front is not None:
                obj_pts_front = pc_to_sphere_mesh(obj_pts_front.cpu().numpy())
                num_front = obj_pts_front.vertices.shape[0]
                mesh = obj_mesh + obj_pts_front
            else:
                obj_pts_front = trimesh.Trimesh(vertices=[], faces=[])
                num_front = 0
                mesh = obj_mesh
            if obj_pts_back is not None:
                obj_pts_back = pc_to_sphere_mesh(obj_pts_back.cpu().numpy())
                mesh = mesh + obj_pts_back
            else:
                obj_pts_back = trimesh.Trimesh(vertices=[], faces=[])

            vertex_colors = np.ones((len(mesh.vertices), 4))  # [R, G, B, A]
            vertex_colors[len(verts):len(verts)+num_front, :] = [1.0, 0.0, 0.0, 1.0]  # Red
            vertex_colors[len(verts)+num_front:, :] = [0.0, 1.0, 0.0, 1.0]  # Green
            mesh.visual.vertex_colors = vertex_colors
            mesh.export(os.path.join(self.cfg.out_dir, "contact", f"{img_id}_obj.ply"))
            
        
    def depth_peel(self, verts, tri, projection, c2ws, resolution, num_layers=4, znear=0.1, zfar=100):
        device = projection.device
        
        ones = torch.ones(1, verts.shape[1], 1).to(device)
        pos = torch.cat((verts, ones), dim=2).float() # augumented pos
        
        view_matrix = torch.inverse(c2ws)
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        depth_list = []
        rast_list = []
        with dr.DepthPeeler(self.glctx, pos_clip, tri, resolution) as peeler:
            for i in range(num_layers):
                rast, _ = peeler.rasterize_next_layer()
                rast = torch.flip(rast, dims=[1]) # Flip vertically.
                
                # rast has shape [minibatch_size, height, width, 4] 
                # and contains the main rasterizer output in order (u, v, z/w, triangle_id)
                depth = rast[..., 2] # [minibatch_size, H, W]
                mask = (depth == 0)
                depth = (2 * znear * zfar) / (zfar + znear - (zfar - znear) * depth)
                depth[mask] = 0
                
                depth_list.append(depth) 
                rast_list.append(rast)
                
        multi_depth = torch.stack(depth_list, dim=0) # [num_layers, minibatch_size, H, W]
        multi_rast = torch.stack(rast_list, dim=0) # [num_layers, minibatch_size, H, W, 4]
        return multi_depth, multi_rast
        
    
    def renderer(self, verts, tri, color, projection, c2ws, resolution):
        device = projection.device
        
        ones = torch.ones(1, verts.shape[1], 1).to(device)
        pos = torch.cat((verts, ones), dim=2).float() # augumented pos
        
        try:
            view_matrix = torch.inverse(c2ws)
        except:
            view_matrix = torch.linalg.pinv(c2ws)
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
        out, _ = dr.interpolate(color, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)
        img = torch.flip(out[0], dims=[0]) # Flip vertically.
        
        return img
    
    def optimize_pca(self, fullpose, betas, mano_layer, num_iterations=500, learning_rate=0.01):
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        gt_verts = mano_output.verts
        gt_joints = mano_output.joints
        
        pca_params = torch.zeros([1,10], dtype=fullpose.dtype, device=fullpose.device, requires_grad=True)
        optimizer = optim.Adam([pca_params], lr=learning_rate)

        for i in range(num_iterations):
            optimizer.zero_grad()
            
            new_pose = torch.concat([fullpose[:,:3], pca_params], dim=-1)
            mano_output: MANOOutput = mano_layer(new_pose, betas)
            pred_verts = mano_output.verts
            pred_joints = mano_output.joints
            
            verts_loss = torch.mean((pred_verts - gt_verts) ** 2)
            joints_loss =  torch.mean((pred_joints - gt_joints) ** 2)
            
            loss = verts_loss + joints_loss
            
            loss.backward()
            optimizer.step()
            
            self.log({
                "verts loss": verts_loss,
                "joints loss": joints_loss
            }, step=i)

        # Return the optimized PCA parameters
        new_pose = torch.concat([fullpose[:,:3], pca_params], dim=-1)
        
        if self.vis_mid_results:
            os.makedirs(os.path.join(self.cfg.out_dir, "pca"), exist_ok=True)
            name = self.data['name']
            mesh = trimesh.Trimesh(pred_verts.detach().squeeze().cpu(), self.hand_faces.cpu())
            mesh.export(os.path.join(self.cfg.out_dir, "pca" , f"{name}_pca_hand.ply"))
            
            mesh = trimesh.Trimesh(gt_verts.detach().squeeze().cpu(), self.hand_faces.cpu())
            mesh.export(os.path.join(self.cfg.out_dir, "pca" , f"{name}_gt_hand.ply"))
            
        return new_pose.detach()
    
    def optim_handpose(self, pca_params, pca_params_orig, betas, mano_layer=None):
        if mano_layer is None:
            mano_output: MANOOutput = self.mano_layer(pca_params, betas)
        else:
            mano_output: MANOOutput = mano_layer(pca_params, betas)
            
        hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        anchors = self.anchor_layer(hand_verts)
        gt_hand_mask = self.data["hand_mask"].float()
        gt_obj_mask = self.data["obj_mask"].float()
        
        """
        penetration and contact loss
        """
                
        penetr_loss, contact_loss = compute_h2o_sdf_loss(self.data["object_sdf"], 
                                            hand_verts,
                                            self.hand_contact_zone)
        
        # contact_loss = compute_obj_contact_loss(self.obj_pts.unsqueeze(0),
        #                                         hand_verts)
        
        contact_loss = self.loss_weights["contact"] * contact_loss
        penetr_loss = self.loss_weights["penetr"] * penetr_loss 
        loss_3d = (contact_loss + penetr_loss)
        
        """
        regularize loss
        """
        # reg_loss = anatomy_loss(mano_output, self.axisFK, self.anatomyLoss)
        # reg_loss = self.loss_weights["regularize"] * anatomy_loss(mano_output, self.axisFK, self.anatomyLoss)
        reg_loss = self.loss_weights["regularize"] * self.L1Loss(pca_params, pca_params_orig)
            
        """
        loss for hand mask under differentialable rendering
        """
        img = self.render_hoi_image(hand_verts=hand_verts,
                                hand_faces=self.hand_faces.squeeze(),
                                obj_verts=obj_verts,
                                obj_faces=self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
        
        pred_hand_mask = img[..., 0]
        
        pred_obj_mask = img[..., 1]
        
        if not torch.any(pred_hand_mask>0):
            loss_2d = 0
        else:
            # sinkhorn_loss = compute_sinkhorn_loss(pred_hand_mask.contiguous(), gt_hand_mask.contiguous()) + compute_sinkhorn_loss(pred_obj_mask.contiguous(), gt_obj_mask.contiguous())
            loss_2d = soft_iou_loss(pred_hand_mask, gt_hand_mask) #+ soft_iou_loss(pred_obj_mask, gt_obj_mask)
            # loss_2d = self.L1Loss(pred_hand_mask, gt_hand_mask) + self.L1Loss(pred_obj_mask, gt_obj_mask)
        
        loss_2d = self.loss_weights["loss_2d"] * loss_2d
        
        loss = (loss_3d +loss_2d +reg_loss)
        self.log({
                "contact": contact_loss, 
                "penetr": penetr_loss,
                "reg loss": reg_loss,
                "2d loss": loss_2d,
                "total loss": loss}, step=self.global_step)
        fullpose = mano_output.full_poses
        return loss, fullpose.detach(), pred_hand_mask.detach(), pred_obj_mask.detach()
    
    def optim_handpose_global(self, fullpose, betas, scale=None, use_3d_loss = False):
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        if scale is None:
            hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        else:
            params = {k: v.clone() for k, v in self.get_params_for('hand').items()}
            params['scale'] = scale
            hand_verts = self.get_hand_verts(mano_output.verts, **params)
        
        info = {
            "mano_verts": mano_output.verts, 
            **self.get_params_for('hand'),
            "hand_verts": hand_verts, 
        }
        
        info = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in info.items()}
        
        """
        loss for hand mask under differentialable rendering
        """
        
        gt_hand_mask = self.data["hamer_hand_mask"].float() # no obj rendered
        
        img = self.render_hand_image(hand_verts)
        pred_hand_mask = img[...,0]
        
        if not torch.any(pred_hand_mask>0):
            loss_2d = 0
            iou = torch.Tensor([1.0])
        else:
            iou = soft_iou_loss(pred_hand_mask, gt_hand_mask)
            if iou.item() >= 0.99:
                sinkhorn_loss = compute_sinkhorn_loss(pred_hand_mask.contiguous(), gt_hand_mask.contiguous())
                loss_2d = sinkhorn_loss
            else:
                loss_2d = 10 * iou 
        
        if use_3d_loss:
            penetr_loss, contact_loss = compute_h2o_sdf_loss(self.data["object_sdf"], 
                                                hand_verts,
                                                self.hand_contact_zone)
            loss_3d = (contact_loss * 10 + penetr_loss)
            loss = loss_2d + loss_3d
            self.log({"iou": iou,
                    "2d mask loss": loss_2d,
                    "3d loss": loss_3d
                    }, step=self.global_step)
        else:
            loss = loss_2d 
            self.log({"iou": iou,
                    "2d mask loss": loss_2d,
                    # "3d loss": loss_3d
                    }, step=self.global_step)

        return loss, pred_hand_mask, info, iou.item()
    
    def run_handpose_refine(self):  
        """ Fix object pose, optimize hand pose"""
        # init param
        fullpose:torch.Tensor = self.mano_params['fullpose'].detach().clone()
        betas:torch.Tensor = self.mano_params['betas'].clone()
        hand_layer = ManoLayer(use_pca=True, ncomps=10).to(self.device)
        
        pca_pose = self.optimize_pca(fullpose, betas, hand_layer)
        fullpose_residual = torch.nn.Parameter(torch.zeros_like(pca_pose))
        fullpose_mask = torch.ones_like(pca_pose)
        fullpose_mask[:, :3] = 0
        fullpose_residual.requires_grad_()
        betas.requires_grad_()
        
        params_group = [
            {'params': self.global_params['hand'], 'lr': 1e-2},
            {'params': fullpose_residual, 'lr': 1e-2},
            {'params': betas, 'lr': 1e-4},
        ]
        outer_iteration = 10
        
        self.optimizer = optim.Adam(params_group)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5) 
        
        num_iterations = self.cfg['iteration']
        _, _, init_hand_mask, _ = self.optim_handpose(pca_pose, pca_pose, betas, hand_layer)
            
        best_loss = float('inf')
        best_fullpose = None
        best_global_param = None
        for iteration in range(outer_iteration * num_iterations):
            self.optimizer.zero_grad()
            pcapose_new = pca_pose + fullpose_residual * fullpose_mask
            loss, fullpose_new, pred_hand_mask, pred_obj_mask = self.optim_handpose(pcapose_new, pca_pose, betas, hand_layer)
                            
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.global_step = iteration
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_fullpose = fullpose_new.detach().clone()
                best_global_param = self.global_params['hand'].detach().clone()

        self.mano_params['fullpose'] = best_fullpose.detach()
        self.mano_params['betas'] = betas.detach()
        self.global_params['hand'] = best_global_param
        
        name = self.data['name']
        
        # vis for check, can be commented
        if self.vis_mid_results:
            pred_hand_mask = pred_hand_mask.cpu()
            pred_hand_mask = ToPILImage(pred_hand_mask)
            pred_hand_mask.save(osp.join(self.cfg.out_dir, 
                                            f"midresult/test_hand_obj_cam/{name}_optim_non_global.png"))
            
                
    def run_handpose_global(self):  
        """ Fix object pose, optimize hand pose"""
        # init param
        fullpose:torch.Tensor = self.mano_params['fullpose'].clone()
        betas:torch.Tensor = self.mano_params['betas'].clone()
        
        orient_res = torch.nn.Parameter(torch.zeros([1,3], device=self.device, dtype=fullpose.dtype))
        orient_res.requires_grad_()
        betas.requires_grad_()
        
        params_group = [
            {'params': self.global_params['hand'], 'lr': 5e-2},
            # {'params': betas, 'lr': 1e-4},
            {'params': orient_res, 'lr': 1e-5},
        ]
        outer_iteration = 20
        
        self.optimizer = optim.Adam(params_group)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5) 
        
        num_iterations = self.cfg['iteration']
        use_3d_loss = False
        
        _, init_hand_mask, init_info, iou = self.optim_handpose_global(fullpose, betas)
        
        for outer_iter in range(outer_iteration):    
            
            best_loss = float('inf')
            best_global_params = None
            best_fullpose = None
            for iteration in range(num_iterations):
                self.optimizer.zero_grad()
                fullpose_new = fullpose.clone()
                fullpose_new[:,:3] += orient_res
                loss, pred_hand_mask, _, _ = self.optim_handpose_global(fullpose_new, betas, 
                                                                        use_3d_loss=use_3d_loss)                     
                
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                self.global_step = iteration
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_global_params = self.global_params['hand'].detach().clone()
                    best_fullpose = fullpose.detach().clone()
                    best_fullpose[:,:3] += orient_res
                    
            
            if outer_iter < 2:
                print("run icp")
                fullpose_new = best_fullpose
                self.global_params['hand'] = best_global_params
                hand_mesh, hand_verts, hand_contact, hand_c_normals = self.get_hand_contact(fullpose_new, betas.detach())
                succ = self.optim_contact(hand_mesh, hand_verts, hand_contact, hand_c_normals) # adjust the transl 
                if succ == False:
                    use_3d_loss = True
        
        self.mano_params['fullpose'] = best_fullpose
        self.global_params['hand'] = best_global_params
        self.mano_params['betas'] = betas.detach()
        name = self.data['name']
        
        
        # vis for check, can be commented
        if self.vis_mid_results:
            pred_hand_mask = pred_hand_mask.detach().cpu()
            # mask = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            pred_hand_mask = ToPILImage(pred_hand_mask)
            
            pred_hand_mask.save(osp.join(self.cfg.out_dir, 
                                            f"midresult/test_hand_obj_cam/{name}_optim.png"))
            
            init_hand_mask = init_hand_mask.detach().cpu()
            init_hand_mask = ToPILImage(init_hand_mask)
            init_hand_mask.save(osp.join(self.cfg.out_dir, 
                                            f"midresult/test_hand_obj_cam/{name}_init.png"))
            
            gt_hand_mask = self.data["hamer_hand_mask"].cpu()
            gt_hand_mask = ToPILImage(gt_hand_mask.float())
            gt_hand_mask.save(osp.join(self.cfg.out_dir, 
                                        f"midresult/test_hand_obj_cam/{name}_gt.png"))
                
            
    def get_hand_contact(self, fullpose, betas):
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        hand_verts_handcam = self.get_hand_for_handcam(mano_output.verts, scale=1., transl=torch.zeros(3, device=self.device))
        hand_verts_objcam = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        
        hand_mesh_objcam = trimesh.Trimesh(vertices=hand_verts_objcam.detach().squeeze().cpu().numpy(), 
                                    faces=self.hand_faces.squeeze().cpu().numpy())
        
        # projections = self.data["hand_cam"]["projection"]
        # c2ws = self.data["hand_cam"]["extrinsics"]
        # c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        # hand_depth, hand_rast = self.depth_peel(hand_verts_handcam, self.hand_faces.squeeze().int(), projections, c2ws, self.data["resolution"])
        
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        
        if c2ws.shape[0] == 3:
            c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        hand_depth, hand_rast = self.depth_peel(hand_verts_objcam, self.hand_faces.squeeze().int(), projections, c2ws, self.data["resolution"])
        
        hoi_mask = self.data["hamer_hand_mask"].bool() & self.data["inpaint_mask"]
        front_mask = ( hoi_mask & self.data["inpaint_mask"] & (~self.data["obj_mask"])).int()
        
        normals = hand_mesh_objcam.vertex_normals.copy()
        normals = torch.Tensor(normals).float().to(self.device)
        hand_pts_front, hand_normals_front, contact_mask_front = compute_hand_contact(
                                                    side='hand_front',
                                                    mask = front_mask,
                                                    verts=hand_verts_objcam.squeeze(),
                                                    faces=self.hand_faces.squeeze(),
                                                    normals=normals,
                                                    rast=hand_rast.detach().squeeze(),
                                                    skipped_face_ids=self.hand_backface_ids
                                                )
        
        back_mask = ( hoi_mask & self.data["hamer_hand_mask"].bool() & (~self.data["hand_mask"])).int()
        hand_pts_back, hand_normals_back, contact_mask_back = compute_hand_contact(
                                                    side='hand_back',
                                                    mask = back_mask,
                                                    verts=hand_verts_objcam.squeeze(),
                                                    faces=self.hand_faces.squeeze(),
                                                    normals=normals,
                                                    rast=hand_rast.detach().squeeze(),
                                                    skipped_face_ids=self.hand_backface_ids
                                                )
        
        hand_pts = torch.concat([hand_pts_front, hand_pts_back], dim=0)
        hand_normals = torch.concat([hand_normals_front, hand_normals_back], dim=0)
        contact_mask = (contact_mask_front | contact_mask_back)
        hand_contact = {'front': hand_pts_front, 
                        'back': hand_pts_back,
                        'both': hand_pts}
        hand_contact_normal = {'front': hand_normals_front, 
                            'back': hand_normals_back,
                            'both': hand_normals}
        
        # <---------- For visualization ------------>
        if self.vis_mid_results:
            name = self.data['name']
            contact_mask_img = ToPILImage(front_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.cfg.out_dir, "contact", f"{name}_hand_mask_front.png"))
            
            contact_mask_img = ToPILImage(back_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.cfg.out_dir, "contact", f"{name}_hand_mask_back.png"))
            
            image_utils.save_depth(hand_depth.detach(), 
                                    os.path.join(self.cfg.out_dir, "contact", f"{name}_hand_depth"),
                                    text_list=["layer_0", "layer_1", "layer_2", "layer_3"])
            
            ids = torch.nonzero(contact_mask == 0)
            hand_depth[:, :, ids[:,0], ids[:,1]] = 0
            image_utils.save_depth(hand_depth.detach(), 
                                    os.path.join(self.cfg.out_dir, "contact", f"{name}_filtered_hand_depth"),
                                    text_list=["layer_0", "layer_1", "layer_2", "layer_3"])
        
        return hand_mesh_objcam, hand_verts_objcam, hand_contact, hand_contact_normal
    
    def optim_contact(self, hand_mesh, hand_verts, hand_contact, hand_contact_normal):
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        gt_hand_mask = self.data["hand_mask"].float()
        
        min_error = float('inf')
        best_t = None
        best_side = None
        trans_hand_pts = {}
        for side in ['front', 'back', 'both']:
            if self.obj_contact[side] is None:
                continue
            if len(hand_contact[side]) == 0:
                hand_contact[side] = hand_verts.squeeze()
                hand_contact_normal[side] = hand_mesh.vertex_normals
                hand_contact[side] = hand_contact[side][self.hand_contact_zone]
                hand_contact_normal[side] = hand_contact_normal[side][self.hand_contact_zone]
                hand_contact_normal[side] = torch.Tensor(hand_contact_normal[side]).float().to(self.device)
            R, t, scale, trans_hand_pts[side], error_3d = icp_with_scale(
                src_points=hand_contact[side],
                src_norm=-hand_contact_normal[side], # we hope the hand normal be opposite to object's
                tgt_points=self.obj_contact[side],
                tgt_norm=self.obj_contact_normals[side],
                fix_R=True,
                fix_scale=True,
                device=self.device
            )
            
            after_hand_verts = (scale * hand_verts) @ R.T + t
            # img = self.render_hand_image(hand_verts=after_hand_verts)
            img = self.render_hoi_image(after_hand_verts, self.hand_faces.squeeze(), 
                                obj_verts, self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
            pred_hand_mask = img[...,0]
            error_2d = soft_iou_loss(pred_hand_mask, gt_hand_mask)
            error = error_3d + error_2d
            print(side, 'error: ', error.item(), 'error_2d: ', error_2d.item(), 'error_3d: ', error_3d.item())
            
            if error < min_error and not torch.isnan(t).any():
                best_t = t
                best_side = side
                min_error = error
        
        if best_t is None:
            return False
        
        with torch.no_grad():
            self.global_params['hand'][1:] += best_t
        
        # after_hand_verts = (R @ (scale * hand_verts).T).T + t
        after_hand_verts = (scale * hand_verts) @ R.T + best_t
        
        
        name = self.data['name']
        # output the hand mesh with contact point
        verts = hand_verts.detach().squeeze().cpu().numpy()
        hand_pts_front = pc_to_sphere_mesh(hand_contact['front'].detach().cpu().numpy())
        hand_pts_back = pc_to_sphere_mesh(hand_contact['back'].detach().cpu().numpy())
        
        num_front = hand_pts_front.vertices.shape[0]
        mesh = hand_mesh + hand_pts_front + hand_pts_back
        vertex_colors = np.tile([0.5,0.5,0.5,1], (len(mesh.vertices), 1))  # [R, G, B, A]
        vertex_colors[len(verts):len(verts)+num_front, :] = [1.0, 0.0, 0.0, 1.0]  # Red
        vertex_colors[len(verts)+num_front:, :] = [0.0, 1.0, 0.0, 1.0]  # Green
        
        mesh.visual.vertex_colors = vertex_colors
        mesh.export(os.path.join(self.cfg.out_dir, "contact", f"{name}_hand.ply"))
        
        # output the transformed hand mesh with contact point
        trans_hand_pts = trans_hand_pts[best_side]
        hand_mesh = trimesh.Trimesh(vertices=after_hand_verts.detach().squeeze().cpu().numpy(), 
                                    faces=self.hand_faces.squeeze().cpu().numpy())
        
        trans_hand_pts = pc_to_sphere_mesh(trans_hand_pts.detach().cpu().numpy())
        
        mesh = hand_mesh + trans_hand_pts
        vertex_colors = np.tile([0.5,0.5,0.5,1], (len(mesh.vertices), 1))  # [R, G, B, A]
        if best_side == 'front':
            vertex_colors[len(verts):, :] = [1.0, 0.0, 0.0, 1.0]  # Red
        elif best_side == 'back':
            vertex_colors[len(verts):, :] = [0.0, 1.0, 0.0, 1.0]  # Green
        else:
            vertex_colors[len(verts):, :] = [0.0, 0.0, 1.0, 1.0]  # Blue
        if vertex_colors.size > 0:
            mesh.visual.vertex_colors = vertex_colors
            mesh.export(os.path.join(self.cfg.out_dir, "contact", f"{name}_hand_after.ply"))
            
        return True
        
    
    def hamer_process(self, vertices):
        """
        check third_party/hamer/hamer/utils/renderer.py vertices_to_trimesh method
        """
        vertices = vertices + self.data["cam_transl"]
        rot = torch.tensor([[[1,0,0],
                            [0,-1,0],
                            [0,0,-1]]], dtype=torch.float, requires_grad=False).to(vertices.device)
        vertices = vertices @ rot.mT
        return vertices
    
    def get_cTw(self):
        rot = torch.tensor([[1,0,0],
                            [0,-1,0],
                            [0,0,-1]], dtype=torch.float, requires_grad=False)
        transl = self.data["cam_transl"].cpu().squeeze()
        cTw = torch.eye(4)
        cTw[:3, :3] = rot
        cTw[:3, -1] = rot @ transl
        # cTw[:3, -1] = transl
        
        # cTw = torch.inverse(cTw)
        
        return cTw
        
    
    def get_hand_for_objcam(self, hand_verts, scale, transl):    
        hand_verts = self.get_hand_for_handcam(hand_verts, scale, transl, need_hamer_process=True)
        hand_verts = verts_transfer_cam(hand_verts, self.data["hand_cam"], self.data["obj_cam"])
        return hand_verts
    
    def get_hand_for_handcam(self, hand_verts, scale, transl, need_hamer_process=True):
        hand_verts = hand_verts * scale
        hand_verts = hand_verts + transl
        
        hand_verts[:,:,0] = (2*self.data["is_right"]-1)*hand_verts[:,:,0]
        
        if need_hamer_process is True:
            hand_verts = self.hamer_process(hand_verts)
        
        return hand_verts
    
    def get_hand_global_rot(self):
        hamer_rot = torch.tensor([[[1,0,0],
                            [0,-1,0],
                            [0,0,-1]]], dtype=torch.float, requires_grad=False).to(self.device)
        
        src_cam_ext = self.data["hand_cam"]["extrinsics"]
        tgt_cam_ext = self.data["obj_cam"]["extrinsics"]
        cam_rot = tgt_cam_ext[None, :3, :3] @ src_cam_ext[None, :3,:3].mT
        
        return cam_rot @ hamer_rot
        
    
    def get_hand_verts(self, hand_verts, scale, transl):
        # hand_verts = self.hamer_process(hand_verts)
        # hand_verts = verts_transfer_cam(hand_verts, self.data["hand_cam"], self.data["obj_cam"])
        hand_verts[:,:,0] = (2*self.data["is_right"]-1)*hand_verts[:,:,0]
        
        global_rot = self.get_hand_global_rot()
        hand_verts = hand_verts @ global_rot.mT
        hand_verts = hand_verts * scale
        hand_verts = hand_verts + transl
        
        return hand_verts
        
    
    def transform_obj(self, scale, transl, orient, need_hamer_process=True):
        # rot_mat = axis_angle_to_matrix(orient)
        # obj_verts = scale * self.data["object_verts"]@rot_mat.T + transl
        
        obj_verts = self.data["object_verts"]
        
        return obj_verts
    
    def transform_obj_origin(self, obj_path, scale, transl, orient, need_hamer_process=True):
        obj_mesh = trimesh.load(obj_path)
        obj_verts = torch.from_numpy(obj_mesh.vertices).to(self.device).unsqueeze(0)
        
        rot_mat = axis_angle_to_matrix(orient)
        obj_verts = scale * self.data["object_verts"]@rot_mat.T + transl
        
        if need_hamer_process is True:
            obj_verts = self.hamer_process(obj_verts)
        obj_mesh.vertices = obj_verts.squeeze().detach().cpu().numpy()
        
        return obj_mesh
        
    
    def vis_hand_object(output, data, image, save_dir):
        hHand = output['hHand']
        hObj = output['hObj']
        device = hObj.device

        cam_f, cam_p = data['cam_f'], data['cam_p']
        cTh = data['cTh']

        hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
        hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)
        cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
        cameras = PerspectiveCameras(cam_f, cam_p, device=device)
        iHoi = mesh_utils.render_mesh(cHoi, cameras,)
        image_utils.save_images(iHoi['image'], save_dir + '_cHoi', bg=data['image']/2+0.5, mask=iHoi['mask'])
        image_utils.save_images(data['image']/2+0.5, save_dir + '_inp')

        image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, view_centric=True)
        image_utils.save_gif(image_list, save_dir + '_cHoi')

        mesh_utils.dump_meshes([save_dir + '_hoi'], hHoi)
    
    def export_for_sim(self):
        name = self.data['name']
        
        out_path = osp.join(self.cfg.out_dir, "sim")
        os.makedirs(out_path, exist_ok=True)
        
        mano_output = self.get_mano_output()
        
        hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        
        
        obj_verts = self.transform_obj(**self.get_params_for('obj'), need_hamer_process=False)
        
        res = {"mano":mano_output, "hand": self.get_params_for('hand'), "obj": self.get_params_for('obj') }
        torch.save(res, os.path.join(out_path, f"{name}.pt"))
        
        # export meshes
        hand_verts = hand_verts.squeeze().detach().cpu().numpy()
        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=self.hand_faces.squeeze().cpu())
        obj_mesh = trimesh.Trimesh(vertices=obj_verts.detach().squeeze().cpu(),
                                   faces=self.data["object_faces"].squeeze().cpu(),
                                   vertex_colors=self.data["object_colors"])
        
        
        hand_mesh.export(os.path.join(out_path, f"{name}_hand.ply"))
        obj_mesh.export(os.path.join(out_path, f"{name}_obj.ply"))
        
    def export_for_eval(self, prefix=None):
        
        if prefix is not None:
            os.makedirs(osp.join(self.cfg.out_dir, "eval"+ "_" + prefix), exist_ok=True)
            foldername = "eval"+ "_" +prefix
        else:
            foldername = "eval"
            
        global_mat = self.get_hand_global_rot()
        
        objVerts = self.transform_obj(**self.get_params_for('obj'), need_hamer_process=False)
        
        objVerts = objVerts @ global_mat # global_mat^-1 @ objVerts
        scaled_objVerts = objVerts/self.global_params['hand'][0] # scale
        
        obj_mesh = trimesh.Trimesh(vertices=scaled_objVerts.detach().squeeze().cpu(),
                                   faces=self.data["object_faces"].squeeze().cpu(),
                                   vertex_colors=self.data["object_colors"])
        
        mano_params = self.mano_params
        fullpose = mano_params['fullpose']
        
        transl = self.global_params['hand'][1:] / self.global_params['hand'][0]
        transl = torch.matmul(global_mat.mT , transl)
        
        rot = fullpose[:,:3]
        rot, transl = hand_utils.cvt_axisang_t_i2o(rot, transl)
        
        objRot = torch.zeros_like(transl)
        objTrans = torch.zeros_like(transl)
        
        wTh = geom_utils.axis_angle_t_to_matrix(rot, transl)
        wTo = geom_utils.axis_angle_t_to_matrix(objRot, objTrans)
        hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo
        
        hA = mano_params['fullpose'][:, 3:]
        
        data = {'wTh': wTh,
                'hTo': hTo, 
                'obj_mesh': obj_mesh, 
                'hA': hA,
                }
        
        hand_rot = self.get_hand_global_rot()
        mano_params = {key: mano_params[key].cpu() for key in mano_params}
        cam_data = {
            "name": self.data["name"],
            "img_path": self.data["img_path"],
            "is_right": self.data["is_right"],
            "cam_projection" : self.data["obj_cam"]["projection"].cpu(),
            "cam_extrinsics" : self.data["obj_cam"]["extrinsics"].cpu(),
            "mano_params": mano_params,
            "hand_scale": self.global_params['hand'][0].cpu(),
            "hand_transl": self.global_params['hand'][1:].cpu(),
            "hand_rot": hand_rot.squeeze().cpu()
        }
        torch.save(data, osp.join(self.cfg.out_dir, foldername, f"{self.data['name']}.pkl"))
        torch.save(cam_data, osp.join(self.cfg.out_dir, foldername, f"{self.data['name']}_hand_in_objcam.pkl"))
        
    
    def export_for_retarget(self):
        global_mat = self.get_hand_global_rot()
        
        objVerts = self.transform_obj(**self.get_params_for('obj'), need_hamer_process=False)
        objVerts = objVerts @ global_mat # global_mat^-1 @ objVerts
        objVerts /= self.global_params['hand'][0] # scale
        
        obj_mesh = trimesh.Trimesh(vertices=objVerts.detach().squeeze().cpu(),
                                   faces=self.data["object_faces"].squeeze().cpu(),
                                   vertex_colors=self.data["object_colors"])
        
        mano_params = self.mano_params
        fullpose = mano_params['fullpose']
        
        transl = self.global_params['hand'][1:] / self.global_params['hand'][0]
        transl = torch.matmul(global_mat.mT , transl)
        
        data = {
            'is_right': self.data["is_right"],
            'global_orient': fullpose[:,:3].detach().cpu(),
            'transl': transl.detach().cpu(),
            'fullpose': fullpose[:,3:].detach().cpu(),
            'beta': mano_params['betas'].detach().cpu() 
        }
        for key in data:
            print(key, ": ", data[key].shape)
        torch.save(data, osp.join(self.cfg.out_dir, "retarget", f"{self.data['name']}.pt"))
        obj_mesh.export(osp.join(self.cfg.out_dir, "retarget", f"{self.data['name']}_obj.ply"))
        
        # fullpose = torch.concat([global_orient, fullpose[:, 3:]], dim=1)
        betas = mano_params['betas']
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        hand_verts = mano_output.verts
        hand_verts[:,:,0] = (2*self.data["is_right"]-1)*hand_verts[:,:,0]
        hand_verts = hand_verts + transl
        hand_mesh = trimesh.Trimesh(vertices=hand_verts.squeeze().detach().cpu().numpy(), 
                                    faces=self.hand_faces.squeeze().cpu())
        hand_mesh.export(osp.join(self.cfg.out_dir, "retarget", f"{self.data['name']}_hand.ply"))
    
    def export_for_retarget_new(self):
        global_mat = self.get_hand_global_rot()
        
        objVerts = self.transform_obj(**self.get_params_for('obj'), need_hamer_process=False)
        objVerts = objVerts @ global_mat # global_mat^-1 @ objVerts
        objVerts /= self.global_params['hand'][0] # scale
        objVerts[:,:,0] = (2*self.data["is_right"]-1)*objVerts[:,:,0]
        objFaces = self.data["object_faces"].squeeze().cpu()
        if not self.data["is_right"]:
            objFaces = objFaces[:,[0,2,1]] # faces for left hand
        
        mano_params = self.mano_params
        fullpose = mano_params['fullpose']
        betas = mano_params['betas']
        rotation_center = self.mano_layer.get_rotation_center(betas)
        rotation_center[0] =  (2*self.data["is_right"]-1)*rotation_center[0]
        global_orient = fullpose[:,:3]
        transl = self.global_params['hand'][1:] / self.global_params['hand'][0]
        transl = torch.matmul(global_mat.mT , transl)
        
        objVerts -= transl
        objVerts -= rotation_center
        obj_mesh = trimesh.Trimesh(vertices=objVerts.detach().squeeze().cpu(),
                                   faces=objFaces,
                                   vertex_colors=self.data["object_colors"])
        obj_transform = geom_utils.axis_angle_t_to_matrix(global_orient).squeeze().detach().cpu()
        obj_mesh.apply_transform(np.linalg.inv(obj_transform))
        obj_mesh.apply_translation((rotation_center).squeeze().detach().cpu().numpy())
        obj_mesh.export(osp.join(self.cfg.out_dir, "retarget", f"{self.data['name']}_obj.ply"))
        
        
        fullpose[:, :3] *=0
        transl *= 0
        
        data = {
            'is_right': self.data["is_right"],
            'global_orient': fullpose[:,:3].detach().cpu(),
            'transl': transl.detach().cpu(),
            'fullpose': fullpose[:,3:].detach().cpu(),
            'beta': mano_params['betas'].detach().cpu() 
        }
        for key in data:
            print(key, ": ", data[key].shape)
        torch.save(data, osp.join(self.cfg.out_dir, "retarget", f"{self.data['name']}.pt"))
        
        
        
        # fullpose = torch.concat([global_orient, fullpose[:, 3:]], dim=1)
        betas = mano_params['betas']
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        hand_verts = mano_output.verts
        # hand_verts[:,:,0] = (2*self.data["is_right"]-1)*hand_verts[:,:,0]
        hand_verts = hand_verts + transl
        if not self.data["is_right"]:
            hand_faces = self.hand_faces[:,[0,2,1]] # faces for left hand
        else:
            hand_faces = self.hand_faces
        hand_mesh = trimesh.Trimesh(vertices=hand_verts.squeeze().detach().cpu().numpy(), 
                                    faces=hand_faces.squeeze().cpu())
        hand_mesh.export(osp.join(self.cfg.out_dir, "retarget", f"{self.data['name']}_hand.ply"))
        
        
    
    def export(self, prefix=None):
        mano_output = self.get_mano_output()
        hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        if prefix is not None:
            filename = f"{prefix}_{self.data['name']}"
        else:
            filename = self.data['name']
    
        # hand_faces = torch.tensor(self.hand_faces.astype(np.int32)).to(hand_verts.device)
        
        # export rendered image
        img = self.render_hoi_image(hand_verts, self.hand_faces.squeeze(), 
                                obj_verts, self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
        img = img.detach().cpu().numpy() 
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        image = Image.fromarray(img)
        image.save(osp.join(self.cfg.out_dir, "render", f"{filename}.png"))
        
        
        # export meshes
        hand_verts = hand_verts.squeeze().detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=hand_verts, faces=self.hand_faces.squeeze().cpu())
        obj_mesh = trimesh.Trimesh(vertices=obj_verts.detach().squeeze().cpu(),
                                   faces=self.data["object_faces"].squeeze().cpu(),
                                   vertex_colors=self.data["object_colors"])
        # obj_mesh = self.transform_obj_origin(self.data["mesh_path"], **self.get_params_for('obj'))
        mesh = mesh+obj_mesh
        path = osp.join(self.cfg.out_dir, f"{filename}.ply")
        mesh.export(path)
        
        
    def exam_mask(self, pred:torch.Tensor, gt:torch.Tensor, prefix:str):
        out_dir = osp.join(self.cfg.out_dir, "render_exam")
        os.makedirs(out_dir, exist_ok=True)
        pred = pred.detach().cpu().numpy()
        gt = gt.cpu().numpy()
        
        cmap = plt.get_cmap('viridis')
        pred = (cmap(pred)[:,:,:3] * 255).astype(np.uint8)
        gt = (cmap(gt)[:,:,:3]*255).astype(np.uint8)
        
        pred_img = Image.fromarray(pred).convert('RGB')  #grayscale
        gt_img = Image.fromarray(gt).convert('RGB') 
        pred_img.save(osp.join(out_dir, f"{prefix}_pred.png"))
        gt_img.save(osp.join(out_dir, f"{prefix}_gt.png"))
        
        

