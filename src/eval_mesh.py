import os
import os.path as osp
from pathlib import Path
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import igl
import trimesh
from PIL import Image
import re
import pandas as pd
import json
import glob
from src.data import oakink_data, ho3d_data
from manotorch.manolayer import ManoLayer, MANOOutput
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from src.utils import to_np, get_data_from_config

def filter_split_file(data_cfg):
    split_path = os.path.join(data_cfg.base_dir, "split", f"{data_cfg.split}.csv")
    df = pd.read_csv(split_path)
    df['img_id'] = df['img_id'].astype(str)
    
    img_list = os.listdir(os.path.join(data_cfg.input_dir))
    img_list = [f.split('.')[0] for f in img_list]
    
    filtered_df = df[df['img_id'].isin(img_list)]
    filtered_df.to_csv(os.path.join(data_cfg.base_dir, "split", f"{data_cfg.split}.csv"), index=False)
    
    
class icp_ts():
    """
    @description:
    icp solver which only aligns translation and scale
    """
    def __init__(self, mesh_source, mesh_target):
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

        self.points_source = self.mesh_source.vertices.copy()
        self.points_target = self.mesh_target.vertices.copy()

    def sample_mesh(self, n=30000, mesh_id='both'):
        if mesh_id == 'source' or mesh_id == 'both':
            self.points_source, _ = trimesh.sample.sample_surface(self.mesh_source, n)
        if mesh_id == 'target' or mesh_id == 'both':
            self.points_target, _ = trimesh.sample.sample_surface(self.mesh_target, n)

        self.offset_source = self.points_source.mean(0)
        self.scale_source = np.sqrt(((self.points_source - self.offset_source)**2).sum() / len(self.points_source))
        self.offset_target = self.points_target.mean(0)
        self.scale_target = np.sqrt(((self.points_target - self.offset_target)**2).sum() / len(self.points_target))

        self.points_source = (self.points_source - self.offset_source) / self.scale_source * self.scale_target + self.offset_target

    def run_icp_f(self, max_iter = 10, stop_error = 1e-3, stop_improvement = 1e-5, verbose=0):
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float32)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        previous_error = error
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            query_source_points = self.points_source * self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index, :]

            # Find closest source point for each target point:
            query_target_points = (self.points_target - self.trans)/self.scale
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index, :]
            closest_source_points = closest_source_points * self.scale + self.trans
            query_target_points = self.points_target

            # Compute current error:
            error = (((query_source_points - closest_target_points)**2).sum() + ((query_target_points - closest_source_points)**2).sum()) / (query_source_points.shape[0] + query_target_points.shape[0])
            error = error ** 0.5
            if verbose >= 1:
                print(i, "th iter, error: ", error)

            if previous_error - error < stop_improvement:
                break
            else:
                previous_error = error
                
            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1, 1), self.points_source[closest_source_points_index, :].reshape(-1, 1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3), dtype=np.float32) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            b = np.vstack([closest_target_points.reshape(-1, 1), query_target_points.reshape(-1, 1)])
            x = np.linalg.lstsq(A, b, rcond=None)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()
            
    def get_trans_scale(self):
        all_scale = self.scale_target * self.scale / self.scale_source 
        all_trans = self.trans + self.offset_target * self.scale - self.offset_source * self.scale_target * self.scale / self.scale_source
        return all_trans, all_scale

    def export_source_mesh(self):
        self.mesh_source.vertices = (self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        return self.mesh_source

def mppe(prd, gt):
    ''' prd: (NP, 3), gt (NP, 3) '''
    if isinstance(prd, np.ndarray):
        prd = torch.from_numpy(prd).float()
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt).float()

    mppe = torch.mean(torch.norm(prd - gt, p=2, dim=1)).item()
    return mppe

def obj_metric(pred_obj_mesh, gt_obj_mesh, use_icp=True):
    # registration
    if use_icp:
        icp_solver = icp_ts(pred_obj_mesh, gt_obj_mesh)
        icp_solver.sample_mesh(30000, 'both')
        icp_solver.run_icp_f(max_iter = 100)
        pred_obj_mesh = icp_solver.export_source_mesh()
        
    pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
    gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)
    
    pred_obj_points *= 100 # convert to cm
    gt_obj_points *= 100 # convert to cm
    # one direction
    gen_points_kd_tree = KDTree(pred_obj_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = KDTree(gt_obj_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_obj = (gt_to_gen_chamfer + gen_to_gt_chamfer)/10

    threshold = 0.5 # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0 # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    
    return chamfer_obj, fscore_obj_5, fscore_obj_10

def get_hamer_result(cfg, data_cfg):
    get_easyhoi_result(cfg, data_cfg, typename="eval_before_camsetup")
    
def get_easyhoi_result(cfg, data_cfg, data_module, typename = "eval"):
    eval_dir = os.path.join(cfg.out_dir, typename)
    print(eval_dir)
    split_path = os.path.join(data_cfg.base_dir, "split", f"{data_cfg.split}.csv")
    df = pd.read_csv(split_path)
    df['img_id'] = df['img_id'].astype(str)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mano_layer = ManoLayer(side="right").to(device)
    
    for index, row in tqdm(df.iterrows()):
        img_id = data_module.get_img_id(row)
        if not os.path.exists(os.path.join(eval_dir, f"{img_id}.pkl")):
            continue
        data = torch.load(os.path.join(eval_dir, f"{img_id}.pkl"))
        data = {k: v.detach().to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        obj_mesh = data["obj_mesh"]
        hA = data["hA"]
        hTo = data["hTo"]
        
        zeros = torch.zeros(hA.shape[0], 3).to(device)
        hand_pose = torch.cat((zeros, hA), dim=1)
        
        mano_output: MANOOutput = mano_layer(hand_pose.to(device))
        hand_mesh = trimesh.Trimesh(to_np(mano_output.verts.squeeze()), 
                                    mano_layer.get_mano_closed_faces())
        hand_mesh.export(os.path.join(cfg.out_dir, "eval_mesh", f"{img_id}_hand.ply"))
        hand_joints = to_np(mano_output.joints.squeeze())
        np.save(os.path.join(cfg.out_dir, "eval_mesh", f"{img_id}_hand_joints.npy"),
                hand_joints)
        
        obj_mesh.apply_transform(to_np(hTo[0]))
        obj_mesh.export(os.path.join(cfg.out_dir, "eval_mesh", f"{img_id}_obj.ply"))

def evaluate(cfg, data_cfg):
    gt_path = os.path.join(data_cfg.base_dir, "gt_hoi")
    pred_path = os.path.join(cfg.out_dir, "eval_mesh")
    log_path = os.path.join(cfg.log_dir, "eval_result.txt")
    log_path = os.path.join(cfg.log_dir, "eval_result.txt")
    pred_name_list = []
    for f in os.listdir(pred_path):
        if not os.path.exists(os.path.join(gt_path, f)):
            continue
        name = f.split('.')[0]
        if name.endswith("_obj"):
            pred_name_list.append(name.replace("_obj", ""))
        
    pred_name_list = list(set(pred_name_list))
    print(pred_name_list)
    
    # Initialize lists to store results
    chamfer_results = []
    fscore_5_results = []
    fscore_10_results = []
    hand_mpvpe_results = []
    hand_mpjpe_results = []

    # Open the file once before the loop
    with open(log_path, "w") as f:
        # Iterate over the lists and compute metrics
        for img_name in pred_name_list:
            pred_obj = trimesh.load(os.path.join(pred_path, f"{img_name}_obj.ply"))
            gt_obj = trimesh.load(os.path.join(gt_path, f"{img_name}_obj.ply"))
            pred_hand_mesh = trimesh.load(os.path.join(pred_path, f"{img_name}_hand.ply"))
            gt_hand_mesh = trimesh.load(os.path.join(gt_path, f"{img_name}_hand.ply"))
            pred_hand_joints = np.load(os.path.join(pred_path, f"{img_name}_hand_joints.npy"))
            gt_hand_joints = np.load(os.path.join(gt_path, f"{img_name}_hand_joints.npy"))
            
            
            chamfer, fscore_5, fscore_10 = obj_metric(pred_obj, gt_obj)
            hand_mpvpe = mppe(pred_hand_mesh.vertices, gt_hand_mesh.vertices)
            hand_mpjpe = mppe(pred_hand_joints, gt_hand_joints)
            
            if fscore_5 == 0:
                continue
            
            # Append results
            chamfer_results.append(chamfer)
            fscore_5_results.append(fscore_5)
            fscore_10_results.append(fscore_10)
            hand_mpvpe_results.append(hand_mpvpe)
            hand_mpjpe_results.append(hand_mpjpe)
            
            
            # Write each result to the file immediately
            f.write(f"Img:{img_name}, Chamfer: {chamfer}, F-score (5mm): {fscore_5}, F-score (10mm): {fscore_10}\n")
            f.write(f"hand mpvpe:{hand_mpvpe}, hand mpjpe:{hand_mpjpe}\n")
        
        # Compute and write results to the file
        f.write("\nOverall Results:\n")
        f.write(f"Chamfer: Mean: {np.mean(chamfer_results)},Std: {np.std(chamfer_results)}, Median: {np.median(chamfer_results)}\n")
        f.write(f"F-score (5mm): Mean: {np.mean(fscore_5_results)},Std: {np.std(fscore_5_results)}, Median: {np.median(fscore_5_results)}\n")
        f.write(f"F-score (10mm): Mean: {np.mean(fscore_10_results)},Std: {np.std(fscore_10_results)}, Median: {np.median(fscore_10_results)}\n")
        f.write(f"Hand MPVPE: Mean: {np.mean(hand_mpvpe_results)},Std: {np.std(hand_mpvpe_results)}, Median: {np.median(hand_mpvpe_results)}\n")
        f.write(f"Hand MPJPE: Mean: {np.mean(hand_mpjpe_results)},Std: {np.std(hand_mpjpe_results)}, Median: {np.median(hand_mpjpe_results)}\n")
        
    
        

@hydra.main(version_base=None, config_path="./configs", config_name="eval_oakink")
def main(cfg : DictConfig) -> None:
    dataset_name = cfg['data']['name']
    print(dataset_name)
    
    data_cfg = OmegaConf.create(cfg['data'])
    os.makedirs(os.path.join(data_cfg.base_dir, "gt_hoi"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "eval_mesh"), exist_ok=True)
    os.makedirs(os.path.join(cfg.log_dir), exist_ok=True)
    
    
    if dataset_name == "oakink":
        oakink_data.get_gt(data_cfg)
        get_easyhoi_result(cfg, data_cfg, oakink_data, typename="eval_final")
        # get_hamer_result(cfg, data_cfg)
        evaluate(cfg, data_cfg)
    elif dataset_name == "ho3d":
        ho3d_data.get_gt(data_cfg)
        get_easyhoi_result(cfg, data_cfg, ho3d_data, typename="eval_final")
        evaluate(cfg, data_cfg)
        

if __name__ == "__main__":
    main()



