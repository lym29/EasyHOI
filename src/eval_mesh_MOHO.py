import os
import argparse
import json
import pickle
from pathlib import Path
from pyhocon import ConfigFactory

import logging
from tqdm import tqdm
import matplotlib.pyplot as plt 

import numpy as np 
import trimesh
from scipy.spatial import cKDTree as KDTree
import open3d as o3d


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

            if error < stop_error:
                break

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


def scatter_pred_gt(pred_pc, gt_pc):
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))
    ax1.scatter3D(pred_pc[:, 0], pred_pc[:, 1], pred_pc[:, 2])
    ax1.set_title('pred')
    ax2.scatter3D(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2])
    ax2.set_title('gt')
    # ax1.view_init(elev=-20, azim=-130)
    # ax2.view_init(elev=-20, azim=-130)
    plt.savefig('eval_ho3d_mesh_vis.png')
    plt.close()


def eval_mesh(pred_obj_mesh_path, gt_obj_mesh_path, use_icp=True, tmp_vis=True):
    pred_obj_mesh = trimesh.load(pred_obj_mesh_path, process=False, force='mesh')
    if gt_obj_mesh_path[-4: ] == '.ply':
        gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)
    else:   # .obj
        gt_obj_mesh = trimesh.load(gt_obj_mesh_path, force='mesh')

    # registration
    if use_icp:
        icp_solver = icp_ts(pred_obj_mesh, gt_obj_mesh)
        icp_solver.sample_mesh(30000, 'both')
        icp_solver.run_icp_f(max_iter = 100)
        pred_obj_mesh = icp_solver.export_source_mesh()
    
    # sample and rescale
    pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
    gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)
    pred_obj_points *= 100.
    gt_obj_points *= 100.
    
    # tmp vis
    if tmp_vis: 
        scatter_pred_gt(pred_obj_points, gt_obj_points)

    # one direction
    gen_points_kd_tree = KDTree(pred_obj_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = KDTree(gt_obj_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer

    threshold = 0.5 # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0 # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    
    return fscore_obj_5, fscore_obj_10, chamfer_obj


def get_pred_mesh_path(exp_dir):
    mesh_dir = os.path.join(exp_dir, 'meshes_test')
    model_list_raw = os.listdir(mesh_dir)
    scene_list = []
    model_list = []
    for model_name in model_list_raw:
        if model_name[-3:] == 'ply':
            model_list.append(os.path.join(mesh_dir, model_name))
            scene_list.append(model_name.split('_')[0])
    return model_list, scene_list


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


if __name__ == '__main__':
    '''
    default is_continue, use the last ply
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--case', type=str, default='')
    # suggest settng here
    parser.add_argument('--shape_path', type=str, default='')

    scene_gt_dict = {'ABF10': '021_bleach_cleanser', 'BB10': '011_banana', 'GPMF10': '010_potted_meat_can', 'GSF10': '037_scissors', 'MC1': '003_cracker_box', 'MDF10': '035_power_drill', 'SM1': '006_mustard_bottle', 'SMu1': '025_mug', 'SS1': '004_sugar_box'}

    args = parser.parse_args()

    # load conf
    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', args.case)
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    # logger
    logger_path = os.path.join(conf['general.base_exp_dir'], 'logs', 'eval_mesh.log')
    logger = create_logger('logger', logger_path)
    logger.info('Load files')
    logger.info('Config path: %s' % args.conf)
    logger.info('Shapenet path: %s' % args.shape_path)
    logger.info('Config: %s' % json.dumps(conf))

    logger.info('Start validation')

    # get mesh path
    data_type = conf['dataset'].get('name', None)
    if data_type == 'ho3d':
        # pred mesh
        pred_mesh_path_list, scene_list = get_pred_mesh_path(conf['general.base_exp_dir'])
        # gt mesh
        gt_mesh_path_list = [os.path.join(args.shape_path, scene_gt_dict[scene], 'textured_simple.obj') for scene in scene_list]
        # log
        logger.info('Pred mesh path length: %s' % len(pred_mesh_path_list))
        logger.info('GT mesh path length: %s' % len(gt_mesh_path_list))
    else:
        raise ValueError('Do not support!')

    category_sta = {'021_bleach_cleanser': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}, '011_banana': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}, '010_potted_meat_can': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}, '037_scissors':{'f_5_list':[], 'f_10_list': [], 'cd_list':[]},'003_cracker_box': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}, '035_power_drill': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}, '006_mustard_bottle': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}, '025_mug': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]},'004_sugar_box': {'f_5_list':[], 'f_10_list': [], 'cd_list':[]}}

    # eval
    f_5_list = []
    f_10_list = []
    cd_list = []
    for pred_path, gt_path in tqdm(zip(pred_mesh_path_list, gt_mesh_path_list), 
                                   total=len(pred_mesh_path_list)):
        try:
            f_5, f_10, cd = eval_mesh(pred_path, gt_path)
        except:
            continue
        f_5_list.append(f_5)
        f_10_list.append(f_10)
        cd_list.append(cd / 10.)
        for key in category_sta.keys():
            if key in gt_path:
                category_sta[key]['f_5_list'].append(f_5)
                category_sta[key]['f_10_list'].append(f_10)
                category_sta[key]['cd_list'].append(cd / 10.)
                print(key, ' ', np.mean(category_sta[key]['f_5_list']), ' ', np.mean(category_sta[key]['f_10_list']), ' ', np.mean(category_sta[key]['cd_list']))
                break
    for key in category_sta.keys():
        logger.info(key + ': F5:     %f' % np.mean(category_sta[key]['f_5_list']))
        logger.info(key + ': F10:    %f' % np.mean(category_sta[key]['f_10_list']))
        logger.info(key + ': mean:      %f' % np.mean(category_sta[key]['cd_list']))
        logger.info(key + ': median:    %f' % np.median(category_sta[key]['cd_list']))
    # log result
    logger.info('F-score obj @ 5mm:     %f' % np.mean(f_5_list))
    logger.info('F-score obj @ 10mm:    %f' % np.mean(f_10_list))
    logger.info('Mean obj chamfer:      %f' % np.mean(cd_list))
    logger.info('Median obj chamfer:    %f' % np.median(cd_list))
    