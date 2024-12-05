import argparse
import os
import os.path as osp
from demo import demo_utils

import torch
import numpy as np
from PIL import Image

import sys
from nnutils.hand_utils import ManopthWrapper
sys.path.append('externals/frankmocap/')
sys.path.append('externals/frankmocap/detectors/body_pose_estimator/')

from renderer.screen_free_visualizer import Visualizer

from nnutils.handmocap import get_handmocap_predictor, process_mocap_predictions, get_handmocap_detector
from nnutils.hoiapi import get_hoi_predictor, vis_hand_object
from nnutils.mesh_utils import to_trimesh, dump_meshes, fscore, pc_to_cubic_meshes
# from nnutils.my_pytorch3d import Meshes
from pytorch3d.structures import Pointclouds
from nnutils import box2mask
import pandas as pd
from nnutils import mesh_utils, geom_utils, hand_utils
import pytorch3d.ops as op_3d
import re
import trimesh
from tqdm import tqdm
import json


def get_args():
    parser = argparse.ArgumentParser(description="Optimize object meshes w.r.t. human.")
    parser.add_argument(
        "--filename", default="demo/test.jpg", help="Path to image."
    )
    parser.add_argument(
        "--data_type", default="mow", help="Path to image."
    )
    parser.add_argument("--out", default="output/mow", help="Dir to save output.")
    parser.add_argument("--view", default="ego_centric", help="Dir to save output.")

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='weights/mow'
    )
    parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

def sample_unit_cube(hObj, num_points, r=1):
    """
    Args:
        points (P, 4): Description
        num_points ( ): Description
        r (int, optional): Description
    
    Returns:
        sampled points: (num_points, 4)
    """
    D = hObj.size(-1)
    points = hObj[..., :3]
    prob = (torch.sum((torch.abs(points) < r), dim=-1) == 3).float()
    if prob.sum() == 0:
        prob = prob + 1
        print('oops')
    inds = torch.multinomial(prob, num_points, replacement=True).unsqueeze(-1)  # (P, 1)

    handle = torch.gather(hObj, 0, inds.repeat(1, D))
    return handle

def normalize_mesh_unit_cube(mesh:trimesh.Trimesh):
    # normalize into a unit cube centered zero, define in the MOW official code
    verts = mesh.vertices
    print(np.min(verts, axis=0))
    verts -= np.min(verts, axis=0)
    verts /= np.abs(verts).max()
    verts *= 2
    verts -= np.max(verts, axis=0) / 2
    mesh.vertices = verts
    return mesh

def BoundingCubeNormalization(mesh:trimesh.Trimesh):
    # defined in DeepSDF https://github.com/facebookresearch/DeepSDF/blob/48c19b8d49ed5293da4edd7da8c3941444bc5cd7/src/Utils.cpp#L170
    # Compute the bounding box
    bounding_box = mesh.bounds
    min_verts = bounding_box[0]
    max_verts = bounding_box[1]

    # Compute the center and scale
    center = (min_verts + max_verts) / 2.0
    scale = (max_verts - min_verts).max()

    # Normalize the vertices
    # normalized_vertices = (mesh.vertices - center) #/ scale
    
    normalized_vertices = mesh.vertices
    normalized_vertices *= 0.001 # convert mm to m

    # Create a new normalized mesh
    normalized_mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=mesh.faces)

    return normalized_mesh

def get_arctic_gt(sid_seq_name, frame, data_dir = "./data/arctic"):
    sid, seq_name = sid_seq_name.split('/')
    obj_name = seq_name.split('_')[0]
    obj_mesh_dir = os.path.join(data_dir, "meta/object_vtemplates", obj_name)
    
    """ Load hand params """
    mano_p = os.path.join(data_dir, 'raw_seqs', sid, f'{seq_name}.mano.npy')
    hand_params = np.load(mano_p,allow_pickle=True).item()
    rh_data = hand_params["right"]
    trans = torch.FloatTensor(rh_data["trans"][frame].reshape(3))[None]
    hA = torch.FloatTensor(rh_data["pose"][frame])[None]
    rot = torch.FloatTensor(rh_data["rot"][frame])[None]
    rot, trans = hand_utils.cvt_axisang_t_i2o(rot, trans)
    wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)
    
    """ Load object params """
    obj_p = os.path.join(data_dir, 'raw_seqs', sid, f'{seq_name}.object.npy')
    obj_params = torch.FloatTensor(np.load(obj_p, allow_pickle=True))
    obj_arti = obj_params[frame, 0]  # radian
    objRot = obj_params[frame, 1:4]
    objTrans = obj_params[frame, 4:] * 0.001
    wTo = geom_utils.axis_angle_t_to_matrix(
        torch.FloatTensor(objRot).reshape(1,3), 
        torch.FloatTensor(objTrans).reshape(1,3)
        )
    hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo
    
    RADIUS = 0.2
    hand_wrapper = ManopthWrapper().to('cpu')
    nTh = hand_utils.get_nTh(hand_wrapper, hA, RADIUS)
    
    """ Load object mesh and sample points on it """
    NUM_POINTS = 12000
    
    # load obj meshes, use obj_arti to get the full mesh
    # check https://github.com/zc-alexfan/arctic/blob/4d41e5da435a355cb496f3a6a4a74bb3e26e9a98/common/object_tensors.py#L73
    mesh_top = trimesh.load(os.path.join(obj_mesh_dir, "top.obj"))
    mesh_bottom = trimesh.load(os.path.join(obj_mesh_dir, "bottom.obj"))
    z_axis = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3)
    
    tf_mat = geom_utils.axis_angle_t_to_matrix(z_axis * obj_arti, torch.zeros_like(z_axis))
    mesh_top.apply_transform(tf_mat[0].numpy())
    
    mesh = BoundingCubeNormalization(mesh_top + mesh_bottom)
    mesh = mesh_utils.load_mesh_from_np(mesh.vertices, mesh.faces)

    # xyz, color = op_3d.sample_points_from_meshes(mesh, NUM_POINTS * 2, return_textures=True)
    
    # xyz = mesh_utils.apply_transform(xyz, (nTh @ hTo))
    # nObj = torch.cat([xyz, color], dim=-1)[0]  # (1, P, 6)
    # nObj = sample_unit_cube(nObj, NUM_POINTS) # nObj is the gt_pc
    # gt_pc = nObj[..., :3]
    
    # # check models/ihoi.py
    # hTn = geom_utils.inverse_rt(mat=nTh, return_mat=True)
    # hTx = hTn
    
    # hGt = mesh_utils.apply_transform(gt_pc, hTx)
    
    # data = {
    #     'xyz': xyz,
    #     'hGt': hGt,
    #     'hA': hA,
    #     'wTh': wTh,
    #     'nTh': nTh,
    #     'hTo': hTo
    # }
    
    return mesh, hTo

def get_hGt(mesh, hTo, nTh):
    
    # in ihoi coord system, the global hand pose wTh is identity
    
    mesh = mesh_utils.apply_transform(mesh, (nTh @ hTo))
    # check models/ihoi.py
    hTn = geom_utils.inverse_rt(mat=nTh, return_mat=True)
    hTx = hTn
    
    hGt = mesh_utils.apply_transform(mesh, hTx)
    
    return hGt

def main(args):
    import json
    import random
    visualizer = Visualizer('pytorch3d')
    
    if args.data_type == "mow":
        with open('/storage/data/jiangqi2022/hoi_data/mow/rhoi_split.json', 'rb') as jf:
            anno = json.load(jf)['test']

        anno = [{'image_id': l} for l in anno]
    elif args.data_type == "oakink":
        data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/images/"
        anno = [osp.join(data_dir,fn) for fn in os.listdir(data_dir) if fn.endswith(".png")]
    elif args.data_type == "arctic":
        csv_file = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/split/test.csv"
        df = pd.read_csv(csv_file)
        data_dict = df.to_dict(orient='list')
        anno = data_dict['img_path']
        
        gt_cache_file = "/storage/group/4dvlab/yumeng/ihoi/cache/arctic_obj_gt.pkl"
        os.makedirs(os.path.dirname(gt_cache_file), exist_ok=True)
        if os.path.exists(gt_cache_file):
            gt_data = torch.load(gt_cache_file)
        else:
            sid_seq_name_list = data_dict['sid_seq_name']
            frame_list = data_dict['frame']
            gt_data = {'mesh': [], 'hTo':[]}
            for (sid_seq_name, frame) in tqdm(zip(sid_seq_name_list,frame_list)):
                mesh, hTo = get_arctic_gt(sid_seq_name, frame)
                gt_data['mesh'].append(mesh)
                gt_data['hTo'].append(hTo)
            
            torch.save(gt_data, gt_cache_file)
        
        
    print(len(anno),"\n\n\n")
    # anno = [{'image_id': "boardgame_v__C60zI5bZ5A_frame000050"}]

    bbox_detector = get_handmocap_detector(args.view)
    hand_predictor = get_handmocap_predictor()
    hand_wrapper = ManopthWrapper().to('cpu')
    hoi_predictor = get_hoi_predictor(args)

    f5, f10, cd = [], [], []
    for img_id, ann in enumerate(anno):
        if args.data_type == "mow":
            args.filename = f"/storage/data/jiangqi2022/hoi_data/mow/images/{ann['image_id']}.jpg"
        elif args.data_type in ["oakink", "arctic"]:
            args.filename = ann
        image = Image.open(args.filename).convert("RGB")
        image = np.array(image)
        print(image.shape)
        
        try:
            # predict hand
            detect_output = bbox_detector.detect_hand_bbox(image[..., ::-1].copy())
            body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
            res_img = visualizer.visualize(image, hand_bbox_list = hand_bbox_list)
            demo_utils.save_image(res_img, osp.join(args.out, f"{osp.basename(args.filename).split('.')[0]}_hand_bbox.jpg"))

            mocap_predictions = hand_predictor.regress(
                image[..., ::-1], hand_bbox_list
            )
            
            print(mocap_predictions)
            
            object_mask = np.ones_like(image[..., 0]) * 255
            # predict hand-held object
            data = process_mocap_predictions(
                mocap_predictions, image, hand_wrapper, mask=object_mask
            )

            output = hoi_predictor.forward_to_mesh(data)
            hObj = output['hObj']
            hHand = output['hHand']
            
            gt_mesh = gt_data['mesh'][img_id].cuda()
            hTo = gt_data['hTo'][img_id].cuda()
            hGt = get_hGt(gt_mesh, hTo, output['nTh'])
            
            output['hGt'] = hGt
            # dump_meshes([osp.join(args.out, "meshes", osp.basename(args.filename).split('.')[0] + '_gt')], hGt)   # check sanity
            # dump_meshes([osp.join(args.out, "meshes", osp.basename(args.filename).split('.')[0] + '_obj')], hObj)   # check sanity
            # dump_meshes([osp.join(args.out, "meshes", osp.basename(args.filename).split('.')[0] + '_hand')], hHand)   # check sanity
            vis_hand_object(output, data, None, 
                            save_dir=osp.join(args.out, "images", osp.basename(args.filename).split('.')[0] ))
            
            th_list = [.5/100, 1/100,]
            f_res = fscore(hObj, hGt, th=th_list)
            # print(f_res)
            f5.append(f_res[0][0])
            f10.append(f_res[1][0])
            cd.append(f_res[2][0])

            # vis_hand_object(output, data, image, args.out + '/%s' % osp.basename(args.filename).split('.')[0])
        except Exception as e:
            print(e)
    
    print(f"f5: {sum(f5)/len(f5)}, f10: {sum(f10)/len(f10)}, cd: {sum(cd)/len(cd)}")
    with open("arctic.log", "w") as file:
        # Write the string to the file
        file.write(f"f5: {sum(f5)/len(f5)}, f10: {sum(f10)/len(f10)}, cd: {sum(cd)/len(cd)}")


if __name__ == "__main__":
    main(get_args())
