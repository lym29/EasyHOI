
import os
import glob
import torch
import pyrender
import numpy as np
from PIL import Image
import trimesh
import manotorch
from tqdm import tqdm
import subprocess
from manotorch.manolayer import ManoLayer, MANOOutput

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import mesh_utils, cam_utils

from src.utils.renderer import make_translation, make_rotation

os.environ['PYOPENGL_PLATFORM'] = 'egl'

colors = {
    # "light_blue": [183,216,254, 255],
    "light_blue": [162,183,254, 255],
    
    "blue": [0,0,255, 255]
}

def read_pkl_files(folder):
    file_pattern = os.path.join(folder, "*_hand_in_objcam.pkl")
    files = glob.glob(file_pattern)
    
    data_list = []
    for file_path in files:
        file_data = torch.load(file_path)
        if "name" not in file_data:
            file_data["name"] = os.path.basename(file_path).split("_")[0]
            print(file_data["name"])
        data_list.append(file_data)
        
    return data_list

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def add_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
    # from phalp.visualize.py_renderer import get_light_poses
    light_poses = get_light_poses()
    light_poses.append(np.eye(4))
    cam_pose = scene.get_pose(cam_node)
    for i, pose in enumerate(light_poses):
        matrix = cam_pose @ pose
        node = pyrender.Node(
            name=f"light-{i:02d}",
            light=pyrender.DirectionalLight(color=color, intensity=intensity),
            matrix=matrix,
        )
        if scene.has_node(node):
            continue
        scene.add_node(node)

def add_point_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
    # from phalp.visualize.py_renderer import get_light_poses
    light_poses = get_light_poses(dist=0.5)
    light_poses.append(np.eye(4))
    cam_pose = scene.get_pose(cam_node)
    for i, pose in enumerate(light_poses):
        matrix = cam_pose @ pose
        # node = pyrender.Node(
        #     name=f"light-{i:02d}",
        #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
        #     matrix=matrix,
        # )
        node = pyrender.Node(
            name=f"plight-{i:02d}",
            light=pyrender.PointLight(color=color, intensity=intensity),
            matrix=matrix,
        )
        if scene.has_node(node):
            continue
        scene.add_node(node)


def render_hoi(renderer, obj_mesh, hand_mesh, camera, camera_pose, img_size=(1000,1000), bg_color=None):
    obj_mesh = pyrender.Mesh.from_trimesh(obj_mesh)
    
    color = colors["light_blue"]
    hand_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(color[0]/255, color[1]/255, color[2]/255, 1.0),  # Normalize RGB
        metallicFactor=0.0,  # Non-metallic
        roughnessFactor=0.8  # High roughness for less reflection
    )
    hand_mesh = pyrender.Mesh.from_trimesh(hand_mesh, material=hand_material)
    
    if bg_color is None:
        scene = pyrender.Scene( bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    else:
        scene = pyrender.Scene( bg_color=bg_color, ambient_light=(0.3, 0.3, 0.3))
        
    scene.add(obj_mesh)
    scene.add(hand_mesh)
    scene.add(camera, pose=camera_pose)
    
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    add_point_lighting(scene, camera_node)
    add_lighting(scene, camera_node)
    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    # scene.add(light, pose=camera_pose)
    
    
    img, _ = renderer.render(scene)
    # img = np.flip(img, dims=[0]) 
    # img = np.flipud(img) # Flip vertically.
    
    return img

def render_rotating_hoi(output_gif_path,
                        renderer, 
                        obj_mesh, hand_mesh, 
                        camera, camera_pose, 
                        bg_color=None, num_frames=36):
    obj_mesh = pyrender.Mesh.from_trimesh(obj_mesh)
    
    color = colors["light_blue"]
    hand_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(color[0]/255, color[1]/255, color[2]/255, 1.0),  # Normalize RGB
        metallicFactor=0.0,  # Non-metallic
        roughnessFactor=0.8  # High roughness for less reflection
    )
    hand_mesh = pyrender.Mesh.from_trimesh(hand_mesh, material=hand_material)
    
    if bg_color is None:
        scene = pyrender.Scene( bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    else:
        scene = pyrender.Scene( bg_color=bg_color, ambient_light=(0.3, 0.3, 0.3))
        
    obj_node = scene.add(obj_mesh)
    hand_node = scene.add(hand_mesh)
    # scene.add(camera, pose=camera_pose)
    
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    add_point_lighting(scene, camera_node)
    add_lighting(scene, camera_node)

    # Generate frames
    frames = []
    for i in range(num_frames):
        angle = (2 * np.pi * i) / num_frames
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        scene.set_pose(obj_node, pose=rotation)
        scene.set_pose(hand_node, pose=rotation)
        color, _ = renderer.render(scene)
        frames.append(Image.fromarray(color))

    # Save frames as a GIF
    frames[0].save(output_gif_path, save_all=True, optimize=True,
                   append_images=frames[1:], duration=100, loop=0)


def blend_image(front_img, bg_img, alpha=1):
    mask = np.any(front_img > 0, axis=-1)
    img = np.array(bg_img)
    img[mask] = front_img[mask] * alpha + img[mask] * (1-alpha)
    return img

def get_hand_mesh(mano_layer, data):
    mano_params = data['mano_params']
    hand_scale = data['hand_scale'].detach()
    hand_transl = data['hand_transl'].detach()
    hand_transl[0] -= 0.15
    # hand_transl[1] -= 0.02
    # hand_transl[2] -= 0.05
    # hand_scale -= 0.1
    global_rot = data['hand_rot'].detach()
    hand_faces = mano_layer.get_mano_closed_faces()
    if not data["is_right"]:
        hand_faces = hand_faces[:,[0,2,1]] # faces for left hand
    
    fullpose = mano_params['fullpose']
    betas = mano_params['betas']
    mano_output: MANOOutput = mano_layer(fullpose, betas)
    hand_verts = mano_output.verts
    hand_verts[:,:,0] = (2*data["is_right"]-1)*hand_verts[:,:,0]
    hand_verts = hand_verts.squeeze()
    
    hand_verts = hand_verts @ global_rot.mT
    hand_verts = hand_verts * hand_scale
    hand_verts = hand_verts + hand_transl
    
    mesh = trimesh.Trimesh(hand_verts.numpy(), hand_faces.numpy())
    mesh.visual.vertex_colors = colors["light_blue"]
      
    return mesh

    
def render_single_data(data, data_dir, out_dir, is_tripo=False):
    name = data['name']
    out_path = os.path.join(out_dir, f"{name}_mesh.gif")
    
    print(out_path)
    
    # if os.path.exists(out_path):
    #     return
        
    img_dir = os.path.join(data_dir, "images")
    if is_tripo:
        mesh_dir = os.path.join(data_dir, "obj_recon/results/tripo/meshes/")
    else:
        mesh_dir = os.path.join(data_dir, "obj_recon/results/instantmesh/instant-mesh-large/meshes/")
    proj = data['cam_projection']
    
    obj_mesh = trimesh.load(os.path.join(mesh_dir, str(name), "full.obj"))
    obj_verts = obj_mesh.vertices
    obj_faces = obj_mesh.faces
    if is_tripo:
        rot1 = np.array([[1,0,0],
                            [0,0,-1],
                            [0,1,0]])
        
        obj_verts = obj_verts @ rot1.T
    else:
        rot1 = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,-1]])
        
        obj_verts = obj_verts @ rot1.T
        obj_faces = np.array(obj_faces[:, ::-1])
        
    if need_vertex_color == True:
        obj_mesh.vertices = obj_verts
        obj_mesh.faces = obj_faces
    else:
        obj_mesh = trimesh.Trimesh(obj_verts, obj_faces)
    
    hand_mesh = get_hand_mesh(mano_layer, data)

    if os.path.exists(os.path.join(img_dir, f"{name}.jpg")):
        bg_img = Image.open(os.path.join(img_dir, f"{name}.jpg")).convert("RGB")
    else:
        bg_img = Image.open(os.path.join(img_dir, f"{name}.png")).convert("RGB")
        
    bg_img, resolution = cam_utils.resize_frame(bg_img)
    W, H = bg_img.size
    print(W, H)
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    
    fx = proj[0,0]*W / 2.0
    fy = proj[1,1]*H / 2.0
    cx = 0.5*W
    cy = 0.5*H
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    
    t_x, t_y, t_z = proj[0, 3], proj[1, 3], proj[2, 3]
    transl = torch.tensor([
        [1, 0, 0, t_x],
        [0, 1, 0, t_y],
        [0, 0, 1, t_z],
        [0, 0, 0, 1]
    ])
    
    cam_pose = data['cam_extrinsics'][:3] # [4,4] or [3,4] -> [3,4]
    cam_pose = torch.concat([cam_pose, torch.tensor([[0,0,0,1]])], dim=0)
    cam_pose = cam_pose @ torch.inverse(transl)
    # cam_pose = cam_pose @ transl
    
    front_img = render_hoi(renderer, obj_mesh, hand_mesh, camera, cam_pose, (W, H))
    
    img = blend_image(front_img, bg_img)
    Image.fromarray(img).save(os.path.join(out_dir, f"{name}.png"))
    
    mesh_img = render_hoi(renderer, obj_mesh, hand_mesh, camera, cam_pose, (W, H), bg_color=(255,255,255,255))
    Image.fromarray(mesh_img).save(os.path.join(out_dir, f"{name}_mesh.png"))
    
    render_rotating_hoi(output_gif_path=out_path,
                        renderer=renderer, 
                        obj_mesh=obj_mesh, hand_mesh=hand_mesh,
                        camera=camera, camera_pose=cam_pose, 
                        bg_color=(255,255,255,255))
    
    

if __name__ == "__main__":
    dataset_type = "realdex"
    need_vertex_color = False
    is_tripo = dataset_type.endswith("tripo")
    
    if dataset_type == "arctic":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/arctic/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/ARCTIC_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/arctic/vis_new/"
    if dataset_type == "in_the_wild":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/in_the_wild/eval_final"
        data_dir = "/storage/group/4dvlab/yumeng/InTheWild_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/in_the_wild/vis/final"
    if dataset_type == "in_the_wild_tripo":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/in_the_wild_tripo/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/InTheWild_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/in_the_wild_tripo/vis_new/"
    if dataset_type == "oakink":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/oakink/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/oakink/vis_new/"
    if dataset_type == "oakink_tripo":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/oakink_tripo/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/oakink_tripo/vis_new/"
    if dataset_type == "mow":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/mow/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/MOW_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/mow/vis/"
    if dataset_type == "dexycb":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/dexycb/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/DexYCB_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/dexycb/vis/"
    if dataset_type == "realdex":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/realdex/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/RealDex_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/realdex/vis/"
    if dataset_type == "realdex_tripo":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/realdex_tripo/eval_init/"
        data_dir = "/storage/group/4dvlab/yumeng/RealDex_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/realdex_tripo/vis/init"
    if dataset_type == "teaser":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/teaser/eval_final/"
        data_dir = "/storage/group/4dvlab/yumeng/Teaser_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/teaser/vis/final"
    if dataset_type == "teaser_tripo":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/teaser_tripo/eval_final/"
        data_dir = "/storage/group/4dvlab/yumeng/Teaser_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/teaser_tripo/vis/final"
        
    print("start")
    os.makedirs(out_dir, exist_ok=True)
    
    data_list = read_pkl_files(result_dir)
    
    print(len(data_list))
    
    mano_layer = ManoLayer(side="right")
    mano_layer.eval()
    
    for data in tqdm(data_list):
        if data["name"] != "69":
            continue
        if "is_right" not in data:
            data["is_right"] = True
        
        render_single_data(data, data_dir, out_dir, is_tripo)
        # try:
        #     render_single_data(data, data_dir, out_dir)
        # except ValueError:
        #     print("hand_mesh wrong")
        # except KeyError:
        #     pass
            