
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
    obj_center = obj_mesh.centroid
    for i in range(num_frames):
        angle = (2 * np.pi * i) / num_frames
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 1, 0], obj_center)
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


if __name__ == "__main__":
    dataset_type = "in_the_wild"
    
    if dataset_type == "arctic":
        data_dir = "/storage/data/v-liuym/EasyHOI_eval/ihoi-arctic-new"
    elif dataset_type == "oakink":
        data_dir = "/storage/group/4dvlab/yumeng/ihoi/oakink_new/"
    elif dataset_type == "in_the_wild":
        data_dir = "/storage/data/v-liuym/EasyHOI_eval/ihoi-inthewild"
    elif dataset_type == "dexycb":
        data_dir = "/storage/data/v-liuym/EasyHOI_eval/ihoi-dexycb"
        
    mesh_dir = os.path.join(data_dir, "cam_meshes")
    cam_dir = os.path.join(data_dir, "camera")
    out_dir = os.path.join(data_dir, "vis")
    
    os.makedirs(out_dir, exist_ok=True)
    
    name_list = []
    for file in os.listdir(cam_dir):
        name_list.append(file.split('.')[0])
        
    W = H = 1000
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    
    # name_list = ["39", "111", "587", "360"]
    for name in tqdm(name_list):
        print(name)
        hand_mesh = trimesh.load(os.path.join(mesh_dir, f"{name}_hand.obj"))
        obj_mesh = trimesh.load(os.path.join(mesh_dir, f"{name}_obj.obj"))
        
        fx = 10*W
        fy = 10*H
        cx = 0.5*W
        cy = 0.5*H
        cam_pose = torch.tensor([
            [1., 0, 0, 0],
            [0, -1., 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1.]
        ])
        
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        
        out_path = os.path.join(out_dir, f"{name}_mesh.gif")
        render_rotating_hoi(output_gif_path=out_path,
                        renderer=renderer, 
                        obj_mesh=obj_mesh, hand_mesh=hand_mesh,
                        camera=camera, camera_pose=cam_pose, 
                        bg_color=(255,255,255,255))
        
            