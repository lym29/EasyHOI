
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
    
    global_rot = global_rot[:3, :]
    hand_verts = hand_verts @ global_rot.mT
    hand_verts = hand_verts * hand_scale
    
    hand_verts = hand_verts + hand_transl
    
    mesh = trimesh.Trimesh(hand_verts.numpy(), hand_faces.numpy())
    mesh.visual.vertex_colors = colors["light_blue"]
      
    return mesh

def find_file_by_id(directory, id):
    for filename in os.listdir(directory):
        if filename.startswith(id):
            return os.path.join(directory, filename)
    
    return None
    
def render_single_data(data, mesh_dir, out_dir):
    name = data['name']
    out_path = os.path.join(out_dir, f"{name}_mesh.gif")
    
    proj = data['cam_projection']
    print(proj)
    
    obj_path = os.path.join(mesh_dir, f"{name}.ply")
    obj_mesh = trimesh.load(obj_path)
    
    hand_mesh = get_hand_mesh(mano_layer, data)
    img_path = data["img_path"]
    if os.path.exists(img_path):
        bg_img = Image.open(img_path).convert("RGB")
    else:
        img_path = img_path.replace(".png", ".jpg")
        bg_img = Image.open(img_path).convert("RGB")
        
    bg_img, resolution = cam_utils.resize_frame(bg_img)
    W, H = bg_img.size
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    
    fx = proj[0,0]/2 #* W / 2.0
    fy = proj[1,1]/2 #* H / 2.0
    cx = 0.5*W
    cy = 0.5*H
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    
    t_x, t_y, t_z = proj[0, 3], proj[1, 3], proj[2, 3] - 2
    transl = torch.tensor([
        [1, 0, 0, t_x],
        [0, 1, 0, t_y],
        [0, 0, 1, t_z],
        [0, 0, 0, 1]
    ])
    
    cam_pose = data['cam_extrinsics']
    print(cam_pose)
    # cam_pose = torch.inverse(data['cam_extrinsics'])
    
    # cam_pose = cam_pose @ torch.inverse(transl)
    
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
    dataset_type = "arctic"
    
    if dataset_type == "arctic":
        result_dir = "/inspurfs/group/mayuexin/yangzemin/code/ToG/MOHO/exp_dp_ho3d/ho3d_arctic2/eval"
        mesh_dir = "/inspurfs/group/mayuexin/yangzemin/code/ToG/MOHO/exp_dp_ho3d/ho3d_arctic2/meshes_test"
        out_dir = "/storage/data/v-liuym/MOHO_results/arctic_2/vis/"
    if dataset_type == "oakink":
        result_dir = "/storage/data/v-liuym/EasyHOI_results/oakink/eval/"
        data_dir = "/storage/group/4dvlab/yumeng/OakInk_easyhoi/"
        out_dir = "/storage/data/v-liuym/EasyHOI_results/oakink/vis/"
    
    os.makedirs(out_dir, exist_ok=True)
    
    data_list = read_pkl_files(result_dir)
    
    mano_layer = ManoLayer(side="right")
    mano_layer.eval()
    
    for data in tqdm(data_list):
        if "is_right" not in data:
            data["is_right"] = True
        try:
            render_single_data(data, mesh_dir, out_dir)
        # except ValueError:
        #     print("hand_mesh wrong")
        except KeyError:
            pass
            