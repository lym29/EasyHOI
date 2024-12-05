import sys
sys.path.append("..")
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import open_clip
from open_clip import tokenizer
import os
import os.path as osp
import subprocess
import trimesh
import json
import pyrender
import argparse
sys.path.append('third_party/hamer')
from hamer.models import MANO
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

class CLIP_classifier:
    def __init__(self, model_dir):
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'convnext_base_w', 
            pretrained=osp.join(model_dir, 'open_clip_pytorch_model.bin')
            )
        self.classes = ['carrot', 'orange', 'book', 'brush', 'bottle', 'kettle', 'pot', 'cup']
        print(len(self.classes))
        self.text_descriptions = [f"A photo of a {label} held in a hand" for label in self.classes]
        # self.text_descriptions = [f"A photo of a {label}." for label in self.classes]
        
        self.text_tokens = tokenizer.tokenize(self.text_descriptions)
        self.model.eval()
        
    def building_feature(self, img_input, text_tokens):
        with torch.no_grad():
            image_features = self.model.encode_image(img_input).float()
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features
    
    def __call__(self, origin_images: list):
        img_tensor = []
        for img in origin_images:
            img_tensor.append(self.preprocess(img))
        img_tensor = torch.tensor(np.stack(img_tensor))
        print(img_tensor.shape)
        image_features, text_features = self.building_feature(img_tensor, self.text_tokens)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
        
        plt.figure(figsize=(16,16))
        
        for i, img in enumerate(origin_images):
            plt.subplot(4, 4, 2*i+1)
            plt.imshow(img)
            plt.axis("off")

            plt.subplot(4, 4, 2*i+2)
            y = np.arange(top_probs.shape[-1])
            plt.grid()
            plt.barh(y, top_probs[i])
            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.yticks(y, [self.classes[index] for index in top_labels[i].numpy()])
            plt.xlabel("probability")

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(fname="./output/clip_result.pdf")

class Segformer:
    def __init__(self, model_dir):
        
        pass

class Inpainter:
    def __init__(self) -> None:
        pass

class HandPoseEstim:
    def __init__(self) -> None:
        pass
    def run(self, img_dir=None, out_dir=None, mask_path=None, save_mesh=True):
        current_dir = os.getcwd()
        if img_dir is None:
            img_dir = osp.join(current_dir, "preprocess/collected_data/input")
        if out_dir is None:
            out_dir = osp.join(current_dir, "preprocess/collected_data/output/hamer")
        if mask_path is None:
            mask_path = osp.join(current_dir, "preprocess/collected_data/hand_mask/")
        cmd = (
            f'cd ./third_party/hamer && '
            f'python demo.py '
            f'--img_folder {img_dir} --out_folder {out_dir} --hand_mask_path {mask_path} '
            f'--batch_size=48 --side_view --full_frame '
        )
        if save_mesh:
            cmd += f'--save_mesh'
        
        print(cmd)
        completed_process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        print(completed_process.stdout)
        if completed_process.stderr:
            print(completed_process.stderr)
    
class ObjectRecon:
    def __init__(self) -> None:
        pass
    def run(self, img_file):
        # EXPORT_VIDEO = "true"
        # EXPORT_MESH = "true"

        INFER_CONFIG = "./configs/infer-b.yaml"
        
        current_dir = os.getcwd()
        MODEL_NAME = osp.join(current_dir, "preprocess/pretrained/openlrm-mix-base-1.1")
        MESH_DUMP = osp.join(current_dir, "preprocess/output/openlrm-inpaint")
        VIDEO_DUMP = MESH_DUMP
        IMAGE_INPUT = osp.join(current_dir, img_file)
        # IMAGE_INPUT = "./assets/sample_input/owl.png"
        
        cmd = (
            f'cd ./third_party/OpenLRM && '
            f'python -m openlrm.launch infer.lrm '
            f'--infer {INFER_CONFIG} '
            f'model_name={MODEL_NAME} '
            f'image_input={IMAGE_INPUT} '
            f'mesh_dump={MESH_DUMP} '
            f'video_dump={VIDEO_DUMP} '
            f'export_video=true '
            f'export_mesh=true'
        )
        
        print(cmd)
        completed_process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        print(completed_process.stdout)
        if completed_process.stderr:
            print(completed_process.stderr)


    @staticmethod
    def crop_obj_img(img, mask, label=1):
        # given an image and a mask, crop the image to the mask
        img[mask!=label] = 255
        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            raise ValueError("The mask is empty.")
        print(coords.min(axis=0))
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        extend_x = (x1 - x0) // 10
        extend_y = (y1 - y0) // 10
        x0, x1 = max(0, x0-extend_x), min(mask.shape[0], x1+extend_x)
        y0, y1 = max(0, y0-extend_y), min(mask.shape[1], y1+extend_y)

        # Crop the image using the mask's bounding box
        cropped_image = img[x0:x1, y0:y1]
        cropped_mask = mask[x0:x1, y0:y1]
        return cropped_image, cropped_mask
    
    @staticmethod
    def get_color_map(N=256):
        """
        Return the color (R, G, B) of each label index.
        """
        
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        return cmap
    
    @staticmethod
    def parse_2Dmask_img(mask_img, N=10):
        """
        mask_img: RGB image, shape = (H, W, 3)
        N: number of labels (including background)

        return: pixel labels, shape = (H, W)
        """

        color_map = ObjectRecon.get_color_map(N=N)

        H, W = mask_img.shape[:2]
        labels = np.zeros((H, W)).astype(np.uint8)

        for i in range(N):
            c = color_map[i]
            valid = (mask_img[..., 0] == c[0]) & (mask_img[..., 1] == c[1]) & (mask_img[..., 2] == c[2])
            labels[valid] = i
        
        return labels
    
def test_classify():
    input_dir = "./img/input"
    
    # List all files in the directory
    all_files = os.listdir(input_dir)

    # Filter for files that end with .png or .jpg
    image_files = [osp.join(input_dir,file) for file in all_files 
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    img_list = []
    for filename in image_files:
        image = Image.open(filename).convert("RGB")
        image = image.rotate(-90, expand=True)
        img_list.append(image)
    classifier = CLIP_classifier(model_dir="./preprocess/pretrained")
    classifier(img_list)
    
def test_openlrm(id):
    input_dir = "./HOI4D/ZY20210800004_H4_C2_N15_S11_s04_T5/"
    image_file = osp.join(input_dir, f"img/{id+1:04d}.png")
    mask_file = osp.join(input_dir,f"shift_mask/{id:05d}.png")
    masked_dir = osp.join(input_dir, "masked_img")
    os.makedirs(masked_dir, exist_ok=True)
    masked_img_file = osp.join(masked_dir, f"{id:05d}.png")
    if os.path.exists(masked_img_file) is False:
        image = np.array(Image.open(image_file))
        mask = np.array(Image.open(mask_file))
        mask = ObjectRecon.parse_2Dmask_img(mask)
        cropped_image, _ = ObjectRecon.crop_obj_img(image, mask, label=1)
        cropped_image = Image.fromarray(cropped_image.astype(np.uint8), 'RGB')
        cropped_image.save(masked_img_file)
    
    obj_recon = ObjectRecon()
    obj_recon.run(masked_img_file)
    
def test_openlrm_self():
    obj_recon = ObjectRecon()
    img_dir = "preprocess/img/inpaint"
    # img_dir = "preprocess/img/masked"
    
    for file in os.listdir(img_dir):
        if not file.endswith((".png", "jpg", "jpeg")):
            continue
        file = osp.join(img_dir, file)
        obj_recon.run(file)
    
def test_hamer_mow():
    hpe = HandPoseEstim()
    # hpe.run()
    hpe.run(img_dir="/storage/group/4dvlab/datasets/mow/images",
            out_dir="/storage/group/4dvlab/datasets/mow/hamer",
            save_mesh=False)
    
def test_hamer_wild():
    hpe = HandPoseEstim()
    # hpe.run()
    hpe.run(img_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/images/",
            out_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/hamer/",
            save_mesh=False)
    
def load_cam(file_path):
    with open(file_path, 'r') as json_file:
        cam = json.load(json_file)

    # Convert the list back to a NumPy array
    cam['extrinsics'] = np.array(cam['extrinsics'])
    cam_intrinsic = np.array([[cam['fx'], 0, cam['cx']],
                              [0, cam['fy'], cam['cy']],
                              [0, 0, 1]])
    return cam, cam_intrinsic

def mesh_transfer_cam(src_mesh, src_cam, tgt_cam):
    src_cam_ext, src_cam_int = src_cam
    tgt_cam_ext, tgt_cam_int = tgt_cam
    new_verts = (src_mesh.vertices - src_cam_ext[:,-1]) @ src_cam_ext[:,:3]
    new_verts = new_verts @ src_cam_int.T
    new_verts = new_verts @np.linalg.inv(tgt_cam_int).T 
    new_verts = new_verts @ tgt_cam_ext[:, :3].T + tgt_cam_ext[:,-1] 
    tgt_mesh = src_mesh.copy()
    tgt_mesh.vertices = new_verts
    return tgt_mesh
    
    
def recon_hoi(cam_type='obj'):
    hoi_dir = "./output/hoi"
    hand_dir = "./output/hamer"
    obj_dir = "./output/openlrm"
    input_img_dir = "./img/input"
    os.makedirs(hoi_dir, exist_ok=True)
    
    for file in os.listdir(input_img_dir):
        if not file.endswith((".png", "jpg", "jpeg")):
            continue
        img_fn = file.split(".")[0]
        input_image = Image.open(osp.join(input_img_dir, file))
        w,h = input_image.width, input_image.height
        
        obj_mesh = trimesh.load(osp.join(obj_dir, f"{img_fn}.ply"))
        obj_cam_file = osp.join(obj_dir, f"{img_fn}_cam.json")
        obj_cam, obj_cam_int = load_cam(obj_cam_file)
        
        hand_mesh = trimesh.load(osp.join(hand_dir, f"{img_fn}_0.obj"))
        hand_cam_file = osp.join(hand_dir, f"{img_fn}_cam.json")
        hand_cam, hand_cam_int = load_cam(hand_cam_file)
        
        if cam_type=='obj':
            hand_mesh = mesh_transfer_cam(src_mesh=hand_mesh,
                                        src_cam=(hand_cam['extrinsics'], hand_cam_int),
                                        tgt_cam=(obj_cam['extrinsics'], obj_cam_int))
            
            export_mesh = hand_mesh + obj_mesh
            export_mesh.export(osp.join(hoi_dir, f"{img_fn}_objcam.ply"))
            obj_cam_view_img = render_mesh(mesh=export_mesh,
                                        cam=obj_cam,
                                        img_fn=img_fn,
                                        render_res=(h, w))
            export_img = Image.fromarray(obj_cam_view_img)
            
            export_img.save(osp.join(hoi_dir, f"{img_fn}_objcam.png"))
        if cam_type == 'hand':
            obj_mesh = mesh_transfer_cam(src_mesh=obj_mesh,
                                         src_cam=(obj_cam['extrinsics'], obj_cam_int),
                                         tgt_cam=(hand_cam['extrinsics'], hand_cam_int))
            export_mesh = hand_mesh + obj_mesh
            export_mesh.export(osp.join(hoi_dir, f"{img_fn}_handcam.ply"))
            hand_cam_view_img = render_mesh(mesh=export_mesh,
                                        cam=hand_cam,
                                        img_fn=img_fn,
                                        render_res=(h, w)
                                        )
            export_img = Image.fromarray(hand_cam_view_img)
            
            export_img.save(osp.join(hoi_dir, f"{img_fn}_handcam.png"))

def render_mesh(mesh:trimesh.Trimesh, 
                cam, 
                img_fn,
                render_res,
                scene_bg_color=(0,0,0)):
    
    print(f"{img_fn}",render_res)
        
    renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                          viewport_height=render_res[1],
                                          point_size=1.0)
        
    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
    
    pymesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pymesh, f"{img_fn}")
    
    fx, fy = cam['fx'] * render_res[0], cam['fy'] * render_res[1]
    cx, cy = cam['cx'] * render_res[0], cam['cy'] * render_res[1]
    
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    
    camera_pose = np.eye(4)
    camera_pose[:3, :] = cam['extrinsics']
    
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    
    # Define a spotlight
    spot_light = pyrender.SpotLight(color=np.array([1.0, 1.0, 1.0]),
                                    intensity=3.0,
                                    innerConeAngle=np.pi/16,
                                    outerConeAngle=np.pi/6)

    # Add the light to the scene at the pose
    light_pose = np.eye(4)
    scene.add(spot_light, pose=light_pose)
    
    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color[:,:,:3]
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation.")
    parser.add_argument('dataset_type', type=str, help='Type of the dataset to process')
    
    args = parser.parse_args()
        
    dataset_type = args.dataset_type
    print(f"Received dataset type: {dataset_type}")
    
    hpe = HandPoseEstim()
    
    # Example of further usage
    if dataset_type == "mow":
        hpe.run(img_dir="/storage/group/4dvlab/datasets/mow/images/",
                out_dir="/storage/group/4dvlab/datasets/mow/hamer/",
                mask_path="/storage/group/4dvlab/datasets/mow/obj_recon/hand_mask",
                save_mesh=False)
    if dataset_type == "in_the_wild":
        hpe.run(img_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/images/",
                out_dir="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/hamer/",
                mask_path="/storage/group/4dvlab/yumeng/EasyHOI/in_the_wild2/obj_recon/hand_mask/",
                save_mesh=False)
    
    