import numpy as np
import glob
import trimesh
import pyrender
from pyrender import RenderFlags
import open3d as o3d
import os
import cv2
from tqdm import tqdm
import pymeshfix
import argparse
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    z = r * np.sin(phi) * np.sin(theta)
    y = r * np.cos(phi)
    return x, y, z

def look_at(camera_position, target_position, up_vector, r):
    z_axis = camera_position - target_position
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up_vector, z_axis)
    y_axis = np.cross(z_axis, x_axis)

    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    rotation_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], 0],
        [x_axis[1], y_axis[1], z_axis[1], 0],
        [x_axis[2], y_axis[2], z_axis[2], 0],
        [0, 0, 0, 1]
    ])

    cam_translation_matrix = np.array([
        [1, 0, 0, camera_position[0]],
        [0, 1, 0, camera_position[1]],
        [0, 0, 1, camera_position[2]],
        [0, 0, 0, 1]
    ])
    
    translation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -r],
        [0, 0, 0, 1]
    ])
    
    
    
    cam_mat = np.stack([x_axis, y_axis, z_axis, camera_position], axis=1)
    cam_mat = np.vstack([cam_mat, np.array([0,0,0,1])])
    
    view_mat = np.linalg.inv(rotation_matrix) @ translation_matrix
    

    return  cam_mat , view_mat

def save_depth_img(depth, outpath):
    # Normalize the depth image
    depth_normalized = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Apply a colormap
    depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Save the color-mapped depth image
    cv2.imwrite(outpath, depth_colormap)

def generate_depth_images(mesh_path, r=4.5, num_images=10):
    # 加载mesh
    mesh = trimesh.load(mesh_path)
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    w, h = (1000,1000)
    fx = 1000
    fy = 1000
    cx, cy = w/2, h/2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    camera = pyrender.IntrinsicsCamera(fx=1000, fy=1000, cx=cx, cy=cy)
    
    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    # render_flags = RenderFlags.DEPTH_ONLY | RenderFlags.OFFSCREEN

    # 渲染器
    renderer = pyrender.OffscreenRenderer(1000, 1000)

    # 生成深度图和点云
    point_clouds = []
    for i in range(num_images):
        ratio = float(i) /num_images
        theta = 2 * np.pi * ratio # Angle around the y-axis
        phi = np.pi / 2
        

        camera_position = spherical_to_cartesian(r, theta, phi)
        camera_matrix, view_mat = look_at(camera_position, np.array([0, 0, 0]),up_vector=np.array([0, 1, 0]), 
                                          r=r)
        
        # Add camera to scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_matrix)
        scene.add_node(camera_node)
        
        # 渲染深度图像
        depth = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
        # save_depth_img(depth, os.path.join(folder, f"depth_{i}.png"))
        
        # 从深度图像生成点云
        pc = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            intrinsic=intrinsic,
            # extrinsic=camera_matrix,
            depth_scale=1, depth_trunc=10.0,
            project_valid_depth_only=True
        )
        # pc.transform(np.linalg.inv(camera_matrix))
        pc.transform(view_mat)
        pc.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        
        
        
        point_clouds.append(pc)
        scene.remove_node(camera_node)
    
    merged_pc = o3d.geometry.PointCloud()
    for pc in point_clouds:
        merged_pc += pc

    renderer.delete()
    return merged_pc

def poisson_reconstruct(pcd):
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)  # Optionally orient normals
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

def transfer_vertex_colors(mesh_a, mesh_b):
    from scipy.spatial import KDTree  # For efficient nearest neighbor search
    tree = KDTree(mesh_a.vertices)
    _, nearest_indices = tree.query(mesh_b.vertices)
    mesh_a_colors = mesh_a.visual.vertex_colors
    mesh_b_colors = mesh_a_colors[nearest_indices]
    mesh_b.visual.vertex_colors = mesh_b_colors
    return mesh_b

def meshfix(infile, outfile, origin_file=None):
    mesh = trimesh.load(infile)
    MAX_FACE_NUM = 50000
    if mesh.faces.shape[0]>MAX_FACE_NUM:
        simp_mesh = mesh.simplify_quadratic_decimation(MAX_FACE_NUM)
        mesh = transfer_vertex_colors(mesh, simp_mesh)
        mesh.export(infile)
    
    tin = pymeshfix.PyTMesh()
    tin.clean(max_iters=10, inner_loops=3)
    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    print("created")
    try:
        meshfix.repair()
    except:
        print("repair failed")
    
    print("repair finished")
    
    tin.load_array(meshfix.v, meshfix.f)
    tin.fill_small_boundaries()
    tin.clean(max_iters=10, inner_loops=3)
    v, f = tin.return_arrays()
    new_mesh = trimesh.Trimesh(v, f)
    
    if origin_file is not None:
        mesh = trimesh.load(origin_file)
    new_mesh = transfer_vertex_colors(mesh, new_mesh)
    new_mesh.export(outfile)
    print(new_mesh.is_watertight)
    
def load_tripo(folder):
    for model in tqdm(os.listdir(folder)):
        if model.startswith('.'):
            continue
        tripo_dir = os.path.join(folder, model, "tripo")
        obj_files = glob.glob(os.path.join(tripo_dir, '*.obj'))
        # print(obj_files)
        
        if len(obj_files) == 0:
            print(model)
            continue
        mesh = trimesh.load(obj_files[0])
        
        if hasattr(mesh.visual, 'uv'):
            vertex_colors = mesh.visual.to_color().vertex_colors
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors)
            
        mesh.export(os.path.join(folder, model, "full.obj"))

def main(folder, resample=False):
    
    # folder = '/storage/group/4dvlab/yumeng/RealDex_easyhoi/obj_recon/results/instantmesh/instant-mesh-large/meshes/'
    for model in tqdm(os.listdir(folder)):
        if model.startswith('.'):
            continue
        data_dir = os.path.join(folder, model)
        out_path =os.path.join(data_dir, 'recon.ply') 
        pc_path =os.path.join(data_dir, 'recon_pc.ply') 
        fixed_path = os.path.join(data_dir, 'fixed.obj') 
        
        orig_mesh_path = os.path.join(data_dir, "full.obj")
        
        print(orig_mesh_path)
        
        
        if os.path.exists(fixed_path):
            continue
        
        if resample:
            try:
                point_clouds = generate_depth_images(orig_mesh_path)
                mesh = poisson_reconstruct(point_clouds)
                o3d.io.write_triangle_mesh(out_path, mesh)
                o3d.io.write_point_cloud(pc_path, point_clouds)
                meshfix(out_path, fixed_path, orig_mesh_path)
            except:
                print(orig_mesh_path, " MeshFix Wrong")
        else:
            try:
                mesh:trimesh.Trimesh = trimesh.load(os.path.join(data_dir, "full.obj"))
                
                meshfix(orig_mesh_path, fixed_path)
            except:
                print(orig_mesh_path, " MeshFix Wrong")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segementation.")
    parser.add_argument('--data_dir', type=str, help='Path to the data to be processed')
    parser.add_argument('--resample', action='store_true', help='Resample the point cloud using the depth camera')
    
    args = parser.parse_args()
    
    folder = os.path.join(args.data_dir, "obj_recon/results/tripo/meshes/")
    if os.path.exists(folder):
        load_tripo(folder)
        main(folder, False)
    
    folder = os.path.join(args.data_dir, "obj_recon/results/instantmesh/instant-mesh-large/meshes/")
    if args.resample:
        print("resample mesh using depth camera")
        main(folder, True)
    else:
        main(folder, False)
        
    
    