import torch
from manotorch.manolayer import ManoLayer, MANOOutput
import trimesh
import numpy as np
import re
import pickle

def get_template():
    fullpose = torch.zeros([1, 48])
    betas = torch.randn([1, 10])
    mano_layer = ManoLayer()
    mano_output: MANOOutput = mano_layer(fullpose, betas)
    mesh = trimesh.Trimesh(mano_output.verts.squeeze(), mano_layer.get_mano_closed_faces())
    mesh.export("./output/mano_template.ply")

def read_selected_mesh(file_path="./output/mano_template_selected.ply", 
                       out_path ="assets/mano_backface_ids.pkl" ):
    
    selected_face_ids = []
    reading_faces = False
    face_id = 0
    vertices_per_face = 3  # Assuming triangular faces

    with open(file_path, 'r') as f:
        for line in f:
            # Skip lines until we reach "end_header"
            if not reading_faces and line.startswith('end_header'):
                reading_faces = True
                continue  # Move to the next line after finding "end_header"

            if reading_faces:
                face_data = line.split()
                if len(face_data) == vertices_per_face + 2:
                    is_selected = float(face_data[-1])
                    if is_selected > 0:
                        selected_face_ids.append(face_id)
                    face_id += 1
                
    with open(out_path, 'wb') as f:
        pickle.dump(selected_face_ids, f)
        
def visualize_selected_faces(file_path, id_path, out_path):

    mesh = trimesh.load_mesh(file_path)
    selected_ids = np.load(id_path, allow_pickle=True)
    print(selected_ids)
    
    selected_faces_mesh = mesh.submesh([selected_ids], append=True)

    # Visualize the selected faces (you can customize the appearance)
    selected_faces_mesh.visual.face_colors = [255, 0, 0, 255]  # Set color to red
    selected_faces_mesh.export(out_path)


if __name__ == "__main__":
    # get_template()
    read_selected_mesh()
    visualize_selected_faces(file_path="./output/mano_template.ply", 
                       id_path="assets/mano_backface_ids.pkl",
                       out_path="./output/mano_template_backhand.ply")