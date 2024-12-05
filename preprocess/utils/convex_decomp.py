import coacd
import trimesh
import os
 

def decomp_mesh(input_file, out_dir):
    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh) # a list of convex hulls.
    for i, p in enumerate(parts):
        v, f = p[0], p[1]
        comp_mesh = trimesh.Trimesh(v, f)
        comp_mesh.export(os.path.join(out_dir, f"{i}.ply"))
        
def main():
    data_dir = "preprocess/collected_data/tripo_model/"
    for model in os.listdir(data_dir):
        if model.startswith("."):
            continue
        model_dir = os.path.join(data_dir, model)
        input_file = os.path.join(model_dir, "output.obj")
        out_dir = os.path.join(model_dir, "coacd")
        if os.path.exists(out_dir):
            continue
        os.makedirs(out_dir)
        print(out_dir)
        decomp_mesh(input_file, out_dir)
        
if __name__ == "__main__":
    main()