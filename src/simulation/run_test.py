import simulation.simulate as sim
from plyfile import PlyData
import argparse
import numpy as np

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))


def read_ply_file(file_path):
    plydata = PlyData.read(file_path)
    
   # Extract vertices as a NumPy array
    vertex_data = plydata['vertex']
    vertices = np.array([(vertex['x'], vertex['y'], vertex['z']) for vertex in vertex_data])

    # Extract faces as a NumPy array
    face_data = plydata['face']
    faces = np.array([face[0] for face in face_data])

    return vertices, faces


if __name__ == '__main__':
     
    #  parser = argparse.ArgumentParser(description="render and evaluate the interaction between hand and objects")

    #  parser.add_argument("--")
    sample_info = {}
    sample_info["hand_verts"],sample_info["hand_faces"]  = read_ply_file("sim_obj/0_hand.ply")
    sample_info["obj_verts"], sample_info["obj_faces"]= read_ply_file("sim_obj/0_obj.ply") 
     
    distances = sim.process_sample(1, sample_info, use_gui= True, vhacd_exe='/home/yaxun/for_yvmeng/v-hacd/app/build/TestVHACD', wait_time=5)
