import numpy as np
import trimesh
from mesh_to_sdf import get_point_cloud
import os
import pandas as pd
import gc

def point_cloud_from_mesh(mesh_id, mesh_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, f"{mesh_id}.obj")
    save_path = os.path.join(save_dir, f"{mesh_id}.npy")
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Generate point cloud
    points = get_point_cloud(mesh, number_of_points=1024*2)  # Oversampling, but will be handled during loading
    
    # Save point cloud
    np.save(save_path, points)
    print(f"Completed processing: {mesh_id} {points.shape}")


if __name__ == "__main__":
    df = pd.read_csv("Data/ShapeNetSem/Datasets/subset_template_200.csv")
    mesh_ids = df['fullId'].to_list()
    mesh_dir = "Data/ShapeNetSem/Files/models-OBJ/models"
    save_dir = "Data/ProcessedData/PointClouds"
    already_processed = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
    print(len(already_processed))
    for mesh_id in mesh_ids:
        if f"{mesh_id}.npy" not in already_processed:
            point_cloud_from_mesh(mesh_id, mesh_dir, save_dir)
