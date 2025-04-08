import os
import random
import numpy as np
from PIL import Image
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset import PCModalityDataset

CANONICAL_VIEWS = [
    [1, 0, 0],     # +X
    [-1, 0, 0],    # -X
    [0, 0, 1],     # +Z
    [0, 0, -1],    # -Z
    [1, 1, 1],     # +XYZ
    [-1, -1, -1]   # -XYZ
]

# Normalize all vectors to make them unit directions
CANONICAL_VIEWS = [np.array(v) / np.linalg.norm(v) for v in CANONICAL_VIEWS]

# === Paths ===
dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat.csv"
pc_dir = "Data/ProcessedData/final_template_30cat_pc"
cache_dir = "Data/ProcessedData/final_template_30cat_pc_dmaps_fixed"
os.makedirs(cache_dir, exist_ok=True)

# === Settings ===
num_points = 1024
num_views = 8
width, height = 224, 224  # Size of depth maps

"""
# === Random Camera Sampling ===
def random_camera_position():
    phi = random.uniform(0, 2 * np.pi)
    theta = random.uniform(0, np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]

# === Convert Point Cloud to Depth Map ===
def point_cloud_to_depth_map(pc_data, width=224, height=224, front_vector=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()

    ctr.set_front(front_vector if front_vector else [0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()

    depth = vis.capture_depth_float_buffer(True)
    vis.destroy_window()

    depth_np = np.asarray(depth)
    depth_image = Image.fromarray((depth_np * 255).astype(np.uint8))

    return depth_image
"""

def point_cloud_to_depth_map_fixed(pc_data, front_vector, width=224, height=224):
    """
    Converts a point cloud to a depth map from a given viewpoint.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()

    ctr.set_front(front_vector.tolist())
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])  # You can tweak this if you want other orientations
    ctr.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()

    depth = vis.capture_depth_float_buffer(True)
    vis.destroy_window()

    depth_np = np.asarray(depth)
    depth_image = Image.fromarray((depth_np * 255).astype(np.uint8))

    return depth_image

# === Create Dataset Instance ===
dataset = PCModalityDataset(dataset_path, pc_dir, num_points=num_points)
already_processed = [f for f in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, f))]
print(len(already_processed))

process_count = 0
num_views = len(CANONICAL_VIEWS)

for idx in range(len(dataset)):
    try:
        _, mesh_id, pc_tensor = dataset[idx]
        if mesh_id in already_processed:
            continue
    except ValueError as e:
        print(f"⚠️ {e}")
        continue

    # Convert torch Tensor to numpy for open3d
    point_cloud_sampled = pc_tensor.numpy()

    mesh_cache_dir = os.path.join(cache_dir, mesh_id)
    os.makedirs(mesh_cache_dir, exist_ok=True)

    for view_idx in range(num_views):
        save_path = os.path.join(mesh_cache_dir, f"{view_idx}.png")

        if os.path.exists(save_path):
            continue

        # Use the fixed canonical camera view instead of random
        front_vector = CANONICAL_VIEWS[view_idx]

        depth_map = point_cloud_to_depth_map_fixed(
            point_cloud_sampled,
            front_vector=front_vector,
            width=224,
            height=224
        )

        depth_map.save(save_path)

    process_count += 1

    # Optional: stop after a few for debugging
    if process_count == 10:
        break

print(f"✅ Processed {process_count} meshes.")