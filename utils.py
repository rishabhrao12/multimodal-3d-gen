import open3d as o3d
import torch
from PIL import Image
import timm
import random
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer
import torchvision.transforms as transforms
import open_clip
import pandas as pd
from torch.utils.data import DataLoader
from dataset import AlignedModalityDataset
from open_clip import image_transform
import numpy as np

def readable_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print in human-readable format
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def load_dinov2():
    # Load Pretrained DINOv2 Model (Use 'vit_small_patch14_dinov2' for smaller models)
    model_path = "PretrainedModels/dinov2_vits14_pretrain.pth"  # Change to your local path
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=False)

    # Load the state dictionary and remove "mask_token"
    checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint = {k: v for k, v in checkpoint.items() if k != "mask_token"}  # Remove unexpected key

    # Load the modified state dict into the model
    model.load_state_dict(checkpoint, strict=False)  # strict=False allows minor mismatches
    model.eval()  # Set model to evaluation mode
    print('Dinov2 Loaded Successfully!')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def load_clip():
    model_name = "ViT-L-14"  # Change this to ViT-L/14 if needed
    save_path = f"PretrainedModels/clip_vitl14_pretrain.pth"

    clip_model = open_clip.create_model(model_name, pretrained=False)

    # Load saved state dict
    checkpoint = torch.load(save_path, map_location="cpu")
    clip_model.load_state_dict(checkpoint)
    clip_model.eval()

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    print("CLIP Model Loaded Successfully!")
    return clip_model

def load_point_clip():
    save_path = "PretrainedModels/clip_vitl14_pretrain.pth"
    model_name = "ViT-L-14"

    # Load Model Without Downloading
    clip_model = open_clip.create_model(model_name, pretrained=False)
    checkpoint = torch.load(save_path, map_location="cpu")
    clip_model.load_state_dict(checkpoint)
    clip_model.eval()

    # Move Model to GPU if Available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

    print("Point CLIP Model Loaded Successfully!")
    return clip_model

def point_cloud_to_depth_map(pc_data):
    """Converts a 3D point cloud to a 2D depth image."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data.numpy())

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()

    # Capture depth image
    depth = vis.capture_depth_float_buffer(True)
    vis.destroy_window()

    # Convert depth to PIL Image
    depth_np = np.asarray(depth)
    depth_image = Image.fromarray((depth_np * 255).astype(np.uint8))

    return depth_image