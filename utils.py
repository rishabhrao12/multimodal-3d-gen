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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def inference_dmap_random_view(pc_data, device="cuda", width=224, height=224):
    """
    Generates a depth map from a random canonical view and returns its embedding.

    Args:
        pc_data (numpy.ndarray): Nx3 point cloud.
        encoder (torch.nn.Module): Pretrained encoder.
        preprocess_fn (callable): Preprocess function for depth map images.
        device (str): 'cuda' or 'cpu'.
        width (int): Width of depth map.
        height (int): Height of depth map.

    Returns:
        torch.Tensor: Embedding for retrieval.
    """
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

    # Select one random view index
    random_view_idx = random.randint(0, len(CANONICAL_VIEWS) - 1)
    front_vector = CANONICAL_VIEWS[random_view_idx]

    # Generate depth map from selected view
    depth_map = point_cloud_to_depth_map_fixed(pc_data, front_vector, width, height)

    return depth_map

def get_all_canonical_dmaps(pc_data, device="cuda", width=224, height=224):
    """
    Generates a depth map from a random canonical view and returns its embedding.

    Args:
        pc_data (numpy.ndarray): Nx3 point cloud.
        encoder (torch.nn.Module): Pretrained encoder.
        preprocess_fn (callable): Preprocess function for depth map images.
        device (str): 'cuda' or 'cpu'.
        width (int): Width of depth map.
        height (int): Height of depth map.

    Returns:
        torch.Tensor: Embedding for retrieval.
    """
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

    # Select one random view index
    random_view_idx = random.randint(0, len(CANONICAL_VIEWS) - 1)
    depth_maps = []

    for front_vector in CANONICAL_VIEWS:
        # Generate depth map from selected view
        depth_map = point_cloud_to_depth_map_fixed(pc_data, front_vector, width, height)

        # Append depth map to the list
        depth_maps.append(depth_map)

    return depth_maps

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

def get_plot(ref_pc):
    predicted_scatter = go.Scatter3d(
        x=ref_pc[:, 0],
        y=ref_pc[:, 1],
        z=ref_pc[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name="Predicted"
    )

    # Create the subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Predicted Point Cloud"),
        specs=[[{'type': 'scatter3d'}]]
    )

    # Add the scatter plots to the subplots
    fig.add_trace(predicted_scatter, row=1, col=1)

    # Update layout with axis titles and a main title
    fig.update_layout(
        title=f"Predicted",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        showlegend=True
    )

    return fig

def get_dmaps_from_batch(point_cloud, batch_size, preprocess):
    depth_maps_batch = []

    for i in range(batch_size):
        # Get all depth maps (PIL images) for each point cloud
        depth_maps = get_all_canonical_dmaps(point_cloud[i])  # Returns list of 6 PIL Images

        # Preprocess each depth map (e.g., Resize, ToTensor, Normalize, etc.)
        preprocessed_maps = [preprocess(dmap).unsqueeze(0) for dmap in depth_maps]  # Each is (1, 3, H, W)

        # Stack them into (Views, 3, H, W)
        preprocessed_maps = torch.cat(preprocessed_maps, dim=0)  # Shape: (Views, 3, H, W)

        # Append to batch list
        depth_maps_batch.append(preprocessed_maps)

    # Final batch tensor shape: (Batch, Views, 3, H, W)
    depth_maps_batch = torch.stack(depth_maps_batch, dim=0)
    return depth_maps_batch

def encode_and_aggregate_views(depth_maps_batch, pointclip_model, device='cpu'):
    """
    Encodes depth maps for all views and aggregates them for each point cloud in the batch.

    Args:
        depth_maps_batch (torch.Tensor): Tensor of shape (B, V, 3, 224, 224)
        pointclip_model (nn.Module): Pretrained PointCLIP model
        aggregation (str): "mean" or "max"
        device (str): Device to run on
    
    Returns:
        torch.Tensor: Aggregated embeddings of shape (B, Embed)
    """
    B, V, C, H, W = depth_maps_batch.shape

    # Flatten batch and view dimensions to feed into the encoder
    flattened_dmaps = depth_maps_batch.view(B * V, C, H, W).to(device)

    with torch.no_grad():
        # Encode all images at once: shape (B * V, Embed)
        encoded_views = pointclip_model.encode_image(flattened_dmaps)

    # Reshape back to (B, V, Embed)
    encoded_views = encoded_views.view(B, V, -1)

    aggregated_embeddings = encoded_views.mean(dim=1)

    return aggregated_embeddings