import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import AlignedModalityDataset
from open_clip import image_transform
from utils import *
from alignment import *
import time

if __name__ == "__main__":
    try:
        dinov2_encoder = load_dinov2()
        clip_encoder = load_clip()
        pclip_encoder = load_point_clip()
        dinov2_encoder.eval()
        clip_encoder.eval()
        pclip_encoder.eval()
        print('All Models loaded succesfully and set to eval mode')
    except:
        print('Error in Loading Models')

    dataset_path = "Data/ShapeNetSem/Datasets/subset_template_200.csv"
    image_dir = "Data/ShapeNetSem/Images/subset_200"
    pc_dir = "Data/ProcessedData/PointClouds"
    #save_dir = "TrainedModels/ALIGN/final_template_1k/"
    save_interval = 5

    # Set up CLIP preprocessing
    preprocess = image_transform(
        clip_encoder.visual.image_size,  # Correct image size for CLIP
        is_train=False  # Ensures we use inference preprocessing
    )

    dataset = AlignedModalityDataset(dataset_path, image_dir, pc_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    align_model = load_alignment(checkpoint_path="TrainedModels/ALIGN/Baseline/150.pth", align_embd=400)  # Ensure the architecture matches
    align_model.to(device)

    loss_fn = NTXentLoss(temperature=0.07)

    all_losses = []

    epoch_losses = []
    align_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, tokenized_text, image_tensor, point_cloud = batch
            tokenized_text = tokenized_text.to(device) # (B, 77)
            image_tensor = image_tensor.to(device) # (B, 3, 518, 518)
            
            point_cloud = point_cloud.to(device) # (B, 1024, 3)

            #print(tokenized_text.shape, image_tensor.shape, point_cloud.shape)

            # Assuming point_cloud is a batch of point clouds (shape: [batch_size, N, 3])
            batch_size = point_cloud.shape[0]

            # Convert each point cloud to a depth map and preprocess it
            depth_maps = [preprocess(point_cloud_to_depth_map(point_cloud[i])).unsqueeze(0) for i in range(batch_size)]

            # Stack depth maps into a single batch tensor
            depth_maps = torch.cat(depth_maps, dim=0).to(device)  # Shape: [batch_size, 3, H, W]
            #print(depth_maps.shape)

            with torch.no_grad():
                text_emb = clip_encoder.encode_text(tokenized_text) # (B, 768)
                img_emb = dinov2_encoder(image_tensor) # (B, 384)
                pc_emb = pclip_encoder.encode_image(depth_maps) # (B, 768)

            #print(text_emb.shape, img_emb.shape, pc_emb.shape)
            text_proj, img_proj, pc_proj = align_model(text_emb, img_emb, pc_emb)
            #print(text_proj.shape, img_proj.shape, pc_proj.shape)
            loss_text_point = loss_fn(text_proj, pc_proj)
            loss_text_image = loss_fn(text_proj, img_proj)
            loss_image_point = loss_fn(img_proj, pc_proj)

            #print('Loss: ', loss_text_point, loss_text_image, loss_image_point)
            avg_loss = (loss_text_point + loss_text_image + loss_image_point) / 3
            epoch_losses.append(avg_loss.item())

    avg_epoch_loss = sum(epoch_losses)/len(epoch_losses)
    print(avg_epoch_loss)
