import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CategoryChoiceEmbeddingDatasetAllDmap
from open_clip import image_transform
from utils import *
from alignment import *
import time
from torch.utils.data import random_split
import os

import torch
import torch.nn.functional as F

def category_supervised_contrastive_loss(embeddings, categories, temperature=0.07):
    """
    Args:
        embeddings: Tensor of shape [batch_size * num_modalities, embed_dim]
        categories: Tensor of shape [batch_size * num_modalities] (integer labels)
        temperature: Scaling factor (default 0.07)

    Returns:
        Scalar loss
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity matrix (scaled by temperature)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # [batch_size, batch_size]

    # Mask to find positives (same category label)
    category_mask = categories.unsqueeze(0) == categories.unsqueeze(1)  # [batch_size, batch_size]

    # Exclude self-comparisons
    self_mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    positive_mask = category_mask & ~self_mask

    # Exponentiate similarities for denominator (zero out self-similarities)
    exp_sim = torch.exp(similarity_matrix) * (~self_mask)

    # Log probability of positive pairs
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Average over positive pairs for each anchor
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-12)

    # Loss is the negative mean over all anchors
    loss = -mean_log_prob_pos.mean()

    return loss


if __name__ == "__main__":

    epochs = 100

    dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat.csv" #"Data/ShapeNetSem/Datasets/subset_template_200.csv"
    train_dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat_train.csv"
    val_dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat_val.csv"
    test_dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat_test.csv"

    image_dir = "Data/ShapeNetSem/Images/final_template_30cat" #"Data/ShapeNetSem/Images/subset_200"
    depth_dir = "Data/ProcessedData/final_template_30cat_pc_dmaps_fixed" #"Data/ProcessedData/subset_template_200_dmaps_fixed_all"
    embed_dir = "Embeddings/PRETRAINED/final_template_30cat_fixed" #"Embeddings/PRETRAINED/subset_template_200_fixed"
    save_dir = "TrainedModels/ALIGN/final_template_30cat_fixed_fullset/" #"TrainedModels/ALIGN/subset_200_direct_new_loss_fixed_all/"
    os.makedirs(save_dir, exist_ok=True)
    save_interval = 20

    
    # all_categories = ['PillBottle', 'Books', 'Couch', 'Plant', 'Truck', 'Table', 'ChestOfDrawers', 'DrinkingUtensil', 'Bed', 'Fan', 'CeilingFan', 'Showerhead', 'Vase', 'PersonStanding', 'Faucet', 'MediaStorage', 'PottedPlant', 'USBStick', 'Camera']
    all_categories = ['WallArt', 'Speaker', 'ChestOfDrawers', 'Chair', 'Plant', 'PersonStanding', 'Couch', 'Candle', 'Desk', 'Bench', 'TV', 'TrashBin', 'Book', 'Monitor', 'Pencil', 'Camera', 'Table', 'PottedPlant', 'Gun', 'Cabinet', 'Vase', 'ToyFigure', 'CellPhone', 'FoodItem', 'Refrigerator', 'Piano', 'Guitar', 'MediaPlayer', 'Curtain', 'Ipod']
    label_to_idx = {label: idx for idx, label in enumerate(all_categories)}

    train_dataset = CategoryChoiceEmbeddingDatasetAllDmap(train_dataset_path, embed_dir, label_to_idx)
    val_dataset = CategoryChoiceEmbeddingDatasetAllDmap(val_dataset_path, embed_dir, label_to_idx)
    test_dataset = CategoryChoiceEmbeddingDatasetAllDmap(test_dataset_path, embed_dir, label_to_idx)
    """
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining 10% for test
    """
    batch_size = 8 # 8
    # Split dataset
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # DataLoaders
    dataset = CategoryChoiceEmbeddingDatasetAllDmap(dataset_path, embed_dir, label_to_idx)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    align_model = ComplexAlignEncoder(400, dropout_rate=0.2) # 400 # Ensure the architecture matches
    align_model.to(device)

    loss_fn = NTXentLoss(temperature=0.07)
    # optimizer = torch.optim.Adam(align_model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(align_model.parameters(), lr=1e-4, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    all_losses,val_losses = [], []

    start_time = time.time()
    for epoch in range(epochs+1):
        epoch_losses = []
        align_model.train()
        epoch_start = time.time()
        for i, batch in enumerate(train_loader): # change back to trainloader
            idx, text_emb, img_emb, pc_emb, category_batch = batch
            agg_pc_emb = pc_emb.mean(dim=1)
            # print(text_emb.shape, img_emb.shape, pc_emb.shape, agg_pc_emb.shape, category_batch.shape)
            
            #print(text_emb.shape, img_emb.shape, agg_pc_emb.shape)
            # print(len(category), len(set(category)))
            text_proj, img_proj, pc_proj = align_model(text_emb, img_emb, agg_pc_emb)
            #print(text_proj.shape, img_proj.shape, pc_proj.shape)
            combined_projection = torch.cat([text_proj, img_proj, pc_proj], dim=0)
            categories = category_batch.repeat(3).to(device)

            avg_loss = category_supervised_contrastive_loss(combined_projection, categories)
            #print(combined_projection.shape, categories.shape)
            
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
            epoch_losses.append(avg_loss.item())
            
        #break
        avg_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        all_losses.append(avg_epoch_loss)
        scheduler.step()

        # Validation loop
        align_model.eval()
        val_epoch_loss = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                idx, text_emb, img_emb, pc_emb, category_batch = batch
                agg_pc_emb = pc_emb.mean(dim=1)

                #print(text_emb.shape, img_emb.shape, agg_pc_emb.shape)
                # print(len(category), len(set(category)))
                text_proj, img_proj, pc_proj = align_model(text_emb, img_emb, agg_pc_emb)
                #print(text_proj.shape, img_proj.shape, pc_proj.shape)
                combined_projection = torch.cat([text_proj, img_proj, pc_proj], dim=0)
                categories = category_batch.repeat(3).to(device)

                avg_loss = category_supervised_contrastive_loss(combined_projection, categories)
                #print(combined_projection.shape, categories.shape)
                val_epoch_loss.append(avg_loss.item())

        avg_val_loss = sum(val_epoch_loss)/len(val_epoch_loss)
        val_losses.append(avg_val_loss)
        
        epoch_end = time.time()
        
        if epoch % save_interval == 0:
            model_path = f"{save_dir}{epoch}.pth"
            torch.save(align_model.state_dict(), model_path)

        print(f'Epoch {epoch} loss: {avg_epoch_loss}, val loss: {avg_val_loss}, time taken: {readable_time(epoch_start, epoch_end)}')

        

    end_time = time.time()
    print(f"Model trained for {epochs} and took {readable_time(start_time, end_time)} to train")
    epoch_losses_np = np.array(all_losses)
    epoch_losses_val_np = np.array(val_losses)
    np.save(f'{save_dir}train_loss.npy', epoch_losses_np)
    np.save(f'{save_dir}val_loss.npy', epoch_losses_val_np)