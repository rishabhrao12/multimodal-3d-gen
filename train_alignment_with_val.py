import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset
from open_clip import image_transform
from utils import *
from alignment import *
import time
import os

if __name__ == "__main__":
    dataset_dir = "Data/ShapeNetSem/Datasets/"
    train_dataset_path = f"{dataset_dir}final_train.csv"
    val_dataset_path = f"{dataset_dir}final_val.csv"
    test_dataset_path = f"{dataset_dir}final_test.csv"
    img_embd_path = 'Embeddings/PRETRAINED/img_final.pt'
    text_embd_path = 'Embeddings/PRETRAINED/text_final.pt'
    pc_embd_path = 'Embeddings/PRETRAINED/pc_final.pt'

    save_model_path = "TrainedModels/ALIGN/Final_400/"
    os.makedirs(save_model_path, exist_ok=True)
    epochs = 50
    batch_size = 8
    learning_rate = 1e-5
    val_interval, save_interval = 2, 5

    # EmbeddingDataset(dataset_path, text_embd_path, img_embd_path, pc_embd_path)
    train_dataset = EmbeddingDataset(train_dataset_path, text_embd_path, img_embd_path, pc_embd_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = EmbeddingDataset(val_dataset_path, text_embd_path, img_embd_path, pc_embd_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = EmbeddingDataset(test_dataset_path, text_embd_path, img_embd_path, pc_embd_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_model = ComplexAlignEncoder(400)
    align_model.to(device)

    loss_fn = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.Adam(align_model.parameters(), lr=learning_rate)

    all_losses = []
    best_val_loss = float('inf')
    best_epoch = -1
    start_time = time.time()
    for epoch in range(epochs + 1):
        epoch_losses = []
        epoch_start = time.time()
        align_model.train()
        for i, batch in enumerate(train_dataloader):
            idx, text_embd, img_embd, pc_embd = batch

            text_proj, img_proj, pc_proj = align_model(text_embd, img_embd, pc_embd)
            loss_text_point = loss_fn(text_proj, pc_proj)
            loss_text_image = loss_fn(text_proj, img_proj)
            loss_image_point = loss_fn(img_proj, pc_proj)

            #print(text_embd.shape, img_embd.shape, pc_embd.shape)
            #print(text_proj.shape, img_proj.shape, pc_proj.shape)
            #print('Loss: ', loss_text_point, loss_text_image, loss_image_point)
            loss = (loss_text_point + loss_text_image + loss_image_point) / 3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            #break

        avg_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        all_losses.append(avg_epoch_loss)
        epoch_end = time.time()

        # Validation Loop
        if epoch % val_interval == 0:
            align_model.eval()
            total_val_loss = 0
            val_losses = []
            with torch.no_grad():
                for i, val_batch in enumerate(val_dataloader):
                    idx, text_embd, img_embd, pc_embd = batch

                    text_proj, img_proj, pc_proj = align_model(text_embd, img_embd, pc_embd)
                    loss_text_point = loss_fn(text_proj, pc_proj)
                    loss_text_image = loss_fn(text_proj, img_proj)
                    loss_image_point = loss_fn(img_proj, pc_proj)

                    val_loss = (loss_text_point + loss_text_image + loss_image_point) / 3
                    val_losses.append(val_loss.item())

                    #print(text_embd.shape, img_embd.shape, pc_embd.shape)
                    #print(text_proj.shape, img_proj.shape, pc_proj.shape)
                    #print('Loss: ', loss_text_point, loss_text_image, loss_image_point)
                    #break

            average_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {epoch} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {average_val_loss:.4f}')

            # Save model if validation loss improved
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_epoch = epoch
                model_path = f"{save_model_path}BestModel.pth"
                torch.save(align_model.state_dict(), model_path)

        if epoch % save_interval == 0:
            model_path = f"{save_model_path}{epoch}.pth"
            torch.save(align_model.state_dict(), model_path)


        print(f'Epoch {epoch} loss: {avg_epoch_loss}, time taken: {readable_time(epoch_start, epoch_end)}')
        #break
        
    print("Testing the model on the test dataset...")
    align_model.eval()
    test_losses = []
    with torch.no_grad():
        for test_batch in test_dataloader:
            idx, text_embd, img_embd, pc_embd = batch

            text_proj, img_proj, pc_proj = align_model(text_embd, img_embd, pc_embd)
            loss_text_point = loss_fn(text_proj, pc_proj)
            loss_text_image = loss_fn(text_proj, img_proj)
            loss_image_point = loss_fn(img_proj, pc_proj)

            test_loss = (loss_text_point + loss_text_image + loss_image_point) / 3
            test_losses.append(test_loss)
            #print(text_embd.shape, img_embd.shape, pc_embd.shape)
            #print(text_proj.shape, img_proj.shape, pc_proj.shape)
            #print('Loss: ', loss_text_point, loss_text_image, loss_image_point)
            #break

    average_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {average_test_loss:.4f}")

    end_time = time.time()
    print(f"Model trained for {epochs} and took {readable_time(start_time, end_time)} to train, validate and test")
    epoch_losses_np = np.array(all_losses)
    np.save(f'{save_model_path}final_loss.npy', epoch_losses_np)