import matplotlib.pyplot as plt
from alignment import *
from utils import *
from rag import *
from dataset import *
import time
from torch.utils.data import random_split

if __name__ == "__main__":
    dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat.csv" #"Data/ShapeNetSem/Datasets/subset_template_200.csv"
    embed_path = "Embeddings/ALIGN/final_template_30cat_fixed_fullset.pt" #"Embeddings/ALIGN/subset_template_new_loss_fixed_all.pt"
    pc_dir = "Data/ProcessedData/final_template_30cat_pc/" #"Data/ProcessedData/PointClouds/"
    save_dir = f"TrainedModels/RAG/final_template_30cat_fixed_fullset_new/" # final_template_30cat #f"TrainedModels/RAG/four_hidden_val/"
    os.makedirs(save_dir, exist_ok=True)
    
    num_points = 1024  # Number of points in the point cloud
    in_dim = 400 + num_points * 3  # Input dimension (projection + flattened point cloud)
    hidden_dim = 512  # Hidden layer size
    out_dim = num_points * 3  # Output dimension (for point cloud with 3 coordinates per point)
    batch_size = 8
    epochs = 500 # 1500
    save_interval = 50 # 250
    
    cmr = CrossModalRetrival(dataset_path, embed_path)

    data_dict = torch.load(embed_path)
    dataset = RAGDataset(data_dict, dataset_path, pc_dir)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = int(0.1 * len(dataset))    # 10% for validation
    test_size = len(dataset) - train_size - val_size  # Remaining 10% for test
    batch_size = 8

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    modality_choices = ['text', 'img', 'pc']

    rag_decoder = RAGDecoder(in_dim, hidden_dim, out_dim, num_hidden_layers=4)
    #rag_decoder = RAGDecoderComplex(in_dim, hidden_dim, out_dim, num_hidden_layers=8, dropout_rate=0.2)
    optimizer = torch.optim.AdamW(rag_decoder.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    all_losses,val_losses = [], []

    start_time = time.time()
    for epoch in range(epochs + 1):
        epoch_losses = []
        epoch_start = time.time()
        rag_decoder.train()
        for batch in train_loader:
            point_clouds = []
            projections = []
            target_point_clouds = batch['target_pc'].to(dtype=torch.float32)
            for i in range(batch_size):
                # idx, text_proj, img_proj, pc_proj, category = batch['index'][i], batch['text_proj'][i], batch['img_proj'][i], batch['pc_proj'][i], batch['category'][i]
                # print(idx, text_proj.shape, img_proj.shape, pc_proj.shape, category)

                # Choose one random projection to use
                choice = random.choice(modality_choices)
                projection = batch[f"{choice}_proj"][i]
                pc = get_aligned_pc_from_projection(pc_dir, projection, choice, cmr)
                # print(pc.shape)
                projections.append(projection)
                point_clouds.append(pc)

            #print(f"Point Cloud Input Shape: {point_clouds[0].shape} {len(point_clouds)}, Projections Shape: {projections[0].shape} {len(projections)}")
            point_clouds = torch.stack(point_clouds) # (B, num_points, 3)
            projections = torch.stack(projections) # (B, 1, 400)
            #print(f"After stack Point Cloud Input Shape: {point_clouds.shape}, Projections Shape: {projections.shape}")
            point_clouds = point_clouds.view(batch_size, -1)
            projections = projections.squeeze(1)
            #print(f"After squeeze/view Point Cloud Input Shape: {point_clouds.shape}, Projections Shape: {projections.shape}")
            combined_input = torch.concat((projections, point_clouds), dim=1)
            combined_input = combined_input.to(dtype=torch.float32)
            pred_point_clouds = rag_decoder(combined_input)
            loss = chamfer_distance(pred_point_clouds, target_point_clouds)

            #print(f"Point Cloud Input Shape: {point_clouds.shape}, Projections Shape: {projections.shape}")
            #print(f"Combined Input Shape: {combined_input.shape}")
            #print(f"Predicted Input Shape: {pred_point_clouds.shape}, Target Shape: {target_point_clouds.shape}")
            #print(f"Loss: {loss}, {type(loss)}")

            optimizer.zero_grad()
            loss.backward()
            #break
            """
            for param in rag_decoder.parameters():
                if param.grad is None:
                    print(f"Gradient is None for parameter: {param}")
                else:
                    print(f"Gradient for has shape {param.grad.shape}")
            
            """
            optimizer.step()
            epoch_losses.append(loss.item())

        #scheduler.step()  
        #break
        avg_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        all_losses.append(avg_epoch_loss)

        val_epoch_loss = []
        with torch.no_grad():
            for batch in val_loader:
                point_clouds = []
                projections = []
                target_point_clouds = batch['target_pc'].to(dtype=torch.float32)
                for i in range(batch_size):
                    # idx, text_proj, img_proj, pc_proj, category = batch['index'][i], batch['text_proj'][i], batch['img_proj'][i], batch['pc_proj'][i], batch['category'][i]
                    # print(idx, text_proj.shape, img_proj.shape, pc_proj.shape, category)

                    # Choose one random projection to use
                    choice = random.choice(modality_choices)
                    projection = batch[f"{choice}_proj"][i]

                    #pc = get_aligned_pc_from_projection(pc_dir, projection, choice, cmr)
                    pc = get_random_aligned_pc_from_projection(pc_dir, projection, choice, cmr, k=10)
                    # print(pc.shape)
                    projections.append(projection)
                    point_clouds.append(pc)

                #print(f"Point Cloud Input Shape: {point_clouds[0].shape} {len(point_clouds)}, Projections Shape: {projections[0].shape} {len(projections)}")
                point_clouds = torch.stack(point_clouds) # (B, num_points, 3)
                projections = torch.stack(projections) # (B, 1, 400)
                #print(f"After stack Point Cloud Input Shape: {point_clouds.shape}, Projections Shape: {projections.shape}")
                point_clouds = point_clouds.view(batch_size, -1)
                projections = projections.squeeze(1)
                #print(f"After squeeze/view Point Cloud Input Shape: {point_clouds.shape}, Projections Shape: {projections.shape}")
                combined_input = torch.concat((projections, point_clouds), dim=1)
                combined_input = combined_input.to(dtype=torch.float32)
                pred_point_clouds = rag_decoder(combined_input)
                loss = chamfer_distance(pred_point_clouds, target_point_clouds)
                val_epoch_loss.append(loss.item())

        avg_val_loss = sum(val_epoch_loss)/len(val_epoch_loss)
        val_losses.append(avg_val_loss)
        epoch_end = time.time()

        if epoch % save_interval == 0:
            model_path = f"{save_dir}{epoch}.pth"
            # print(model_path)
            torch.save(rag_decoder.state_dict(), model_path)
        
            print(f'Epoch {epoch} loss: {avg_epoch_loss}, val loss: {avg_val_loss}, time taken: {readable_time(epoch_start, epoch_end)}')
        
    end_time = time.time()

    print(f"Model trained for {epochs} and took {readable_time(start_time, end_time)} to train")
    epoch_losses_np = np.array(all_losses)
    epoch_val_losses_np = np.array(val_losses)
    #print(loss_path)
    np.save(f"{save_dir}train_loss.npy", epoch_losses_np)
    np.save(f"{save_dir}val_loss.npy", epoch_val_losses_np)