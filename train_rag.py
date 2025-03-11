import matplotlib.pyplot as plt
from alignment import *
from utils import *
from rag import *
from dataset import *
import time

if __name__ == "__main__":
    dataset_path = "Data/ShapeNetSem/Datasets/subset_template_200.csv"
    embed_path = "Embeddings/ALIGN/subset_template_200.pt"
    pc_dir = "Data/ProcessedData/PointClouds/"
    save_dir = f"TrainedModels/RAG/four_hidden/"
    os.makedirs(save_dir, exist_ok=True)
    
    num_points = 1024  # Number of points in the point cloud
    in_dim = 400 + num_points * 3  # Input dimension (projection + flattened point cloud)
    hidden_dim = 512  # Hidden layer size
    out_dim = num_points * 3  # Output dimension (for point cloud with 3 coordinates per point)
    batch_size = 8
    epochs = 4000

    cmr = CrossModalRetrival(dataset_path, embed_path)

    data_dict = torch.load(embed_path)
    dataset = RAGDataset(data_dict, dataset_path, pc_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    modality_choices = ['text', 'img', 'pc']

    rag_decoder = RAGDecoder(in_dim, hidden_dim, out_dim, num_hidden_layers=4)
    optimizer = torch.optim.Adam(rag_decoder.parameters(), lr=1e-4)
    all_losses = []

    start_time = time.time()
    for epoch in range(epochs + 1):
        epoch_losses = []
        epoch_start = time.time()
        for batch in dataloader:
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

            point_clouds = torch.stack(point_clouds) # (B, num_points, 3)
            projections = torch.stack(projections) # (B, 1, 400)
            point_clouds = point_clouds.view(batch_size, -1)
            projections = projections.squeeze(1)
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

            """
            for param in rag_decoder.parameters():
                if param.grad is None:
                    print(f"Gradient is None for parameter: {param}")
                else:
                    print(f"Gradient for has shape {param.grad.shape}")
            
            """
            optimizer.step()
            epoch_losses.append(loss.item())
            
        
        avg_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        all_losses.append(avg_epoch_loss)
        epoch_end = time.time()

        if epoch % 500 == 0:
            model_path = f"{save_dir}{epoch}.pth"
            # print(model_path)
            torch.save(rag_decoder.state_dict(), model_path)
        
            print(f'Epoch {epoch} loss: {avg_epoch_loss}, time taken: {readable_time(epoch_start, epoch_end)}')
        
    end_time = time.time()

    print(f"Model trained for {epochs} and took {readable_time(start_time, end_time)} to train")
    epoch_losses_np = np.array(all_losses)
    loss_path = f"{save_dir}loss.npy"
    #print(loss_path)
    np.save(f"{save_dir}loss.npy", epoch_losses_np)