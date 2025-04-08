from utils import *
from alignment import *
import time
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np

def get_dmaps(point_cloud, batch_size):
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

def encode_and_aggregate_views(depth_maps_batch, pointclip_model):
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

def get_pca(embeddings):
    print(f"Before PCA embedding shape: {embeddings.shape}")
    pca = PCA(n_components=150, random_state=42)
    pca_result = pca.fit_transform(embeddings)
    print(f"After PCA embedding shape: {pca_result.shape}")
    return pca_result

def get_tsne(pca_result):
    tsne = TSNE(n_components=2, random_state=42)
    print(f"Before TSNE embedding shape: {pca_result.shape}")
    tsne_result = tsne.fit_transform(pca_result)
    print(f"After TSNE embedding shape: {tsne_result.shape}")
    return tsne_result

def plot_tsne(tsne_result, labels, cat_mapping, modality, exp_name):
    plot_data = pd.DataFrame({
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'numeric_label': labels,
        'category': [cat_mapping[label] for label in labels]
    })

    # Create an interactive scatter plot using Plotly Express
    fig = px.scatter(
        plot_data, x='x', y='y', color='category',
        hover_data={'numeric_label': True, 'category': True},
        title=f"Interactive t-SNE Visualization with Category Info for {modality}"
    )

    save_path = f"{exp_name}_align_{modality}_clusters.png"
    fig.write_image(save_path)

if __name__ == "__main__":
    checkpoint_path = "TrainedModels/ALIGN/final_template_30cat_fixed/100.pth"# "TrainedModels/ALIGN/final_template_30cat_fixed/100.pth" #"TrainedModels/ALIGN/subset_200_direct_new_loss_fixed_all/140.pth"
    # dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat.csv"
    dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat_test.csv"
    image_dir = "Data/ShapeNetSem/Images/final_template_30cat"
    pc_dir = "Data/ProcessedData/final_template_30cat_pc"
    exp_name = "Clusters/30cat_test"
    save_path = f"Embeddings/ALIGN/final_template_30cat_test.pt" #f"Embeddings/ALIGN/final_template_30cat_fixed.pt"
    num_dataset_samples = 1500
    try:
        dinov2_encoder = load_dinov2()
        clip_encoder = load_clip()
        dinov2_encoder.eval()
        clip_encoder.eval()
        print('All Models loaded succesfully and set to eval mode')

        # Initialize the model
        align_model = ComplexAlignEncoder(400, dropout_rate=0.2)  # Ensure the architecture matches

        # Load the saved weights
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU

        # Apply the weights to the model
        align_model.load_state_dict(state_dict)

        # Set to evaluation mode (if needed)
        align_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        align_model.to(device)

        print("Align Model weights loaded successfully and model set to eval")
    except:
        print('Error in Loading Models')

    # Set up CLIP preprocessing
    preprocess = image_transform(
        clip_encoder.visual.image_size,  # Correct image size for CLIP
        is_train=False  # Ensures we use inference preprocessing
    )

    dataset = AlignedModalityDataset(dataset_path, image_dir, pc_dir)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data = pd.read_csv(dataset_path)

    all_text_projections, all_image_projections, all_pc_projections  = [], [], []
    all_idx = []
    all_cats = []

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, tokenized_text, image_tensor, point_cloud = batch
            tokenized_text = tokenized_text.to(device) # (B, 77)
            image_tensor = image_tensor.to(device) # (B, 3, 518, 518)

            point_cloud = point_cloud.to(device) # (B, 1024, 3)

            # Assuming point_cloud is a batch of point clouds (shape: [batch_size, N, 3])
            batch_size = point_cloud.shape[0]

            # Convert each point cloud to a depth map and preprocess it
            #depth_maps = [preprocess(inference_dmap_random_view(point_cloud[i])).unsqueeze(0) for i in range(batch_size)]
            depth_maps = get_dmaps(point_cloud, batch_size)
            # Stack depth maps into a single batch tensor
            # depth_maps = torch.cat(depth_maps, dim=0).to(device)  # Shape: [batch_size, 3, H, W]
            #print(depth_maps.shape)
            
            #print(tokenized_text.shape, image_tensor.shape, depth_maps.shape)
            text_emb = clip_encoder.encode_text(tokenized_text) # (B, 768)
            img_emb = dinov2_encoder(image_tensor) # (B, 384)
            # pc_emb = clip_encoder.encode_image(depth_maps) # (B, 768)
            pc_emb = encode_and_aggregate_views(depth_maps, clip_encoder)
            #print(text_emb.shape, img_emb.shape, pc_emb.shape)
            
            text_proj, img_proj, pc_proj = align_model(text_emb, img_emb, pc_emb)

            all_text_projections.append(text_proj)
            all_image_projections.append(img_proj)
            all_pc_projections.append(pc_proj)
            all_idx.append(idx.item())

            all_cats.append(data.loc[int(idx.item()), 'category'])
            print(f"{i} complete")
            if i == num_dataset_samples - 1:
                break
    #"""
    end_time = time.time()
    print(f"Model took {readable_time(start_time, end_time)} to train")

    print(len(all_text_projections), len(all_image_projections), len(all_pc_projections))   
    print(all_text_projections[0].shape, all_image_projections[0].shape, all_pc_projections[0].shape)

    projections_t = torch.concat(all_text_projections, dim=0)
    projections_i = torch.concat(all_image_projections, dim=0)
    projections_pc = torch.concat(all_pc_projections, dim=0)

    print(projections_t.shape, projections_i.shape, projections_pc.shape)

    data_dict = {
        "text_proj": projections_t,
        "img_proj": projections_i,
        "pc_proj": projections_pc,
        "index": all_idx,
        "category": all_cats
    }

    torch.save(data_dict, save_path)

    projections_t = projections_t.numpy()
    projections_i = projections_i.numpy()
    projections_pc = projections_pc.numpy()

    formatted_categories = []
    for subcategory in all_cats:
        subcategories = subcategory.split(',')
        new_subcategories = [s for s in subcategories if '_' not in s]
        formatted_categories.append(new_subcategories[0])
    
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    categories = np.array(formatted_categories).reshape(-1, 1)
    # Fit and transform the data
    one_hot_encoded = encoder.fit_transform(categories)
    print(f"After OHE categories shape: {one_hot_encoded.shape}")
    labels = one_hot_encoded.argmax(axis=1)
    print(f"After Argmax labels shape: {labels.shape}")
    cat_mapping = {i: cat for i, cat in enumerate(encoder.categories_[0])}
    print("Mapping (integer label -> string category):", cat_mapping)
    

    pca_t = get_pca(projections_t)
    pca_i = get_pca(projections_i)
    pca_pc = get_pca(projections_pc)
    print('PCA Completed')

    tsne_t = get_tsne(pca_t)
    tsne_i = get_tsne(pca_i)
    tsne_pc = get_tsne(pca_pc)
    print('TSNE Completed')

    plot_tsne(tsne_t, labels, cat_mapping, "text", exp_name)
    plot_tsne(tsne_i, labels, cat_mapping, "image", exp_name)
    plot_tsne(tsne_pc, labels, cat_mapping, "pc", exp_name)
    print("Plotting Completed")
    #"""