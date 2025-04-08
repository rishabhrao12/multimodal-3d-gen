from utils import *
from alignment import *
import time
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
from dataset import ChoiceEmbeddingDataset
import os

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
    train_dataset_path = "Data/ShapeNetSem/Datasets/final_template_20cat_train.csv"
    val_dataset_path = "Data/ShapeNetSem/Datasets/final_template_20cat_val.csv"
    test_dataset_path = "Data/ShapeNetSem/Datasets/final_template_20cat_test.csv"
    embd_dir = "Embeddings/PRETRAINED/final_template_30cat"

    image_dir = "Data/ShapeNetSem/Images/final_template_30cat"
    pc_dir = "Data/ProcessedData/final_template_30cat"
    checkpoint_path = "TrainedModels/ALIGN/final_template_20cat/250.pth"
    num_dataset_samples = 800
    exp = "train"
    save_path = f"Embeddings/ALIGN/final_template_20cat/"
    os.makedirs(save_path, exist_ok=True)

    try:
        # Initialize the model
        align_model = AlignEncoder(embed_dim=400)  
        # Load the saved weights
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU
        # Apply the weights to the model
        align_model.load_state_dict(state_dict)
        # Set to evaluation mode (if needed)
        align_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        align_model.to(device)

        print("Align Model weights loaded successfully and model set to eval")
    except Exception as e:
        print(f'Error in Loading Models \n {e}')


    train_dataset = ChoiceEmbeddingDataset(train_dataset_path, embd_dir)
    val_dataset = ChoiceEmbeddingDataset(val_dataset_path, embd_dir)
    test_dataset = ChoiceEmbeddingDataset(test_dataset_path, embd_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_text_projections, all_image_projections, all_pc_projections, all_cats, all_idx = [], [], [], [], []
    start_time = time.time()

    dataset_path = "Data/ShapeNetSem/Datasets/final_template_20cat.csv"
    data = pd.read_csv(dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            idx, text_emb, img_emb, pc_emb = batch
            text_emb = text_emb.to(device) # (B, 77)
            img_emb = img_emb.to(device) # (B, 3, 518, 518)
            pc_emb = pc_emb.to(device) # (B, 1024, 3)
            text_proj, img_proj, pc_proj = align_model(text_emb, img_emb, pc_emb)

            all_text_projections.append(text_proj)
            all_image_projections.append(img_proj)
            all_pc_projections.append(pc_proj)

            all_cats.append(data.loc[int(idx.item()), 'category'])
            all_idx.append(idx)
            if i == num_dataset_samples - 1:
                break

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

    #torch.save(data_dict, save_path)

    #"""
    projections_t = projections_t.numpy()
    projections_i = projections_i.numpy()
    projections_pc = projections_pc.numpy()

    
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    categories = np.array(all_cats).reshape(-1, 1)
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

    plot_tsne(tsne_t, labels, cat_mapping, "text", exp)
    plot_tsne(tsne_i, labels, cat_mapping, "image", exp)
    plot_tsne(tsne_pc, labels, cat_mapping, "pc", exp)
    print("Plotting Completed")
    # """