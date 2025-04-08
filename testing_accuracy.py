from alignment import *
from dataset import *
import os

class EncodeUserInputComplexTrial(nn.Module):
    def __init__(self, align_path="TrainedModels/Baseline/150.pth", align_embd=400):
        super().__init__()
        self.align_path = align_path
        self.align_embd = align_embd

        self.clip_encoder = None

        # Loading models
        self.load_models()

        # Preprocessing functions
        self.tokenizer = open_clip.tokenize
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # Resize to DINO's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])
        self.pclip_preprocess = image_transform(
            self.clip_encoder.visual.image_size,  # Correct image size for CLIP
            is_train=False  # Ensures we use inference preprocessing
        )

    def preprocess_text(self, text_prompt):
        max_length = 77
        tokenized_text = self.tokenizer([text_prompt])
        #print(tokenized_text.shape)
        return tokenized_text

    def preprocess_img(self, img_prompt):
        image_tensor = self.transform(img_prompt).unsqueeze(0)
        #print(image_tensor.shape)
        return image_tensor

    def preprocess_pc(self, pc_prompt):
        num_points = 1024
        indices = np.random.choice(pc_prompt.shape[0], num_points, replace=False)
        # Sample the selected points
        sampled_pc = pc_prompt[indices]
        depth_maps = get_all_canonical_dmaps(sampled_pc)  # Returns list of 6 PIL Images
        # Preprocess each depth map (e.g., Resize, ToTensor, Normalize, etc.)
        preprocessed_maps = [self.pclip_preprocess(dmap).unsqueeze(0) for dmap in depth_maps]  # Each is (1, 3, H, W)
        # Stack them into (Views, 3, H, W)
        preprocessed_maps = torch.cat(preprocessed_maps, dim=0)  # Shape: (Views, 3, H, W)
        return preprocessed_maps

    def preprocess_input(self, prompt, modality):
        if modality == "text":
            processed_output = self.preprocess_text(prompt)
        elif modality == "img":
            processed_output = self.preprocess_img(prompt)
        else:
            processed_output = self.preprocess_pc(prompt)
        return processed_output
    
    def load_models(self):
        try:
            self.dinov2_encoder = load_dinov2()
            self.clip_encoder = load_clip()
            self.pclip_encoder = load_point_clip()
            self.align_model = load_complex_alignment(self.align_path, self.align_embd)
            self.dinov2_encoder.eval()
            self.clip_encoder.eval()
            self.pclip_encoder.eval()
            self.align_model.eval()
        except Exception as e:
            print(f'Error in Loading Models {e}')
    
    def get_projection(self, prompt, modality):
        preprocessed_prompt = self.preprocess_input(prompt, modality)

        with torch.no_grad():
            if modality == "text":
                embedding = self.clip_encoder.encode_text(preprocessed_prompt)
                projection = self.align_model.text_proj_head(embedding)
            elif modality == "img":
                embedding = self.dinov2_encoder(preprocessed_prompt)
                projection = self.align_model.img_proj_head(embedding)
            else:
                V, C, H, W = preprocessed_prompt.shape
                # Flatten batch and view dimensions to feed into the encoder
                flattened_dmaps = preprocessed_prompt.view(V, C, H, W)
                with torch.no_grad():
                    # Encode all images at once: shape (B * V, Embed)
                    encoded_views = self.pclip_encoder.encode_image(flattened_dmaps)
                # Reshape back to (B, V, Embed)
                encoded_views = encoded_views.view(V, -1)
                embedding = encoded_views.mean(dim=0).unsqueeze(0)
                projection = self.align_model.pc_proj_head(embedding)

        return projection
    
class AlignedModalityDatasetNoPreprocessing(Dataset):
    """Creates a paired modality dataset that returns text prompt, image and 3D mesh using index

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, image_dir, pc_dir):
        super().__init__()
        # For Text
        self.dataframe = pd.read_csv(dataset_path)

        # For image
        self.image_dir = image_dir
        self.mesh_ids = self.dataframe['fullId'].to_list()

        # For PC
        self.pc_dir = pc_dir

    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Returns:
            idx (int): Index
            tokenized_text (torch.Tensor): Tokenized text for CLIP (B, 77)
            image_tensor (torch.Tensor): preprocessed image for Dinov2 (B, 3, 518, 518)
            point_cloud (torch.Tensor): point cloud of mesh (B, 1024, 3)
        """
        mesh_id = self.dataframe.loc[idx, 'fullId']

        # Choose a random template description
        templates = self.dataframe.loc[idx, ['template1_desc', 'template2_desc', 'template3_desc']]
        text_prompt = random.choice(templates.dropna().tolist())  # Drop NaN values safely

        # Get image views        
        image_views_dir = os.path.join(self.image_dir, mesh_id)
        image_views = [os.path.join(image_views_dir, f) for f in os.listdir(image_views_dir) if os.path.isfile(os.path.join(image_views_dir, f))]
        
        # If no image views
        if not image_views:
            print(f"No views for for {image_views_dir} returning empty tensors")
            return idx, mesh_id, torch.zeros((3, 518, 518))
        
        # Select one view from all
        image_path = random.choice(image_views)
        #image = Image.open(image_path).convert('RGB')

        # Load point cloud
        point_cloud = np.load(os.path.join(self.pc_dir, f"{mesh_id}.npy"))
        
        return idx, text_prompt, image_path, torch.from_numpy(point_cloud)
    
def get_aligned_mesh_ids_from_projection(projection, input_modality, output_modality, cmr):
    idx, mesh_ids, arrays = cmr.retrieve(projection, input_modality, output_modality, top_k=5)

    return idx

def get_retrieved_categories(text_proj, img_proj, pc_proj, act_cat, ca_tt, ca_ti, ca_tp, ca_ii, ca_ip, ca_pp):
    text_text = get_aligned_mesh_ids_from_projection(text_proj, 'text', 'text', cmr)
    text_text = dataframe.iloc[text_text]['category'].to_list()

    text_img = get_aligned_mesh_ids_from_projection(text_proj, 'text', 'img', cmr)
    text_img = dataframe.iloc[text_img]['category'].to_list()

    text_pc = get_aligned_mesh_ids_from_projection(text_proj, 'text', 'pc', cmr)
    text_pc = dataframe.iloc[text_pc]['category'].to_list()

    img_img = get_aligned_mesh_ids_from_projection(text_proj, 'img', 'img', cmr)
    img_img = dataframe.iloc[img_img]['category'].to_list()

    img_pc = get_aligned_mesh_ids_from_projection(text_proj, 'img', 'pc', cmr)
    img_pc = dataframe.iloc[img_pc]['category'].to_list()

    pc_pc = get_aligned_mesh_ids_from_projection(text_proj, 'pc', 'pc', cmr)
    pc_pc = dataframe.iloc[pc_pc]['category'].to_list()

    ca_tt = update_category_accuracies(act_cat, text_text, ca_tt)
    ca_ti = update_category_accuracies(act_cat, text_img, ca_ti)
    ca_tp = update_category_accuracies(act_cat, text_pc, ca_tp)
    ca_ii = update_category_accuracies(act_cat, img_img, ca_ii)
    ca_ip = update_category_accuracies(act_cat, img_pc, ca_ip)
    ca_pp = update_category_accuracies(act_cat, pc_pc, ca_pp)
    return ca_tt, ca_ti, ca_tp, ca_ii, ca_ip, ca_pp

def calculate_accuracy(category, retrieved_cats):
    count = retrieved_cats.count(category)
    acc = count/len(retrieved_cats)
    return acc

def update_category_accuracies(actual_category, retrieved_categories, category_accuracies):
    # Calculate the accuracy for this sample
    accuracy = calculate_accuracy(actual_category, retrieved_categories)
    
    # Check if the category is already in the dictionary
    if actual_category in category_accuracies:
        # If it exists, extend the list of accuracies
        category_accuracies[actual_category].append(accuracy)
    else:
        # If it doesn't exist, add a new entry with the current accuracy in a list
        category_accuracies[actual_category] = [accuracy]
    
    return category_accuracies

def calculate_mean_accuracies(category_accuracies):
    # Iterate over each category and calculate the mean accuracy
    mean_accuracies = {category: np.mean(accuracies) for category, accuracies in category_accuracies.items()}
    return mean_accuracies

def write_to_csv(cat_acc, path):
    df = pd.DataFrame(list(cat_acc.items()), columns=['Category', 'Accuracy'])

    # Write the DataFrame to a CSV file
    df.to_csv(path, index=False)

    print("CSV file has been created.")

if __name__ == "__main__":
    align_path = "TrainedModels/ALIGN/final_template_30cat_fixed/100.pth" # "TrainedModels/ALIGN/final_template_30cat_fixed/100.pth" #"TrainedModels/ALIGN/subset_200_direct_new_loss_fixed_all/140.pth"
    dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat_test.csv" #"Data/ShapeNetSem/Datasets/final_template_30cat.csv" # "Data/ShapeNetSem/Datasets/subset_template_200.csv"
    img_dir = "Data/ShapeNetSem/Images/final_template_30cat/" # "Data/ShapeNetSem/Images/subset_200/"
    pc_dir = "Data/ProcessedData/final_template_30cat_pc/" # "Data/ProcessedData/PointClouds/"
    mesh_dir = "Data/ShapeNetSem/Files/models-OBJ/models/"
    embed_path = f"Embeddings/ALIGN/final_template_30cat_test.pt" #f"Embeddings/ALIGN/final_template_30cat_fixed.pt" # f"Embeddings/ALIGN/subset_template_new_loss_fixed_all.pt" # f"Embeddings/ALIGN/subset_template_200.pt"
    save_path = "Accuracy/final_template_30cat_test/"
    os.makedirs(save_path, exist_ok=True)
    align_embd = 400

    encoder = EncodeUserInputComplexTrial(align_path=align_path, align_embd=align_embd)
    cmr = CrossModalRetrival(dataset_path, embed_path)
    dataframe = pd.read_csv(dataset_path)

    dataset = AlignedModalityDatasetNoPreprocessing(dataset_path, img_dir, pc_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    ca_tt, ca_ti, ca_tp, ca_ii, ca_ip, ca_pp = {}, {}, {}, {}, {}, {}
    for i, batch in enumerate(dataloader):
        idx, text_prompt, image_path, pc = batch
        actual_category = dataframe.iloc[idx]['category'].iloc[0]
        #print(idx, text_prompt, image_path, pc.shape)
        image = Image.open(image_path[0]).convert('RGB')
        text_proj = encoder.get_projection(text_prompt[0], 'text')
        img_proj = encoder.get_projection(image, 'img')
        pc_proj = encoder.get_projection(pc[0], 'pc')
        ca_tt, ca_ti, ca_tp, ca_ii, ca_ip, ca_pp = get_retrieved_categories(text_proj, img_proj, pc_proj, actual_category, ca_tt, ca_ti, ca_tp, ca_ii, ca_ip, ca_pp)

    print("Accuracy for all samples done")

    ca_tt = calculate_mean_accuracies(ca_tt)
    ca_ti = calculate_mean_accuracies(ca_ti)
    ca_tp = calculate_mean_accuracies(ca_tp)
    ca_ii = calculate_mean_accuracies(ca_ii)
    ca_ip = calculate_mean_accuracies(ca_ip)
    ca_pp = calculate_mean_accuracies(ca_pp)

    write_to_csv(ca_tt, os.path.join(save_path, "text_text.csv"))
    write_to_csv(ca_ti, os.path.join(save_path, "text_img.csv"))
    write_to_csv(ca_tp, os.path.join(save_path, "text_pc.csv"))
    write_to_csv(ca_ii, os.path.join(save_path, "img_img.csv"))
    write_to_csv(ca_ip, os.path.join(save_path, "img_pc.csv"))
    write_to_csv(ca_pp, os.path.join(save_path, "pc_pc.csv"))
    print("Completed Writing")