from alignment import *
import time
from torch.utils.data import DataLoader, Dataset
import os

class AlignedModalityDatasetForEmbedding(Dataset):
    """Creates a paired modality dataset that returns text prompt, image and 3D mesh using index

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, image_dir, depth_dir, max_length=77, transform=None):
        super().__init__()
        # For Text
        self.dataframe = pd.read_csv(dataset_path)
        self.tokenizer = open_clip.tokenize  # Use OpenCLIP's built-in tokenizer
        self.max_length = max_length

        # For image
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((518, 518)),  # Resize to DINO's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])

        # For PC
        self.depth_dir = depth_dir
        self.preprocess_depth = image_transform(
            (224, 224),  # Correct image size for CLIP
            is_train=False  # Ensures we use inference preprocessing
        )
    
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
        text_prompts = templates.dropna().tolist() # Drop NaN values safely
        # Tokenize using OpenCLIP tokenizer (returns a tensor)
        tokenized_text = torch.stack([self.tokenizer(text_prompt).squeeze(0) for text_prompt in text_prompts]) # Shape: [1, 77]

        # Get image views        
        image_views_dir = os.path.join(self.image_dir, mesh_id)
        image_views = [os.path.join(image_views_dir, f) for f in os.listdir(image_views_dir) if os.path.isfile(os.path.join(image_views_dir, f))]

        # Select one view from all
        # image_path = random.choice(image_views)
        images = [Image.open(image_path).convert('RGB') for image_path in image_views]
        images_tensor = torch.stack([self.transform(image) for image in images])

        # Get image views        
        depth_views_dir = os.path.join(self.depth_dir, mesh_id)
        depth_views = [os.path.join(depth_views_dir, f) for f in os.listdir(depth_views_dir) if os.path.isfile(os.path.join(depth_views_dir, f))]

        # Select one view from all
        # image_path = random.choice(image_views)
        depth_maps = [Image.open(depth_path).convert('RGB') for depth_path in depth_views]
        dmaps_tensor = torch.stack([self.preprocess_depth(depth_map) for depth_map in depth_maps])

        return idx, mesh_id, tokenized_text, images_tensor, dmaps_tensor
    
if __name__ == "__main__":
    try:
        dinov2_encoder = load_dinov2()
        clip_encoder = load_clip()
        dinov2_encoder.eval()
        clip_encoder.eval()
        print('All Models loaded succesfully and set to eval mode')
    except:
        print('Error in Loading Models')

    dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat.csv" #"Data/ShapeNetSem/Datasets/subset_template_200.csv"
    image_dir = "Data/ShapeNetSem/Images/final_template_30cat" #"Data/ShapeNetSem/Images/subset_200"
    depth_dir = "Data/ProcessedData/final_template_30cat_pc_dmaps_fixed" #"Data/ProcessedData/subset_template_200_dmaps_fixed"
    embd_dir = "Embeddings/PRETRAINED/final_template_30cat_fixed/" #"Embeddings/PRETRAINED/subset_template_200_fixed/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = AlignedModalityDatasetForEmbedding(dataset_path, image_dir, depth_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Dataset path: ", dataset_path)
    print("Image path: ", image_dir)
    print("Depth path: ", depth_dir)
    print("Save embed in path: ", embd_dir)

    os.makedirs(embd_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, mesh_id, text, image, dmap = batch
            mesh_id = mesh_id[0]
            text = text.squeeze(0)
            image = image.squeeze(0)
            dmap = dmap.squeeze(0)

            # torch.Size([1, 3, 77]) torch.Size([1, 4, 3, 518, 518]) torch.Size([1, 8, 3, 224, 224]
            #print(idx, mesh_id, text.shape, image.shape, dmap.shape)
            with torch.no_grad():
                text_emb = clip_encoder.encode_text(text) # (B, 768)
                img_emb = dinov2_encoder(image) # (B, 384)
                pc_emb = clip_encoder.encode_image(dmap) # (B, 768)
                #print(readable_time(a, b))
            save_path = os.path.join(embd_dir, f"{mesh_id}.pt")
            #print(idx, mesh_id, text_emb.shape, img_emb.shape, pc_emb.shape, save_path)
            data_dict = {
                "mesh_id":mesh_id,
                "text_emb":text_emb,
                "img_emb":img_emb,
                "pc_emb":pc_emb
            }
            torch.save(data_dict, save_path)
            print(f"{idx} processed and saved {mesh_id}")