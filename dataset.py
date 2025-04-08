import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import random
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer
import torchvision.transforms as transforms
import open_clip
import pandas as pd

class CLIPTextModalityDataset(Dataset):
    """Creates text modality dataset for ShapeNetSem with CLIP Tokenizer"""

    def __init__(self, dataset_path, max_length=77):
        super().__init__()
        self.dataframe = pd.read_csv(dataset_path)
        self.tokenizer = open_clip.tokenize  # Use OpenCLIP's built-in tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Returns:
            idx (int): Index
            tokenized_text (torch.Tensor): Tokenized text for CLIP
            text_prompt (str): The actual text prompt
        """
        mesh_id = self.dataframe.loc[idx, 'fullId']

        # Choose a random template description
        templates = self.dataframe.loc[idx, ['template1_desc', 'template2_desc', 'template3_desc']]
        text_prompt = random.choice(templates.dropna().tolist())  # Drop NaN values safely

        # Tokenize using OpenCLIP tokenizer (returns a tensor)
        tokenized_text = self.tokenizer([text_prompt])  # Shape: [1, 77]

        return idx, mesh_id, tokenized_text.squeeze(0)

class Dinov2ImageModalityDataset(Dataset):
    """Creates image modality dataset for ShapeNetSem with Dinov2 image preprocessing

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.dataframe = pd.read_csv(dataset_path)
        self.mesh_ids = self.dataframe['fullId'].to_list()
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((518, 518)),  # Resize to DINO's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])
    def __len__(self):
        return len(self.mesh_ids)
    
    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            image_tensor (torch.tensor): preprocessed image tensor using CLIP image preprocessor
            image_path (string): image path
        """
        # mesh_id = self.dataframe.loc[idx, 'fullId']
        mesh_id = self.mesh_ids[idx]
        # image_views_dir = self.meshes[idx]
        image_views_dir = os.path.join(self.image_dir, mesh_id)
        image_views = [os.path.join(image_views_dir, f) for f in os.listdir(image_views_dir) if os.path.isfile(os.path.join(image_views_dir, f))]
        
        if not image_views:
            print(f"No views for for {image_views_dir} returning empty tensors")
            return idx, mesh_id, torch.zeros((3, 518, 518))
        
        image_path = random.choice(image_views)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image) 

        return idx, mesh_id, image_tensor
    
class CLIPImageModalityDataset(Dataset):
    """Creates image modality dataset for ShapeNetSem with CLIP image preprocessing

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, image_dir, processor=None):
        super().__init__()
        self.image_dir = image_dir
        self.meshes = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, f))]
        self.processor = processor if processor else CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def __len__(self):
        return len(self.meshes)
    
    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            image_tensor (torch.tensor): preprocessed image tensor using CLIP image preprocessor
            image_path (string): image path
        """
        image_views_dir = self.meshes[idx]
        image_views = [os.path.join(image_views_dir, f) for f in os.listdir(image_views_dir) if os.path.isfile(os.path.join(image_views_dir, f))]
        image_path = random.choice(image_views.tolist())
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0) # [B, C, H, W]

        return image_tensor, image_path

class PCModalityDataset(Dataset):
    """Creates point cloud modality dataset for ShapeNetSem"""

    def __init__(self, dataset_path, pc_dir, num_points=1024):
        super().__init__()
        self.dataframe = pd.read_csv(dataset_path)
        self.mesh_ids = self.dataframe['fullId'].to_list()
        self.pc_dir = pc_dir
        self.num_points = num_points
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Returns:
            idx (int): Index
            tokenized_text (torch.Tensor): Tokenized text for CLIP
            text_prompt (str): The actual text prompt
        """
        mesh_id = self.mesh_ids[idx]
        point_cloud = np.load(os.path.join(self.pc_dir, f"{mesh_id}.npy"))
        if point_cloud.shape[0] < self.num_points:
            raise ValueError("Point cloud has fewer points than the requested sample size.")
    
        indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
        
        return idx, mesh_id, torch.from_numpy(point_cloud[indices])
    
class AlignedModalityDataset(Dataset):
    """Creates a paired modality dataset that returns text prompt, image and 3D mesh using index

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, image_dir, pc_dir, max_length=77, transform=None, num_points=1024):
        super().__init__()
        # For Text
        self.dataframe = pd.read_csv(dataset_path)
        self.tokenizer = open_clip.tokenize  # Use OpenCLIP's built-in tokenizer
        self.max_length = max_length

        # For image
        self.image_dir = image_dir
        self.mesh_ids = self.dataframe['fullId'].to_list()
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((518, 518)),  # Resize to DINO's expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINOv2 normalization
        ])

        # For PC
        self.pc_dir = pc_dir
        self.num_points = num_points

    
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

        # Tokenize using OpenCLIP tokenizer (returns a tensor)
        tokenized_text = self.tokenizer([text_prompt])  # Shape: [1, 77]

        # Get image views        
        image_views_dir = os.path.join(self.image_dir, mesh_id)
        image_views = [os.path.join(image_views_dir, f) for f in os.listdir(image_views_dir) if os.path.isfile(os.path.join(image_views_dir, f))]
        
        # If no image views
        if not image_views:
            print(f"No views for for {image_views_dir} returning empty tensors")
            return idx, mesh_id, torch.zeros((3, 518, 518))
        
        # Select one view from all
        image_path = random.choice(image_views)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image) 

        # Load point cloud
        point_cloud = np.load(os.path.join(self.pc_dir, f"{mesh_id}.npy"))
        if point_cloud.shape[0] < self.num_points:
            raise ValueError("Point cloud has fewer points than the requested sample size.")

        # Randomly sample required points
        indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
        
        return idx, tokenized_text.squeeze(0), image_tensor, torch.from_numpy(point_cloud[indices])
    
class RAGDataset(Dataset):
    def __init__(self, data_dict, dataset_path, pc_dir, num_points=1024):
        """
        Initialize the dataset with data from the dictionary.
        
        :param data_dict: Dictionary containing the projections and other data
        """
        self.data_dict = data_dict
        self.all_idx = data_dict['index']  # This is a tensor
        self.text_proj = data_dict["text_proj"]
        self.img_proj = data_dict["img_proj"]
        self.pc_proj = data_dict["pc_proj"]

        self.pc_dir = pc_dir
        self.dataframe = pd.read_csv(dataset_path)
        self.num_points = num_points

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.all_idx)

    def __getitem__(self, idx):
        """
        Fetches a sample at a specific index.
        
        :param idx: Sample index (provided when accessing the dataset)
        :return: A dictionary with the selected data (text, image, pc projection)
        """

        # Use the idx to retrieve data from the tensor of indices
        selected_idx = self.all_idx[idx]  # This is still a tensor, but works for indexing
        # print(idx, selected_idx)
        # Get corresponding projections and category using the selected idx
        text_projection = self.text_proj[selected_idx]
        img_projection = self.img_proj[selected_idx]
        pc_projection = self.pc_proj[selected_idx]
        category1 = self.data_dict['category'][selected_idx]
        #print(category1)
        # Getting true point cloud
        mesh_id = self.dataframe.loc[selected_idx, 'fullId']
        category = self.dataframe.loc[selected_idx, 'category']
        # print(category, category1)
        pc_path = f'{self.pc_dir}{mesh_id}.npy'
        true_pc = np.load(pc_path)
        indices = np.random.choice(true_pc.shape[0], self.num_points, replace=False)
        # Sample the selected points
        true_pc = torch.from_numpy(true_pc[indices])
        # Returning the selected data as a dictionary
        return {
            "text_proj": text_projection,
            "img_proj": img_projection,
            "pc_proj": pc_projection,
            "index": selected_idx,
            "category": category,
            "target_pc": true_pc
        }

class ChoiceEmbeddingDataset(Dataset):
    """Creates a paired modality dataset that returns text image and pc embedding (from pretrained encoders)

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, embd_dir):
        super().__init__()
        # For Text
        self.dataframe = pd.read_csv(dataset_path)
        self.embed_dir = embd_dir
        """ 
        data_dict = {
            "mesh_id": all_text_emb,
            "text_emb": [3, 768],
            "img_emb": [4, 384],
            "pc_emb": [8, 768],
        }
        """
    
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

        dict_path = os.path.join(self.embed_dir, f'{mesh_id}.pt')
        data_dict = torch.load(dict_path)
        # Retrieve the corresponding embedding using the index
        text_embedding = data_dict['text_emb']
        img_embedding = data_dict['img_emb']
        pc_embedding = data_dict['pc_emb']

        text_index = random.randint(0, text_embedding.shape[0] - 1)
        img_index = random.randint(0, img_embedding.shape[0] - 1)
        pc_index = random.randint(0, pc_embedding.shape[0] - 1)

        # Now return the embedding (and other data if needed)
        return idx, text_embedding[text_index], img_embedding[img_index], pc_embedding[pc_index]
    
class CategoryChoiceEmbeddingDataset(Dataset):
    """Creates a paired modality dataset that returns text image and pc embedding (from pretrained encoders)

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, embd_dir, label_to_idx):
        super().__init__()
        # For Text
        self.dataframe = pd.read_csv(dataset_path)
        self.embed_dir = embd_dir
        """ 
        data_dict = {
            "mesh_id": all_text_emb,
            "text_emb": [3, 768],
            "img_emb": [4, 384],
            "pc_emb": [8, 768],
        }
        """
        self.label_to_idx = label_to_idx
    
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

        category = self.dataframe.loc[idx, 'category']
        if ',' in category:
            categories = category.split(',')
            category = [category for category in categories if '_' not in category][0]

        category_label = self.label_to_idx[category]
        category_tensor = torch.tensor(category_label, dtype=torch.long)

        dict_path = os.path.join(self.embed_dir, f'{mesh_id}.pt')
        data_dict = torch.load(dict_path)
        # Retrieve the corresponding embedding using the index
        text_embedding = data_dict['text_emb']
        img_embedding = data_dict['img_emb']
        pc_embedding = data_dict['pc_emb']

        text_index = random.randint(0, text_embedding.shape[0] - 1)
        img_index = random.randint(0, img_embedding.shape[0] - 1)
        pc_index = random.randint(0, pc_embedding.shape[0] - 1)

        # Now return the embedding (and other data if needed)
        return idx, text_embedding[text_index], img_embedding[img_index], pc_embedding[pc_index], category_tensor
    
class CategoryChoiceEmbeddingDatasetAllDmap(Dataset):
    """Creates a paired modality dataset that returns text image and pc embedding (from pretrained encoders)

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, embd_dir, label_to_idx):
        super().__init__()
        # For Text
        self.dataframe = pd.read_csv(dataset_path)
        self.embed_dir = embd_dir
        """ 
        data_dict = {
            "mesh_id": all_text_emb,
            "text_emb": [3, 768],
            "img_emb": [4, 384],
            "pc_emb": [8, 768],
        }
        """
        self.label_to_idx = label_to_idx
    
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

        category = self.dataframe.loc[idx, 'category']
        if ',' in category:
            categories = category.split(',')
            category = [category for category in categories if '_' not in category][0]

        category_label = self.label_to_idx[category]
        category_tensor = torch.tensor(category_label, dtype=torch.long)

        dict_path = os.path.join(self.embed_dir, f'{mesh_id}.pt')
        data_dict = torch.load(dict_path)
        # Retrieve the corresponding embedding using the index
        text_embedding = data_dict['text_emb']
        img_embedding = data_dict['img_emb']
        pc_embedding = data_dict['pc_emb']

        text_index = random.randint(0, text_embedding.shape[0] - 1)
        img_index = random.randint(0, img_embedding.shape[0] - 1)
        pc_index = random.randint(0, pc_embedding.shape[0] - 1)

        # Now return the embedding (and other data if needed)
        return idx, text_embedding[text_index], img_embedding[img_index], pc_embedding, category_tensor