import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import random
from PIL import Image
from transformers import CLIPImageProcessor
from transformers import CLIPTokenizer
import torchvision.transforms as transforms
import timm 

class TextModalityDataset(Dataset):
    """Creates text modality dataset for ShapeNetSem with CLIP Tokenizer

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, tokenizer=None, max_length=77):
        super().__init__()
        self.dataframe = pd.read_csv(dataset_path)
        self.tokenizer = tokenizer if tokenizer else CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        mesh_id = self.dataframe.loc[idx, 'fullId']

        # Text Modality
        templates = self.dataframe.loc[idx, ['template1_desc','template2_desc','template3_desc']]
        text_prompt = random.choice(templates.tolist())

        tokenized_text = self.tokenizer(
            text_prompt,
            padding="max_length",
            truncation=True,
            max_length = self.max_length,
            return_tensors = "pt"
        )

        return tokenized_text["input_ids"].squeeze(0), tokenized_text["attention_mask"].squeeze(0), text_prompt # as of now only returns path for image and mesh
    
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

class PairedModalityDataset(Dataset):
    """Creates a paired modality dataset that returns text prompt, image and 3D mesh using index

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path, image_dir, mesh_dir):
        super().__init__()
        self.dataframe = pd.read_csv(dataset_path)
        self.image_dir = image_dir
        self.mesh_dir = mesh_dir
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        mesh_id = self.dataframe.loc[idx, 'fullId']

        # Text Modality
        templates = self.dataframe.loc[idx, ['template1_desc','template2_desc','template3_desc']]
        text_prompt = random.choice(templates.tolist())

        # Image Modality
        image_views_dir = os.path.join(self.image_dir, mesh_id)
        image_views = [os.path.join(image_views_dir, f) for f in os.listdir(image_views_dir) if os.path.isfile(os.path.join(image_views_dir, f))]
        image_prompt = random.choice(image_views.tolist())

        # 3D Modality
        mesh_path = os.path.join(self.mesh_dir, mesh_id)

        return text_prompt, image_prompt, mesh_path # as of now only returns path for image and mesh