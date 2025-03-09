import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Computes NT-Xent loss for a batch of paired embeddings.
        - z1: First modality (e.g., text)
        - z2: Second modality (e.g., pointcloud)
        """
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        #print(z1.shape, z2.shape)
        # Compute cosine similarity matrix
        similarity = torch.mm(z1, z2.T) / self.temperature  # Shape: [batch_size, batch_size]
        #print(similarity.shape)
        # Labels should be in range [0, batch_size-1]
        labels = torch.arange(batch_size, device=z1.device)  # Correct labels
        #print(labels)
        # Compute contrastive loss using cross-entropy
        loss = F.cross_entropy(similarity, labels)

        # Debugging output
        """
        with torch.no_grad():
            print(f"Mean Similarity: {similarity.mean().item():.4f}, Loss: {loss.item():.4f}")
        """

        return loss



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.ReLU(),
            nn.Linear(2*in_dim, out_dim)
        )
    
    def forward(self, x):
        out = self.net(x)
        return out

class AlignEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.text_proj_head = MLP(768, embed_dim)
        self.img_proj_head = MLP(384, embed_dim)
        self.pc_proj_head = MLP(768, embed_dim)
    
    def forward(self, text, img, pc):
        text_proj = self.text_proj_head(text)
        img_proj = self.img_proj_head(img)
        pc_proj = self.pc_proj_head(pc)

        return text_proj, img_proj, pc_proj

def load_alignment(checkpoint_path, align_embd):
    align_model = AlignEncoder(align_embd)  # Ensure the architecture matches
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU
    # Apply the weights to the model
    align_model.load_state_dict(state_dict)
    # Set to evaluation mode (if needed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_model.to(device)
    return align_model
    
class CrossModalRetrival:
    def __init__(self, dataset_path, embedding_path, device='cpu'):
        self.device = device
        self.dataset_path = dataset_path
        self.embedding_path = embedding_path
        self.embeddings = {}
        self.index = None
        self.categories = None
        self.load_embeddings()

    def load_embeddings(self):
        self.dataframe = pd.read_csv(self.dataset_path)
        data = torch.load(self.embedding_path)
        self.embeddings["text"] = data["text_proj"].numpy()
        self.embeddings["img"] = data["img_proj"].numpy()
        self.embeddings["pc"] = data["pc_proj"].numpy()
        self.index = data["index"]
        self.categories = data["category"]

    def retrieve(self, query, query_modality, target_modality, top_k=5):
        query_proj = F.normalize(query, dim=-1).cpu().numpy()
        similarities = np.dot(self.embeddings[target_modality], query_proj.T).squeeze()
        indices = np.argsort(-similarities)[:top_k]  # Get top-k indices
        
        # Retrieve corresponding samples from target modality
        retrieved_idx = [self.index[i].item() for i in indices]
        mesh_ids = self.dataframe.iloc[retrieved_idx]['fullId']
        results = self.embeddings[target_modality][indices]
        return retrieved_idx, mesh_ids, results

class EncodeUserInput(nn.Module):
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
        depth_map = self.pclip_preprocess(point_cloud_to_depth_map(sampled_pc)).unsqueeze(0)  # Add batch dimension
        return depth_map

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
            self.align_model = load_alignment(self.align_path, self.align_embd)
            self.dinov2_encoder.eval()
            self.clip_encoder.eval()
            self.pclip_encoder.eval()
            self.align_model.eval()
        except:
            print('Error in Loading Models')
    
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
                embedding = self.pclip_encoder.encode_image(preprocessed_prompt)
                projection = self.align_model.pc_proj_head(embedding)

        return projection

def get_aligned_output(dataset_path, img_dir, pc_dir, prompt, input_modality, output_modality, encoder, cmr):
    df = pd.read_csv(dataset_path)
    projection = encoder.get_projection(prompt, input_modality)
    idx, mesh_ids, arrays = cmr.retrieve(projection, input_modality, output_modality, top_k=5)

    if output_modality == "text":
        output = df.iloc[idx[2]]['template1_desc']
    elif output_modality == "img":
        output = f'{img_dir}{mesh_ids[idx[1]]}/view0.png'
    else:
        output = f'{pc_dir}{mesh_ids[idx[1]]}.obj'
    
    return output