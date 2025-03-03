import torch
import torch.nn.functional as F
import torch.nn as nn

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
