import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def chamfer_distance(pc1, pc2):
    """
    Chamfer Distance between two point clouds.
    pc1 and pc2 should be of shape (B, N, 3).
    """
    pc1 = pc1.unsqueeze(2)  # Expand to (B, N, 3, 1)
    pc2 = pc2.unsqueeze(1)  # Expand to (B, 1, N, 3)
    
    dist = torch.norm(pc1 - pc2, dim=3)  # Calculate the L2 distance between points
    min_dist1 = dist.min(dim=2)[0]  # Min distance along the second axis
    min_dist2 = dist.min(dim=1)[0]  # Min distance along the first axis
    
    # Return the mean Chamfer Distance
    return min_dist1.mean() + min_dist2.mean()

class HiddenLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        return x
    
class RAGDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers=1, num_points=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Sequential(*[HiddenLayer(hidden_dim) for i in range(num_hidden_layers)])
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.num_points = num_points
        self.relu = nn.ReLU
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.view(-1, self.num_points, 3)
    
class HiddenLayerComplex(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate) 
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x
    
class RAGDecoderComplex(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers=1, num_points=1024, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Sequential(*[HiddenLayerComplex(hidden_dim, dropout_rate) for i in range(num_hidden_layers)])
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.num_points = num_points
        self.relu = nn.ReLU
        self.dropout = nn.Dropout(dropout_rate) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.view(-1, self.num_points, 3)
    
def load_rag(checkpoint_path="TrainedModels/RAG/four_hidden/2000.pth"):
    num_points = 1024  # Number of points in the point cloud
    in_dim = 400 + num_points * 3  # Input dimension (projection + flattened point cloud)
    hidden_dim = 512  # Hidden layer size
    out_dim = num_points * 3
    num_hidden_layers = 4
    rag_decoder = RAGDecoder(in_dim, hidden_dim, out_dim, num_hidden_layers=4)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU
    # Apply the weights to the model
    rag_decoder.load_state_dict(state_dict)
    # Set to evaluation mode (if needed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag_decoder.to(device)
    return rag_decoder