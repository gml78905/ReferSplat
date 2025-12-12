import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # 缩放因子

        
        self.q_linear = nn.Linear(dim, dim)  # Query
        self.k_linear = nn.Linear(dim, dim)  # Key
        self.v_linear = nn.Linear(dim, dim)  # Value
        self.gp_linear = nn.Linear(dim, dim)
        self.kp_linear = nn.Linear(dim, dim)
        self.norm=nn.LayerNorm(dim)
        

    def forward(self,g,g_p,W):
        
        W=W.squeeze(0)
        
        k_p = torch.matmul(F.softmax(torch.matmul(W, g.transpose(-1, -2)), dim=-1), g_p)
        k_p=self.kp_linear(k_p)
        g_p=self.gp_linear(g_p)
        Q = self.q_linear(g)+g_p
        K = self.k_linear(W)+k_p
        V=self.v_linear(W)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # [5000, x]
        output = torch.matmul(attention_weights, V)
        
        output=output+g
        output=self.norm(output)
        
        return output

class MLP1(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
       
        x = self.fc3(x)
        return x  

class MLP2(nn.Module):
    def __init__(self, in_dim=16, out_dim=128):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32 ,64)
        self.fc3 = nn.Linear(64, 128)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x 
    
class MLP3(nn.Module):
    def __init__(self, in_dim=12, out_dim=128):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32 ,64)
        self.fc3 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x    

class MLP_xyz(nn.Module):
    """MLP for xyz position features: 3 -> embedding_dim"""
    def __init__(self, in_dim=3, embedding_dim=32):
        super(MLP_xyz, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, embedding_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_cov(nn.Module):
    """MLP for covariance upper triangle features: 6 -> embedding_dim"""
    def __init__(self, in_dim=6, embedding_dim=32):
        super(MLP_cov, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, embedding_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_dc(nn.Module):
    """MLP for features_dc (color DC): 3 -> embedding_dim"""
    def __init__(self, in_dim=3, embedding_dim=32):
        super(MLP_dc, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, embedding_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_geometry_fusion(nn.Module):
    """Final MLP to fuse individual geometry embeddings: 3*embedding_dim -> 128"""
    def __init__(self, in_dim=96, out_dim=128):  # 3 * 32 = 96
        super(MLP_geometry_fusion, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


