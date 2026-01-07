import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, in_dim=3, out_dim=128):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16 ,64)
        self.fc3 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x    


class AttributeEncoder(nn.Module):
    """
    Attribute Encoder: 가우시안의 모든 속성을 인코딩
    Input: xyz, scale, rot, opacity, sh
    Output: 128 channels
    """
    def __init__(self, input_dim=116, hidden_dim=256, out_dim=128):
        super().__init__()
        # Positional Encoding 설정
        self.L = 10
        self.pe_channels = 3 * 2 * self.L  # 60
        self.input_norm = nn.LayerNorm(input_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
            # 필요시 LayerNorm 추가 가능
        )
    
    def positional_encoding(self, xyz):
        """Vectorized Positional Encoding"""
        # xyz: [N, 3]
        B, _ = xyz.shape
        device = xyz.device
        
        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        # [L] -> [1, L]
        bands = (2.0 ** torch.arange(self.L, device=device)).view(1, -1) 
        
        # [N, 3, 1] * [1, 1, L] -> [N, 3, L]
        # coord * freq * pi
        x = xyz.unsqueeze(-1) * bands.unsqueeze(1) * math.pi
        
        # sin, cos -> [N, 3, L, 2] -> flatten -> [N, 60]
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        pe = torch.cat([sin_x, cos_x], dim=-1).view(B, -1)
        return pe
    
    def forward(self, xyz, scale, rotation, opacity, sh_features):
        """
        xyz: [N, 3]
        scale: [N, 3]
        rotation: [N, 4]
        opacity: [N, 1]
        sh_features: [N, 16, 3] for degree 3
        """
        # Positional encoding for xyz
        pe = self.positional_encoding(xyz)  # [N, 60]
        
        # Flatten sh_features: [N, 16, 3] -> [N, 48]
        sh_flat = sh_features.view(sh_features.shape[0], -1)  # [N, 48]
        
        # Concatenate all features: 60 (pe) + 3 (scale) + 4 (rotation) + 1 (opacity) + 48 (sh) = 116
        features = torch.cat([pe, scale, rotation, opacity, sh_flat], dim=1)  # [N, 116]
        
        # Normalize and encode
        features = self.input_norm(features)
        encoded = self.mlp(features)  # [N, 128]
        
        return encoded


