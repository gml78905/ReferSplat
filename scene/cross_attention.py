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
    Dual-Stream Attribute Encoder (No KNN)
    - Geometry Stream: IPE + Explicit Shape Metrics (Westin's) + Covariance
    - Appearance Stream: Disentangled SH (DC/Rest) + Opacity
    """
    def __init__(self, out_dim=128):
        super().__init__()

        # -----------------------------------------------------------
        # 1. Geometry Stream
        # -----------------------------------------------------------
        self.L = 10
        self.geo_input_dim = 60 + 3 + 4 + 4 + 1
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.geo_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # 2. Appearance Stream
        # -----------------------------------------------------------
        self.sh_compressor = nn.Linear(45, 16)
        self.app_input_dim = 3 + 16 + 1
        self.app_mlp = nn.Sequential(
            nn.Linear(self.app_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # 3. Final Projection
        # -----------------------------------------------------------
        self.final_head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def integrated_positional_encoding(self, xyz, scale):
        B, _ = xyz.shape
        device = xyz.device

        freqs = 2.0 ** torch.arange(self.L, device=device).view(1, -1)
        args = xyz.unsqueeze(-1) * freqs.unsqueeze(1) * math.pi

        scale_input = scale.unsqueeze(-1)
        var = scale_input ** 2
        coeff = (freqs * math.pi) ** 2 * var
        attenuation = torch.exp(-0.5 * coeff)

        sin_x = torch.sin(args) * attenuation
        cos_x = torch.cos(args) * attenuation
        return torch.cat([sin_x, cos_x], dim=-1).view(B, -1)

    def compute_westin_metrics(self, scale):
        sorted_scale, _ = torch.sort(scale, dim=1, descending=True)
        l1, l2, l3 = sorted_scale[:, 0], sorted_scale[:, 1], sorted_scale[:, 2]
        denom = l1 + 1e-9

        linearity = (l1 - l2) / denom
        planarity = (l2 - l3) / denom
        sphericity = l3 / denom
        anisotropy = (l1 - l3) / denom

        return torch.stack([linearity, planarity, sphericity, anisotropy], dim=1)

    def forward(self, xyz, scale, rotation, opacity, sh_features):
        # Geometry stream
        ipe = self.integrated_positional_encoding(xyz, scale)
        westin = self.compute_westin_metrics(scale)
        scale_log = torch.log(scale + 1e-9)

        geo_in = torch.cat([ipe, scale_log, westin, rotation, opacity], dim=1)
        f_geo = self.geo_mlp(geo_in)

        # Appearance stream
        if sh_features.dim() == 2:
            sh_features = sh_features.view(-1, 16, 3)
        sh_dc = sh_features[:, 0, :]
        sh_rest = sh_features[:, 1:, :].reshape(sh_features.shape[0], -1)
        sh_rest_comp = self.sh_compressor(sh_rest)
        app_in = torch.cat([sh_dc, sh_rest_comp, opacity], dim=1)
        f_app = self.app_mlp(app_in)

        # Fusion
        f_fused = torch.cat([f_geo, f_app], dim=1)
        out = self.final_head(f_fused)
        return out