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


# --- Helper: Quaternion -> Rotation Matrix ---
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class AttributeEncoder(nn.Module):
    """
    Attribute Encoder: 가우시안의 모든 속성을 인코딩
    Input: xyz, scale, rot, opacity, sh
    Output: 128 channels
    * Modified: Rotation(4) -> Covariance(6)
    """
    def __init__(self, input_dim=118, hidden_dim=256, out_dim=128): # input_dim: 116 -> 118
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
        B, _ = xyz.shape
        device = xyz.device
        bands = (2.0 ** torch.arange(self.L, device=device)).view(1, -1) 
        x = xyz.unsqueeze(-1) * bands.unsqueeze(1) * math.pi
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        pe = torch.cat([sin_x, cos_x], dim=-1).view(B, -1)
        return pe
    
    def compute_covariance_features(self, scale, rotation):
        """Scale과 Rotation을 결합하여 6D Covariance Feature 생성"""
        # 1. Scaling Matrix S
        S = torch.zeros((scale.shape[0], 3, 3), device=scale.device)
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 1]
        S[:, 2, 2] = scale[:, 2]
        
        # 2. Rotation Matrix R
        R = build_rotation(rotation)
        
        # 3. Covariance Matrix Sigma = R S S^T R^T = (RS)(RS)^T
        M = torch.bmm(R, S)
        Cov = torch.bmm(M, M.transpose(1, 2)) # [N, 3, 3]
        
        # 4. Extract Upper Triangle (6 elements)
        # xx, xy, xz, yy, yz, zz
        cov_features = torch.stack([
            Cov[:, 0, 0], Cov[:, 0, 1], Cov[:, 0, 2],
            Cov[:, 1, 1], Cov[:, 1, 2],
            Cov[:, 2, 2]
        ], dim=1) # [N, 6]
        
        # 5. Normalization (중요!)
        # Covariance 값은 Scale의 제곱에 비례하므로 값이 매우 커질 수 있음.
        # 학습 안정을 위해 tanh로 범위를 -1~1로 압축
        return torch.tanh(cov_features)
    
    def forward(self, xyz, scale, rotation, opacity, sh_features):
        """
        xyz: [N, 3]
        scale: [N, 3]
        rotation: [N, 4] -> 사용 안 함 (Covariance 계산에만 사용)
        opacity: [N, 1]
        sh_features: [N, 16, 3]
        """
        # Positional encoding for xyz
        xyz_normalized = torch.tanh(xyz / 20.0)
        pe = self.positional_encoding(xyz_normalized)  # [N, 60]
        scale_log = torch.log(scale + 1e-9) # [N, 3]
        
        # [수정됨] Rotation 대신 Covariance 계산
        cov_features = self.compute_covariance_features(scale, rotation) # [N, 6]
        
        # Flatten sh_features: [N, 16, 3] -> [N, 48]
        sh_flat = sh_features.view(sh_features.shape[0], -1)  # [N, 48]
        
        # Concatenate all features
        # PE(60) + Scale(3) + Covariance(6) + Opacity(1) + SH(48) = 118
        features = torch.cat([pe, scale_log, cov_features, opacity, sh_flat], dim=1)  # [N, 118]
        
        # Normalize and encode
        features = self.input_norm(features)
        encoded = self.mlp(features)  # [N, 128]
        
        return encoded


