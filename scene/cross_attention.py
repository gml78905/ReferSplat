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
        self.fc3 = nn.Linear(64, out_dim)
        
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

class MLP_attribute_features(nn.Module):
    """MLP to create attribute_features from cov_embed and dc_embed: 64 -> 128"""
    def __init__(self, in_dim=64, out_dim=128):  # 32 + 32 = 64
        super(MLP_attribute_features, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_position_feature(nn.Module):
    """MLP to create position_feature from xyz_embed: 32 -> 128"""
    def __init__(self, in_dim=32, out_dim=128):
        super(MLP_position_feature, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NeighborSelfAttention(nn.Module):
    """
    Self-attention을 사용하여 이웃 features를 aggregate합니다.
    각 가우시안의 k개 이웃들 간의 attention을 계산하여 가중치를 학습합니다.
    """
    def __init__(self, feature_dim=128, num_heads=1):
        super(NeighborSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_linear = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, neighbor_features):
        """
        Args:
            neighbor_features: (N, k, feature_dim) 각 가우시안의 k개 이웃 features
        
        Returns:
            aggregated: (N, feature_dim) self-attention으로 aggregate된 feature
        """
        N, k, d = neighbor_features.shape
        residual = neighbor_features.mean(dim=1, keepdim=True)  # (N, 1, d) - mean pooling을 residual로 사용
        
        # Query, Key, Value 생성
        Q = self.q_linear(neighbor_features)  # (N, k, d)
        K = self.k_linear(neighbor_features)  # (N, k, d)
        V = self.v_linear(neighbor_features)  # (N, k, d)
        
        # Multi-head attention (단순화를 위해 num_heads=1로 가정)
        # Attention scores 계산
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (N, k, k)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, k, k)
        
        # Weighted sum
        attended = torch.bmm(attention_weights, V)  # (N, k, d)
        
        # Aggregate: 각 가우시안의 k개 이웃을 평균 (또는 다른 방식)
        aggregated = attended.mean(dim=1)  # (N, d)
        
        # Output projection
        output = self.out_linear(aggregated)  # (N, d)
        
        # Residual connection (mean pooling 결과와 더하기)
        output = output + residual.squeeze(1)  # (N, d)
        
        # Layer normalization
        output = self.norm(output)  # (N, d)
        
        return output


