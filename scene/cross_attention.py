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
    Query-based attention을 사용하여 이웃 features를 aggregate합니다.
    각 가우시안의 feature를 query로 사용하고, neighbor features와의 유사도만 계산합니다.
    (N, k, k) 대신 (N, k) 크기의 attention만 계산하여 메모리를 크게 절약합니다.
    """
    def __init__(self, feature_dim=128):
        super(NeighborSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.scale = feature_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_linear = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query_features, neighbor_features):
        """
        Args:
            query_features: (N, feature_dim) 각 가우시안의 feature (query로 사용)
            neighbor_features: (N, k, feature_dim) 각 가우시안의 k개 이웃 features
        
        Returns:
            aggregated: (N, feature_dim) attention으로 aggregate된 feature
        """
        N, k, d = neighbor_features.shape
        residual = neighbor_features.mean(dim=1)  # (N, d) - mean pooling을 residual로 사용
        
        # Query: 각 가우시안의 feature
        Q = self.q_linear(query_features)  # (N, d)
        
        # Key, Value: neighbor features
        K = self.k_linear(neighbor_features)  # (N, k, d)
        V = self.v_linear(neighbor_features)  # (N, k, d)
        
        # Attention scores: query와 각 neighbor의 유사도만 계산 (N, k)
        # Q: (N, 1, d), K: (N, k, d) -> (N, 1, k)
        attention_scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)) * self.scale  # (N, 1, k)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, 1, k)
        
        # Weighted sum: (N, 1, k) x (N, k, d) -> (N, 1, d) -> (N, d)
        attended = torch.bmm(attention_weights, V).squeeze(1)  # (N, d)
        
        # Output projection
        output = self.out_linear(attended)  # (N, d)
        
        # Residual connection (mean pooling 결과와 더하기)
        output = output + residual  # (N, d)
        
        # Layer normalization
        output = self.norm(output)  # (N, d)
        
        return output


