import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class IntrinsicEncoder(nn.Module):
    """
    Intrinsic feature encoder: 9 -> 64 -> 64 -> 16
    Role: Encodes geometric (covariance) and appearance (color) cues into a semantic ID.
    """
    def __init__(self, in_dim=9, hidden_dim=64, out_dim=16):
        super(IntrinsicEncoder, self).__init__()
        
        self.net = nn.Sequential(
            # [중요 1] Input Normalization
            # Covariance(-3~6)와 RGB(0~1)의 분포 차이를 해소합니다.
            # 이것이 없으면 학습 초기 수렴이 매우 느리거나 불안정합니다.
            nn.LayerNorm(in_dim),
            
            # Layer 1: Expansion & Feature Mixing
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Hidden Layer 분포 안정화
            nn.GELU(),                # ReLU보다 부드러운 활성화 함수 (최신 트렌드)
            
            # [중요 2] Depth 추가 (Reasoning)
            # 물리적 수치를 의미적 특징으로 바꾸려면 비선형성이 더 필요합니다.
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            # Layer 3: Compression to f_int
            nn.Linear(hidden_dim, out_dim)
        )
        
        # [중요 3] 가중치 초기화
        # 작은 네트워크일수록 초기화가 학습 속도에 영향을 줍니다.
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
    
class PositionalEncoding(nn.Module):
    """
    Phase 3 Branch B: Spatial Position Encoder
    Input: (N, 3) -> Fourier Mapping -> MLP -> (N, 16)
    """
    def __init__(self, in_dim=3, out_dim=16, n_freqs=4):
        super().__init__()
        
        self.n_freqs = n_freqs
        self.funcs = [torch.sin, torch.cos]
        # 2^0 부터 2^(n-1) 까지 주파수 밴드 생성
        self.freq_bands = 2**torch.linspace(0, n_freqs - 1, n_freqs)
        
        # Fourier Mapping 차원: 3 + (3 * 2 * 6) = 39
        input_dim = in_dim + (in_dim * 2 * n_freqs)
        
        self.net = nn.Sequential(
            # Layer 1: Expansion & Feature Extraction
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            
            # Layer 2: Reasoning (Depth)
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            
            # Layer 3: Compression -> 16차원
            nn.Linear(32, out_dim) 
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_embed(self, x):
        """
        Fourier Feature Mapping
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(x * freq * np.pi))
        return torch.cat(out, dim=-1)

    def forward(self, x):
        # x: (N, 3) -> Fourier Embed (N, 39)
        # Device mismatch 방지: freq_bands를 입력 x와 같은 장치로 이동
        if self.freq_bands.device != x.device:
            self.freq_bands = self.freq_bands.to(x.device)
            
        x_embed = self.get_embed(x)
        return self.net(x_embed) # Output: (N, 16)

