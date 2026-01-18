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
    def __init__(self, out_dim=128, owner=None):
        super().__init__()
        self._owner = owner

        # -----------------------------------------------------------
        # 1. Geometry Stream
        # -----------------------------------------------------------
        self.L = 10
        self.geo_input_dim = 60 + 3 + 4 + 6
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.geo_input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # 2. Appearance Stream
        # -----------------------------------------------------------
        self.sh_compressor = nn.Linear(45, 16)
        self.app_input_dim = 3 + 16 + 1
        self.app_mlp = nn.Sequential(
            nn.Linear(self.app_input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # 3. Final Projection
        # -----------------------------------------------------------
        self.final_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.dual_scale_context_block = DualScaleContextBlock(in_dim=32)

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

    def compute_cov_features(self, scale, rotation):
        return torch.zeros(scale.shape[0], 6, device=scale.device)

    def forward(self, xyz, scale, rotation, opacity, sh_features):
        # Geometry stream
        ipe = self.integrated_positional_encoding(xyz, scale)
        westin = self.compute_westin_metrics(scale)
        scale_log = torch.log(scale + 1e-9)
        cov_feat = self.compute_cov_features(scale, rotation)
        geo_in = torch.cat([ipe, scale_log, westin, cov_feat], dim=1)
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
        f_fused = f_geo + f_app

        
        knn_idx = None
        if self._owner is not None:
            knn_idx = getattr(self._owner, "_neighbor_indices", None)

        if knn_idx is not None:
            f_fused = self.dual_scale_context_block(f_fused, xyz, scale, rotation, knn_idx)

        out = self.final_head(f_fused)
        return out


class ContextBlock(nn.Module):
    """
    Geometry-Aware Context Fusion Module
    - Implicit Rotation Learning (No explicit matrix multiplication)
    - Dual-Scale Normalization (Self & Neighbor perspective)
    - Gated Residual Connection
    """
    def __init__(self, in_dim=128, head_dim=32):
        super().__init__()

        # -------------------------------------------------------
        # 1. Geometric Encoder (공간 관계 해석기)
        # -------------------------------------------------------
        # 입력 차원 구성 (총 17 dim):
        # - Relative XYZ (3): 물리적 상대 위치
        # - Distance (1): 물리적 거리
        # - Norm Relative Self (3): 내 크기 기준 상대 위치
        # - Norm Relative Neigh (3): 이웃 크기 기준 상대 위치
        # - Neighbor Log Scale (3): 이웃의 절대적 체급
        # - Self Rotation Quat (4): 나의 방향 (이게 들어가서 회전을 학습함)
        self.geo_input_dim = 3 + 1 + 3 + 3 + 3 + 4

        self.geo_mlp = nn.Sequential(
            nn.Linear(self.geo_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim)  # Feature 차원과 맞춤 (Positional Embedding)
        )

        # -------------------------------------------------------
        # 2. Attention Mechanism (정보 융합기)
        # -------------------------------------------------------
        self.scale = head_dim ** -0.5

        # 메모리 절약을 위해 Projection 차원(head_dim)을 작게 유지
        self.to_q = nn.Linear(in_dim, head_dim, bias=False)
        self.to_k = nn.Linear(in_dim, head_dim, bias=False)
        self.to_v = nn.Linear(in_dim, in_dim, bias=False)  # Value는 정보 보존 (128)

        # -------------------------------------------------------
        # 3. Gating Mechanism (안전 장치)
        # -------------------------------------------------------
        # Context가 내 정보를 덮어쓰지 못하도록 조절
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, f_self, xyz, scale, rot, knn_idx):
        """
        Args:
            f_self: [N, C] (AttributeEncoder에서 나온 Point-wise Feature)
            xyz: [N, 3]
            scale: [N, 3]
            rot: [N, 4] (Quaternion)
            knn_idx: [N, K] (미리 계산된 이웃 인덱스)
        Returns:
            f_enhanced: [N, C] (Context가 주입된 최종 Feature)
        """
        N, K = knn_idx.shape

        # =======================================================
        # Step 1: Gathering (이웃 정보 가져오기)
        # =======================================================
        # 내 Feature가 아니라 '이웃'의 Feature와 기하 정보를 가져옴
        # knn_idx가 LongTensor여야 함

        f_neigh = f_self[knn_idx]      # [N, K, C]
        xyz_neigh = xyz[knn_idx]       # [N, K, 3]
        scale_neigh = scale[knn_idx]   # [N, K, 3]

        # =======================================================
        # Step 2: Geometric Embedding (Implicit Geometry)
        # =======================================================

        # (A) 물리적 상대 위치 & 거리
        rel_xyz = xyz_neigh - xyz.unsqueeze(1)  # [N, K, 3]
        dist = torch.norm(rel_xyz, dim=-1, keepdim=True)  # [N, K, 1]

        # (B) Broadcasting 준비
        scale_self_exp = scale.unsqueeze(1)  # [N, 1, 3]
        scale_neigh_exp = scale_neigh        # [N, K, 3]

        # (C) Dual Normalization (내 기준 vs 이웃 기준)
        # 엡실론(1e-9) 더해서 0으로 나누기 방지
        norm_rel_self = rel_xyz / (scale_self_exp + 1e-9)
        norm_rel_neigh = rel_xyz / (scale_neigh_exp + 1e-9)

        # (D) Contextual Properties
        log_scale_neigh = torch.log(scale_neigh + 1e-9)
        rot_self_exp = rot.unsqueeze(1).expand(-1, K, -1)  # [N, K, 4]

        # (E) 입력 벡터 결합 (총 17차원)
        geo_vector = torch.cat([
            rel_xyz,        # 물리적 위치 (3)
            dist,           # 물리적 거리 (1)
            norm_rel_self,  # 내 체급 기준 위치 (3)
            norm_rel_neigh, # 이웃 체급 기준 위치 (3)
            log_scale_neigh,# 이웃의 절대 체급 (3)
            rot_self_exp    # 나의 회전 (4) -> MLP가 방향성 학습!
        ], dim=-1)

        # (F) Positional Embedding 생성
        pos_emb = self.geo_mlp(geo_vector)  # [N, K, 128]

        # =======================================================
        # Step 3: Attention-based Fusion
        # =======================================================

        # Query: 나 자신
        q = self.to_q(f_self).unsqueeze(1)  # [N, 1, 32]

        # Key & Value: 이웃의 정보 (Feature + Geometric Context)
        # "내 오른쪽(Geo)에 있는 노란색(Feature)"
        kv_input = f_neigh + pos_emb

        k = self.to_k(kv_input)  # [N, K, 32]
        v = self.to_v(kv_input)  # [N, K, 128]

        # Dot Product Attention
        scores = (q * k).sum(dim=-1, keepdim=True) * self.scale
        attn_weights = F.softmax(scores, dim=1)  # [N, K, 1]

        # Weighted Sum (Context Vector)
        context = (v * attn_weights).sum(dim=1)  # [N, 128]

        # =======================================================
        # Step 4: Gated Residual Connection
        # =======================================================

        # Gate 계산: 내 정보와 외부 정보를 보고 섞는 비율 결정 (0~1)
        # [N, 256] -> [N, 1]
        gate = self.gate_mlp(torch.cat([f_self, context], dim=-1))

        # 최종 융합
        out = f_self + gate * self.out_proj(context)

        return out


class DualScaleContextBlock(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()

        # 1. 두 개의 시야를 담당할 Attention 모듈 (가볍게 설계)
        # Micro: 형상 파악용
        self.micro_attn = ContextBlock(in_dim, head_dim=32)
        # Macro: 관계 파악용
        self.macro_attn = ContextBlock(in_dim, head_dim=32)

        # 2. 융합 레이어 (Concatenation -> Reduction)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)  # 원래 차원으로 복구
        )

    def forward(self, f_self, xyz, scale, rot, knn_idx_large):
        """
        knn_idx_large: [N, 64] (넉넉하게 검색된 이웃 인덱스)
        """

        # --- [핵심] 인덱스 슬라이싱 (Index Slicing) ---

        # 1. Micro Group: 아주 가까운 8개
        # 목적: 젓가락의 매끈한 표면 학습
        micro_idx = knn_idx_large[:, :8]

        # 2. Macro Group: 건너뛰며 뽑은 8개 (Dilated)
        # 목적: 젓가락 옆에 있는 그릇 감지
        # 8번부터 4칸 간격으로: [8, 12, 16, 20, 24, 28, 32, 36]
        macro_idx = knn_idx_large[:, 8:40:4]

        # --- 병렬 처리 (Parallel Processing) ---
        # 각각의 관점에서 Context를 추출
        f_micro = self.micro_attn(f_self, xyz, scale, rot, micro_idx)
        f_macro = self.macro_attn(f_self, xyz, scale, rot, macro_idx)

        # --- 융합 (Fusion) ---
        # 두 정보를 이어 붙임 (Concat)
        f_cat = torch.cat([f_micro, f_macro], dim=-1)  # [N, 256]

        # 정보를 섞어서 압축
        context = self.fusion(f_cat)  # [N, 128]

        # Residual Connection (내 원래 정보 보존)
        return f_self + context