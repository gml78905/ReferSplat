
import time
import torch.nn.functional as F
import torch
import math
from sklearn.neighbors import NearestNeighbors
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def min_max_normalize_torch(points):
    min_vals = points.min(dim=0).values  
    max_vals = points.max(dim=0).values  
    
    normalized_points = 2 * (points - min_vals) / (max_vals - min_vals) - 1
    return normalized_points

def fuse_neighbor_features(pc, neighbor_indices):
    """
    각 가우시안이 주변 가우시안의 정보와 융합하는 과정
    
    Args:
        pc: GaussianModel 인스턴스
        neighbor_indices: (N, K) 각 가우시안의 K개 neighbor 인덱스
    
    Returns:
        f_ctx: (N, 16) 주변 맥락 정보가 융합된 intrinsic feature
    """
    # 1. Intrinsic feature 계산
    intrinsic_feature = pc.get_intrinsic_feature()  # (N, 9)
    intrinsic_feature = pc.intrinsic_encoder(intrinsic_feature)  # (N, 16)
    
    # 2. 상대 위치 계산 및 Positional Encoding
    xyz = pc.get_xyz  # (N, 3)
    k_neighbors = neighbor_indices.shape[1]  # K
    
    p_i = xyz.unsqueeze(1).expand(-1, k_neighbors, -1)  # (N, K, 3) - 현재 가우시안
    p_j = xyz[neighbor_indices]  # (N, K, 3) - neighbor 가우시안들
    delta_p = p_j - p_i  # (N, K, 3) - 상대 위치
    pos_emb = pc.pos_mlp(delta_p)  # (N, K, 16) - Positional Encoding
    
    # 3. Feature 준비
    f_i = intrinsic_feature.unsqueeze(1).expand(-1, k_neighbors, -1)  # (N, K, 16) - 현재 가우시안
    f_j = intrinsic_feature[neighbor_indices]  # (N, K, 16) - neighbor 가우시안들
    
    # 4. Attention score 계산: e_ij = LeakyReLU(W_att [f_i || f_j || MLP_pos(Δp_ij)])
    concat_feat = torch.cat([f_i, f_j, pos_emb], dim=-1)  # (N, K, 48)
    e_ij = pc.att_layer(concat_feat)  # (N, K, 1)
    e_ij = F.leaky_relu(e_ij, negative_slope=0.2)  # LeakyReLU
    e_ij = e_ij.squeeze(-1)  # (N, K)
    
    # 5. Attention weights 계산
    alpha_ij = F.softmax(e_ij, dim=1)  # (N, K)
    alpha_ij = alpha_ij.unsqueeze(-1)  # (N, K, 1)
    
    # 6. Aggregation (Weighted Sum)
    # 가중치(alpha) * neighbor feature(f_j)를 이웃 차원(dim=1)에 대해 합산
    context = torch.sum(alpha_ij * f_j, dim=1)  # (N, 16) - 주변 맥락 정보
    
    # 7. Residual Connection (Update)
    # 원래 intrinsic_feature를 유지하면서 맥락 정보 추가
    f_ctx = intrinsic_feature + context  # (N, 16)
    f_ctx = pc.layer_norm(f_ctx)
    
    return f_ctx

def get_knn_neighbors(xyz, k=8):
    """
    KNN을 사용해서 각 가우시안에 대해 k개의 가장 가까운 이웃을 찾습니다.
    업계 표준 방식 (KD-Tree)을 사용하여 메모리 효율적이고 빠른 검색을 수행합니다.
    
    Args:
        xyz: (N, 3) 가우시안의 3D 위치 (GPU tensor)
        k: 이웃의 개수
    
    Returns:
        neighbor_indices: (N, k) 각 가우시안의 k개 이웃 인덱스 (GPU tensor)
        neighbor_distances: (N, k) 각 가우시안과 이웃 간의 거리 (GPU tensor)
    """
    # 1. CPU로 내리기 (메모리 안전지대)
    points_np = xyz.detach().cpu().numpy()
    
    # 2. KD-Tree 빌드 및 검색 (n_jobs=-1로 병렬 처리)
    # 알고리즘이 알아서 'ball_tree', 'kd_tree', 'brute' 중 최적을 선택함
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1).fit(points_np)
    
    # 3. 검색 (메모리 안 터짐)
    distances, indices = nbrs.kneighbors(points_np)
    
    # 4. 자기 자신 제외 후 GPU로 복귀
    return torch.from_numpy(indices[:, 1:]).cuda(), torch.from_numpy(distances[:, 1:]).cuda()

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None,sentence=None,ratio=0.03):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

   
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    t_token=pc.get_text(sentence).to("cuda")
    t_token=pc.mlp1(t_token)
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if pc._neighbor_indices is None:
        xyz = pc.get_xyz  # (N, 3)
        k_neighbors = pc._k_neighbors
        neighbor_indices, neighbor_distances = get_knn_neighbors(xyz, k=k_neighbors)
        pc._neighbor_indices = neighbor_indices  # 캐시 저장
    else:
        neighbor_indices = pc._neighbor_indices  # 캐시된 결과 사용
    
    # 주변 가우시안 정보와 융합
    f_ctx = fuse_neighbor_features(pc, neighbor_indices)  # (N, 16)
    
    # Phase 3: Decoupled Spatial-Semantic Matching Module
    # Branch 1: Semantic Reasoning
    # t_token: (seq_len, 128) - 각 토큰의 embedding
    # 1. 점수 계산: 각 단어(토큰)가 얼마나 중요한지 점수를 매김
    # score = Linear(T_emb) -> (seq_len, 1)
    score = pc.text_score_layer_sem(t_token)  # (seq_len, 1)
    score = score.squeeze(-1)  # (seq_len,)
    
    # 2. 가중치 변환: Softmax를 취해 합이 1이 되도록 만듦
    # alpha = Softmax(score) -> (seq_len,)
    alpha = F.softmax(score, dim=0)  # (seq_len,)
    
    # 3. 가중 합: 가중치만큼 곱해서 더함
    # T_final = sum(alpha * T_emb) -> (1, 128)
    # 메모리 효율: 브로드캐스팅으로 직접 계산
    T_emb_weighted = alpha.unsqueeze(-1) * t_token  # (seq_len, 128)
    T_final = torch.sum(T_emb_weighted, dim=0, keepdim=True)  # (1, 128)
    
    # T_final -> text_proj_sem -> T_sem
    T_sem = pc.text_proj_sem(T_final)  # (1, 16)
    T_sem = F.normalize(T_sem, dim=-1)  # L2 normalize
    
    # 메모리 최적화: 중간 텐서 삭제
    del score, alpha, T_emb_weighted, T_final
    
    # Cosine Similarity: f_ctx (N, 16) vs T_sem (1, 16)
    # 메모리 효율: 브로드캐스팅으로 직접 계산 (큰 텐서 생성 방지)
    f_ctx_norm = F.normalize(f_ctx, dim=-1)  # (N, 16)
    S_sem = torch.sigmoid(torch.sum(f_ctx_norm * T_sem, dim=-1))  # (N,) - 브로드캐스팅
    
    # Branch 2: Spatial Reasoning
    # 좌표 입력 -> PositionalEncoding -> f_pos (N, 16)
    xyz = pc.get_xyz  # (N, 3)
    f_pos = pc.pos_mlp(xyz)  # (N, 16) - PositionalEncoding
    f_pos = F.normalize(f_pos, dim=-1)  # L2 normalize
    
    # t_token에 대해 spatial용 별도 attention 가중치 계산
    # text_score_layer_pos를 사용하여 spatial에 중요한 토큰에 더 높은 가중치 부여
    score_pos = pc.text_score_layer_pos(t_token)  # (seq_len, 1)
    score_pos = score_pos.squeeze(-1)  # (seq_len,)
    alpha_pos = F.softmax(score_pos, dim=0)  # (seq_len,)
    T_emb_weighted_pos = alpha_pos.unsqueeze(-1) * t_token  # (seq_len, 128)
    T_final_pos = torch.sum(T_emb_weighted_pos, dim=0, keepdim=True)  # (1, 128)
    
    T_pos = pc.text_proj_pos(T_final_pos)  # (1, 16)
    
    # 메모리 최적화: 중간 텐서 삭제
    del score_pos, alpha_pos, T_emb_weighted_pos, T_final_pos
    T_pos = F.normalize(T_pos, dim=-1)  # L2 normalize
    
    # Cosine Similarity: f_pos (N, 16) vs T_pos (1, 16)
    # 메모리 효율: 브로드캐스팅으로 직접 계산
    S_pos = torch.sigmoid(torch.sum(f_pos * T_pos, dim=-1))  # (N,) - 브로드캐스팅
    
    # Final Fusion
    final_score = S_sem * S_pos  # (N,)
    
    # 메모리 최적화: 중간 텐서 삭제
    del T_sem, T_pos, f_ctx_norm, f_pos
    
    # final_score를 features로 사용 (rasterizer에 전달)
    features = final_score.unsqueeze(-1)  # (N, 1)
    
    
    sorted_indices = torch.argsort(features, descending=True)
    indices = sorted_indices[:int(len(sorted_indices) * ratio)].squeeze(1)
   
    selected_tensors = g[indices]

    mean_tensor = torch.mean(selected_tensors, dim=0, keepdim=True)
    
    # # Positional encoding을 위한 상대 위치 계산 (cross_attention용)
    # xyz = pc.get_xyz  # (N, 3)
    # k_neighbors = neighbor_indices.shape[1]
    # p_i = xyz.unsqueeze(1).expand(-1, k_neighbors, -1)  # (N, K, 3)
    # p_j = xyz[neighbor_indices]  # (N, K, 3)
    # delta_p = p_j - p_i  # (N, K, 3)
    # pos_emb = pc.pos_mlp(delta_p)  # (N, K, 16)
    
    # # Attention weights로 pos_emb를 가중 평균하여 p 생성
    # intrinsic_feature = pc.get_intrinsic_feature()  # (N, 9)
    # intrinsic_feature = pc.intrinsic_encoder(intrinsic_feature)  # (N, 16)
    # f_i = intrinsic_feature.unsqueeze(1).expand(-1, k_neighbors, -1)  # (N, K, 16)
    # f_j = intrinsic_feature[neighbor_indices]  # (N, K, 16)
    # concat_feat = torch.cat([f_i, f_j, pos_emb], dim=-1)  # (N, K, 48)
    # e_ij = pc.att_layer(concat_feat)  # (N, K, 1)
    # e_ij = F.leaky_relu(e_ij, negative_slope=0.2).squeeze(-1)  # (N, K)
    # attention_weights = F.softmax(e_ij, dim=1)  # (N, K)
    # p = torch.sum(attention_weights.unsqueeze(-1) * pos_emb, dim=1)  # (N, 16)
    # p = F.normalize(p, dim=-1)
    
    # g = pc.cross_attention(f_ctx, p, t_token)
    # features=torch.matmul(g,t_token.transpose(-1,-2)).squeeze(0)
    # features=features.sum(dim=-1,keepdim=True)

    
    # sorted_indices = torch.argsort(features, descending=True)
    # indices = sorted_indices[:int(len(sorted_indices) * ratio)].squeeze(1)
   
    # selected_tensors = g[indices]

    # mean_tensor = torch.mean(selected_tensors, dim=0, keepdim=True)

    

    rendered_image, language_feature_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        language_feature_precomp = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return {"render": rendered_image,
            "language_feature_image": language_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "mean_tensor": mean_tensor}
