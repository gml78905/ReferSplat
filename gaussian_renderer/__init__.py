

import time
import torch.nn.functional as F
import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from sklearn.neighbors import NearestNeighbors

def min_max_normalize_torch(points):
    min_vals = points.min(dim=0).values  
    max_vals = points.max(dim=0).values  
    
    normalized_points = 2 * (points - min_vals) / (max_vals - min_vals) - 1
    return normalized_points

def build_static_graph(xyz_numpy, k=30):
    """
    고정된 XYZ 좌표를 사용하여 KNN 그래프를 구축합니다.
    
    Args:
        xyz_numpy: [N, 3] CPU numpy array - 고정된 3D 좌표
        k: 이웃의 개수 (기본값: 30)
    
    Returns:
        neighbor_indices: [N, k] GPU Tensor - 각 포인트의 k개 이웃 인덱스
        weights: [N, k] GPU Tensor - 거리 기반 가중치 (정규화됨)
    """
    # xyz_numpy: [N, 3] cpu array
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz_numpy)
    distances, indices = nbrs.kneighbors(xyz_numpy)
    
    # GPU Tensor로 변환하여 모델에 등록
    neighbor_indices = torch.from_numpy(indices).long().cuda()
    
    # 거리 기반 가중치 계산 (Inverse distance weighting)
    weights = 1.0 / (distances + 1e-6)
    weights = torch.from_numpy(weights).float().cuda()
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
    
    return neighbor_indices, weights

def aggregate_neighbors(f_raw, neighbor_indices, weights):
    """
    이웃 특징들을 가중 평균하여 aggregate하고 residual connection을 적용합니다.
    
    Args:
        f_raw: [N, 128] - 원본 특징
        neighbor_indices: [N, K] - 각 포인트의 K개 이웃 인덱스
        weights: [N, K] - 각 이웃에 대한 가중치 (정규화됨)
    
    Returns:
        f_agg: [N, 128] - aggregate된 특징 (residual connection 적용)
    """
    # f_raw: [N, 128]
    # neighbor_indices: [N, K]
    # 1. 이웃 특징 가져오기 (Gather)
    # [N, K, 128] 형태가 됨
    neighbor_feats = f_raw[neighbor_indices] 
    
    # 2. 가중 평균 (Weighted Mean)
    # weights: [N, K] -> [N, K, 1] for broadcasting
    weighted_feats = neighbor_feats * weights.unsqueeze(-1)
    f_agg = torch.sum(weighted_feats, dim=1)  # [N, 128]
    
    # 3. Residual Connection
    return f_raw + f_agg

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
    

    # 고정된 XYZ 좌표로 KNN 수행 (한 번만 계산하고 캐시)
    if pc._static_neighbor_indices is None or pc._static_neighbor_weights is None:
        # 고정된 XYZ 좌표 가져오기 (detach로 gradient 차단)
        xyz_fixed = pc.get_xyz.detach().cpu().numpy()  # [N, 3]
        
        # KNN 그래프 구축
        neighbor_indices, neighbor_weights = build_static_graph(xyz_fixed, k=30)
        
        # 캐시에 저장
        pc._static_neighbor_indices = neighbor_indices
        pc._static_neighbor_weights = neighbor_weights
    else:
        # 캐시된 결과 사용
        neighbor_indices = pc._static_neighbor_indices
        neighbor_weights = pc._static_neighbor_weights
    
    # 가우시안의 모든 속성을 활용하여 x 생성
    # Attribute Encoder에 속성들을 개별적으로 전달 (내부에서 처리)
    xyz = pc.get_xyz  # [N, 3]
    scale = pc.get_scaling  # [N, 3]
    rotation = pc.get_rotation  # [N, 4]
    opacity = pc.get_opacity  # [N, 1]
    
    # SH features: [N, 16, 3] (DC + rest)
    sh_features = pc.get_features  # [N, 16, 3] for degree 3
    
    # Language feature: [N, 16]
    language_feature = pc._language_feature if pc._language_feature is not None else None
    
    # Attribute Encoder를 통해 128차원으로 변환
    x = pc.attribute_encoder(xyz, scale, rotation, opacity, sh_features, language_feature)  # [N, 128]
    
    # 이웃 특징들과 aggregate (Residual Connection 적용)
    f_3d = aggregate_neighbors(x, neighbor_indices, neighbor_weights)  # [N, 128]
    
    # ATGM을 사용하여 features 계산
    # f_3d: [N, 128], f_text: [1, T, 128] (t_token)
    features = pc.atgm(f_3d, t_token)  # [N, 1]

    
    sorted_indices = torch.argsort(features.squeeze(), descending=True)
    indices = sorted_indices[:int(len(sorted_indices) * ratio)]
   
    selected_tensors = f_3d[indices]

    mean_tensor = torch.mean(selected_tensors, dim=0, keepdim=True)

    

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