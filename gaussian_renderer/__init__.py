

import time
import torch.nn.functional as F
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def min_max_normalize_torch(points):
    min_vals = points.min(dim=0).values  
    max_vals = points.max(dim=0).values  
    
    normalized_points = 2 * (points - min_vals) / (max_vals - min_vals) - 1
    return normalized_points

def get_knn_neighbors_torch(xyz, k=16):
    """
    PyTorch를 사용하여 KNN 이웃을 찾습니다 (GPU에서 직접 계산).
    
    Args:
        xyz: (N, 3) 가우시안의 3D 위치 (GPU tensor)
        k: 이웃의 개수
    
    Returns:
        neighbor_indices: (N, k) 각 가우시안의 k개 이웃 인덱스 (GPU tensor)
    """
    N = xyz.shape[0]
    
    # 모든 점 간의 거리 계산: (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> (N, N)
    xyz_expanded = xyz.unsqueeze(1)  # (N, 1, 3)
    xyz_transposed = xyz.unsqueeze(0)  # (1, N, 3)
    distances = torch.norm(xyz_expanded - xyz_transposed, dim=2)  # (N, N)
    
    # 자기 자신 제외 (거리를 매우 큰 값으로 설정)
    distances.fill_diagonal_(float('inf'))
    
    # 가장 가까운 k개 이웃 찾기
    _, neighbor_indices = torch.topk(distances, k, dim=1, largest=False)  # (N, k)
    
    return neighbor_indices

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
    

    # Use full geometry features: each feature processed by individual MLP, then fused
    # Returns (N, 128) - already processed through MLPs
    p = pc.get_full_geometry_features()  # (N, 128)
    p=F.normalize(p,dim=-1)
    
    # KNN을 사용하여 주변 16개 feature의 평균값과 자신의 값을 concat
    # xyz 기반 KNN 이웃 찾기 (캐시 사용)
    if pc._neighbor_indices_lang is None:
        xyz = pc.get_xyz  # (N, 3)
        neighbor_indices_lang = get_knn_neighbors_torch(xyz, k=16)  # (N, 16)
        pc._neighbor_indices_lang = neighbor_indices_lang  # 캐시 저장
    else:
        neighbor_indices_lang = pc._neighbor_indices_lang  # 캐시된 결과 사용
    
    # 주변 이웃들의 language_feature 가져오기 (N, 16, 16)
    neighbor_lang_features = pc._language_feature[neighbor_indices_lang]  # (N, 16, 16)
    
    # 주변 feature의 평균 계산
    neighbor_lang_mean = torch.mean(neighbor_lang_features, dim=1)  # (N, 16)
    
    # 자신의 값과 concat: (N, 16) + (N, 16) -> (N, 32)
    language_feature_concat = torch.cat([pc._language_feature, neighbor_lang_mean], dim=-1)  # (N, 32)
    
    x=pc.mlp2(language_feature_concat)
    g=pc.cross_attention(x,p,t_token)
    features=torch.matmul(g,t_token.transpose(-1,-2)).squeeze(0)
    features=features.sum(dim=-1,keepdim=True)

    
    sorted_indices = torch.argsort(features, descending=True)
    indices = sorted_indices[:int(len(sorted_indices) * ratio)].squeeze(1)
   
    selected_tensors = g[indices]

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