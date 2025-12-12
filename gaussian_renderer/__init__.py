

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

def aggregate_neighbor_features(geometry_features, neighbor_indices, aggregation='mean'):
    """
    주변 가우시안의 geometry features를 aggregate합니다.
    
    Args:
        geometry_features: (N, 128) 각 가우시안의 geometry features
        neighbor_indices: (N, k) 각 가우시안의 k개 이웃 인덱스
        aggregation: 'mean', 'max', 'sum' 중 하나
    
    Returns:
        neighbor_features: (N, 128) 주변 가우시안의 aggregate된 features
    """
    # 각 가우시안의 이웃 features를 가져오기 (N, k, 128)
    neighbor_features = geometry_features[neighbor_indices]  # (N, k, 128)
    
    # Aggregate
    if aggregation == 'mean':
        aggregated = torch.mean(neighbor_features, dim=1)  # (N, 128)
    elif aggregation == 'max':
        aggregated, _ = torch.max(neighbor_features, dim=1)  # (N, 128)
    elif aggregation == 'sum':
        aggregated = torch.sum(neighbor_features, dim=1)  # (N, 128)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return aggregated

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
    
    # KNN을 사용해서 주변 가우시안의 정보를 포함
    # xyz가 변하지 않으므로 한 번만 계산하고 캐시
    if pc._neighbor_indices is None:
        xyz = pc.get_xyz  # (N, 3)
        k_neighbors = pc._k_neighbors
        neighbor_indices, neighbor_distances = get_knn_neighbors(xyz, k=k_neighbors)
        pc._neighbor_indices = neighbor_indices  # 캐시 저장
    else:
        neighbor_indices = pc._neighbor_indices  # 캐시된 결과 사용
    
    # 주변 가우시안의 features를 aggregate
    neighbor_features = aggregate_neighbor_features(p, neighbor_indices, aggregation='mean')  # (N, 128)
    
    # 원래 feature와 주변 feature를 결합 (concatenate 또는 add)
    # 방법 1: Concatenate (256차원) 후 128차원으로 projection
    p_enhanced = torch.cat([p, neighbor_features], dim=-1)  # (N, 256)
    p = pc.p_proj(p_enhanced)  # (N, 256) -> (N, 128)
    p = F.normalize(p, dim=-1)
    
    # 방법 2: Weighted sum (128차원 유지)
    # alpha = 0.5  # 원래 feature와 주변 feature의 가중치 (필요에 따라 조정 가능)
    # p_enhanced = alpha * p + (1 - alpha) * neighbor_features  # (N, 128)
    # p = F.normalize(p_enhanced, dim=-1)
    
    x=pc.mlp2(pc._language_feature)
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