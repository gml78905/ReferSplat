

import time
import torch.nn.functional as F
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from sklearn.neighbors import NearestNeighbors

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
    cls_token, t_token=pc.get_text(sentence)
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
    
    # Extract attributes
    max_coord = torch.max(torch.abs(pc.get_xyz.detach())).item() + 0.1
    xyz = pc.get_xyz / max_coord  # [N, 3]
    scale = pc.get_scaling / max_coord  # [N, 3]
    rotation = pc.get_rotation  # [N, 4]
    opacity = pc.get_opacity  # [N, 1]
    
    # SH features: [N, 16, 3] (DC + rest)
    sh_features = pc.get_features  # [N, 16, 3] for degree 3
    
    if pc._neighbor_indices is None:
        xyz_knn = pc.get_xyz  # (N, 3)
        k_neighbors = pc._k_neighbors
        neighbor_indices, neighbor_distances = get_knn_neighbors(xyz_knn, k=k_neighbors)
        pc._neighbor_indices = neighbor_indices  # 캐시 저장
    else:
        neighbor_indices = pc._neighbor_indices  # 캐시된 결과 사용
    
    x = pc.attribute_encoder(xyz, scale, rotation, opacity, sh_features, cls_token)
    
    p = pc.mlp3(pc.get_xyz)
    p = F.normalize(p, dim=-1)
    g = pc.cross_attention(x, p, t_token)
    features = torch.matmul(g, t_token.transpose(-1, -2)).squeeze(0)
    features = features.sum(dim=-1, keepdim=True)

    
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