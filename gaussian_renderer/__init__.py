

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
    
    # Extract attributes and normalize
    max_coord = torch.max(torch.abs(pc.get_xyz)).item() + 0.1
    xyz = pc.get_xyz / max_coord  # [N, 3] - 정규화
    scale = pc.get_scaling / max_coord  # [N, 3] - 정규화
    rotation = pc.get_rotation  # [N, 4]
    opacity = pc.get_opacity  # [N, 1]
    
    # SH features: [N, 16, 3] (DC + rest)
    sh_features = pc.get_features  # [N, 16, 3] for degree 3
    
    # Attribute Encoder를 통해 128차원으로 변환
    x = pc.attribute_encoder(xyz, scale, rotation, opacity, sh_features)

    f_x, features = pc.atgm(x, t_token)

    sorted_indices = torch.argsort(features.squeeze(), descending=True)
    indices = sorted_indices[:int(len(sorted_indices) * ratio)]
   
    selected_tensors = f_x[indices]

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