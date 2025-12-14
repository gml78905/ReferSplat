import re
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torch.nn as nn
import torch.nn.functional as F
from gaussian_renderer import render
import torchvision
import random
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def overlay_mask_with_boundary(original_image, mask, boundary_color=[1.0, 1.0, 0.0], darken_factor=0.3, boundary_width=2):
    """
    원본 이미지에 마스크를 오버레이합니다. 마스크 외부는 어둡게, 마스크 경계는 강조합니다.
    
    Args:
        original_image: [3, H, W] 텐서 (원본 이미지)
        mask: [1, H, W], [3, H, W] 또는 [H, W] 텐서 (마스크, 0 또는 1)
        boundary_color: 경계 색상 [R, G, B] (0~1 범위)
        darken_factor: 마스크 외부 어둡게 할 비율 (0~1, 작을수록 어둡게)
        boundary_width: 경계선 두께 (픽셀)
    
    Returns:
        [3, H, W] 텐서 (오버레이된 이미지)
    """
    # 디바이스 일치 확인
    device = original_image.device
    mask = mask.to(device)
    
    # 마스크 차원 정규화
    if mask.dim() == 3:
        if mask.shape[0] == 3:
            mask = mask[0]
        else:
            mask = mask[0]
    elif mask.dim() == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    
    # 마스크를 binary로 변환 (0 또는 1)
    mask_binary = (mask > 0.5).float()
    
    # 경계 찾기: Sobel 필터나 간단한 gradient 사용
    # 마스크를 [1, 1, H, W] 형태로 변환
    mask_4d = mask_binary.unsqueeze(0).unsqueeze(0)
    
    # Sobel 필터로 경계 검출
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=mask_binary.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=mask_binary.dtype).view(1, 1, 3, 3)
    
    # Padding 추가
    mask_padded = F.pad(mask_4d, (1, 1, 1, 1), mode='replicate')
    
    # Gradient 계산
    grad_x = F.conv2d(mask_padded, sobel_x, padding=0)
    grad_y = F.conv2d(mask_padded, sobel_y, padding=0)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
    # 경계는 gradient가 큰 부분
    boundary = (gradient_magnitude > 0.1).float().squeeze(0).squeeze(0)
    
    # 경계선 두께 조정을 위해 dilation
    if boundary_width > 1:
        kernel_size = boundary_width * 2 - 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device, dtype=boundary.dtype) / (kernel_size * kernel_size)
        boundary_4d = boundary.unsqueeze(0).unsqueeze(0)
        padding = boundary_width - 1
        boundary_padded = F.pad(boundary_4d, (padding, padding, padding, padding), mode='replicate')
        boundary_dilated = F.conv2d(boundary_padded, kernel, padding=0)
        boundary = (boundary_dilated > 0.1).float().squeeze(0).squeeze(0)
    
    # 마스크 외부 어둡게 처리
    mask_3d = mask_binary.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    darkened_image = original_image * (darken_factor + (1 - darken_factor) * mask_3d)
    
    # 경계선 그리기
    boundary_3d = boundary.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    boundary_color_tensor = torch.tensor(boundary_color, device=device, dtype=original_image.dtype).view(3, 1, 1)
    
    # 경계선이 있는 부분은 경계 색상으로 덮기
    result = darkened_image * (1 - boundary_3d) + boundary_color_tensor * boundary_3d
    
    return result


def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args,model=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")
    overlay_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_overlay")
    gt_overlay_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_overlay")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(overlay_path, exist_ok=True)
    makedirs(gt_overlay_path, exist_ok=True)
    ans=0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for i in range(len(view.sentence)):
            ans+=1
            sn=view.image_name
            number = re.findall(r'\d+', sn)
            number_int = int(number[0])
            output = render(view, gaussians, pipeline, background, args,sentence=view.sentence[i])
            if not args.include_feature:
                rendering = output["render"]
            else:
                rendering = output["language_feature_image"]
                rendering = torch.sigmoid(rendering)
                rendering = (rendering>=0.5).float()
                
            if not args.include_feature:
                gt = view.original_image[0:3, :, :]
                
            else:
                gt=view.gt_mask[view.category[i]] 
            np.save(os.path.join(render_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),rendering.permute(1,2,0).cpu().numpy())
            np.save(os.path.join(gts_npy_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".npy"),gt.permute(1,2,0).cpu().numpy())
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            original_image = view.original_image[0:3, :, :].to(rendering.device)
            # 예측 마스크 오버레이: 마스크 외부 어둡게, 경계 강조 (노란색 경계)
            overlay_image = overlay_mask_with_boundary(original_image, rendering, boundary_color=[1.0, 1.0, 0.0], darken_factor=0.3, boundary_width=2)
            torchvision.utils.save_image(overlay_image, os.path.join(overlay_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
            # GT 마스크 오버레이: 마스크 외부 어둡게, 경계 강조 (노란색 경계)
            gt_overlay_image = overlay_mask_with_boundary(original_image, gt, boundary_color=[1.0, 1.0, 0.0], darken_factor=0.3, boundary_width=2)
            torchvision.utils.save_image(gt_overlay_image, os.path.join(gt_overlay_path, '{0:05d}'.format(number_int) + '{}'.format(view.category[i])+".png"))
               
               
def render_sets(dataset : ModelParams,model_path, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, k_neighbors=16):
    with torch.no_grad():  
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians._k_neighbors = k_neighbors  # KNN 이웃 개수 설정
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, model_path)
        (model_params, first_iter) = torch.load(checkpoint,map_location=f'cuda:{torch.cuda.current_device()}')
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        iteration = args.iteration
        
        # if not skip_train:
        #      render_set(dataset.model_path, dataset.source_path, "render_train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             # 렌더링 결과 경로: render/{name}/{run_number}/...
             render_name = os.path.join("render", args.name, str(args.run_number))
             render_set(dataset.model_path, dataset.source_path, render_name, iteration, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--name", type=str, required=True, help="Experiment name for checkpoint loading")
    parser.add_argument("--run_number", type=int, default=1, help="Run number (for multi-run training). Default: 1")
    parser.add_argument("--k_neighbors", type=int, default=16, help="Number of KNN neighbors for feature aggregation. Default: 16")
    args = get_combined_args(parser)
    args.include_feature=True
    
    # 체크포인트 경로 생성: {model_path}/checkpoints/stage2/{name}/{run_number}/chkpnt_{iteration}.pth
    if args.iteration == -1:
        args.iteration = 4  # 기본값
    model_path = os.path.join("checkpoints", "stage2", args.name, str(args.run_number), f"chkpnt_{args.iteration}.pth")
    
    render_sets(model.extract(args), model_path, pipeline.extract(args), args.skip_train, args.skip_test, args, k_neighbors=args.k_neighbors)