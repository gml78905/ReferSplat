import os
import torch
import numpy as np
from PIL import Image

def calculate_iou(pred_mask, gt_mask):

    intersection = torch.logical_and(pred_mask, gt_mask).sum().float()
    union = torch.logical_or(pred_mask, gt_mask).sum().float()
    iou = intersection / union
    return iou.item()

def load_mask(file_path):
    mask = Image.open(file_path).convert("L") 
    mask = np.array(mask) 
    mask = torch.from_numpy(mask).float() 
    mask = mask > 0 
    return mask

def calculate_iou_for_directory(render_dir, gt_dir):
    iou_list = []

    for filename in os.listdir(render_dir):
        if filename.endswith(".png"): 
            render_path = os.path.join(render_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            pred_mask = load_mask(render_path)
            gt_mask = load_mask(gt_path)
            if gt_mask.sum() == 0:
                continue
            iou = calculate_iou(pred_mask, gt_mask)
            iou_list.append(iou)
    
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    return mean_iou

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate mIoU for rendered results")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--run_number", type=int, default=1, help="Run number (for multi-run training). Default: 1")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    args = parser.parse_args()
    
    # 렌더링 결과 경로 구성: render/{name}/{run_number}/ours_{iteration}/...
    render_dir = os.path.join(args.model_path, "render", args.name, str(args.run_number), f"ours_{args.iteration}", "renders")
    gt_dir = os.path.join(args.model_path, "render", args.name, str(args.run_number), f"ours_{args.iteration}", "gt")
    
    print(f"Render directory: {render_dir}")
    print(f"GT directory: {gt_dir}")
    
    average_iou = calculate_iou_for_directory(render_dir, gt_dir)
    print(f'Run {args.run_number}, Iteration {args.iteration} Average IoU: {average_iou:.4f}')

    # 결과를 파일에 저장: render/{name}/{run_number}/miou_results.txt
    results_dir = os.path.join(args.model_path, "render", args.name, str(args.run_number))
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "miou_results.txt")
    
    # 첫 번째 iteration (0)일 때 결과 파일 초기화
    if args.iteration == 0:
        with open(results_file, "w") as f:
            f.write("")  # 파일 초기화
        print(f"Results file initialized: {results_file}")
    
    # 파일에 결과 추가 (iteration별로)
    with open(results_file, "a") as f:
        f.write(f"Iteration {args.iteration}: {average_iou:.4f}\n")
    
    print(f"Results saved to: {results_file}")

