import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DINOv2 patch features.")
    parser.add_argument("--images_dir", required=True, help="Directory with input images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save features.")
    parser.add_argument("--model", default="dinov2_vitb14", help="DINOv2 model name.")
    parser.add_argument("--patch_size", type=int, default=14, help="ViT patch size.")
    parser.add_argument("--image_size", type=int, default=None, help="Optional square resize.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extensions", default="jpg,jpeg,png")
    return parser.parse_args()


def resize_to_multiple(image, patch_size, image_size=None):
    orig_w, orig_h = image.size
    if image_size is not None:
        new_w = image_size
        new_h = image_size
    else:
        new_w = max(patch_size, (orig_w // patch_size) * patch_size)
        new_h = max(patch_size, (orig_h // patch_size) * patch_size)
    if (new_w, new_h) != (orig_w, orig_h):
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image, (orig_h, orig_w), (new_h, new_w)


def load_image(path, patch_size, image_size, transform):
    image = Image.open(path).convert("RGB")
    image, orig_size, resize_size = resize_to_multiple(image, patch_size, image_size)
    tensor = transform(image).unsqueeze(0)
    return tensor, orig_size, resize_size


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().to(args.device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    exts = [ext.strip().lower() for ext in args.extensions.split(",")]
    image_paths = []
    for ext in exts:
        image_paths.extend(Path(args.images_dir).rglob(f"*.{ext}"))
    image_paths = sorted(set(image_paths))

    for image_path in tqdm(image_paths, desc="DINOv2 precompute"):
        image_tensor, orig_size, resize_size = load_image(
            image_path, args.patch_size, args.image_size, transform
        )
        image_tensor = image_tensor.to(args.device)

        with torch.no_grad():
            feats = model.forward_features(image_tensor)
            patch_tokens = feats["x_norm_patchtokens"]

        _, num_patches, feat_dim = patch_tokens.shape
        h_p = resize_size[0] // args.patch_size
        w_p = resize_size[1] // args.patch_size
        if h_p * w_p != num_patches:
            raise ValueError(
                f"Patch count mismatch for {image_path}: "
                f"{h_p}x{w_p} != {num_patches}"
            )

        patch_tokens = patch_tokens.reshape(1, h_p, w_p, feat_dim).permute(0, 3, 1, 2)
        feat = patch_tokens.squeeze(0).contiguous().cpu().numpy()
        if args.dtype == "float16":
            feat = feat.astype(np.float16)
        else:
            feat = feat.astype(np.float32)

        out_path = Path(args.output_dir) / f"{image_path.stem}.npz"
        np.savez_compressed(
            out_path,
            feat=feat,
            orig_size=np.array(orig_size, dtype=np.int32),
            resize_size=np.array(resize_size, dtype=np.int32),
            patch_size=np.array(args.patch_size, dtype=np.int32),
            model=np.array(args.model),
        )


if __name__ == "__main__":
    main()
