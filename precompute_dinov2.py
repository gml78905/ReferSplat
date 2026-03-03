import argparse
import os
from pathlib import Path
import sys
from typing import Optional

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
    default_local_repo = Path(__file__).resolve().parent / "third_party" / "dinov2"
    parser.add_argument(
        "--dinov2_dir",
        default=str(default_local_repo) if default_local_repo.is_dir() else None,
        help="Optional path to a local dinov2 repo. Useful on Python<3.10 where torch.hub main may not import.",
    )
    return parser.parse_args()


def load_dinov2_model(model_name: str, device: str, dinov2_dir: Optional[str]):
    """
    Load a DINOv2 backbone model.

    - If `dinov2_dir` is provided, load from that local checkout (Python 3.8-friendly).
    - Otherwise, fall back to `torch.hub.load("facebookresearch/dinov2", ...)`.
    """
    if dinov2_dir is not None:
        repo_root = Path(dinov2_dir).expanduser().resolve()
        if not repo_root.is_dir():
            raise FileNotFoundError(f"--dinov2_dir not found: {repo_root}")
        sys.path.insert(0, str(repo_root))
        from dinov2.hub import backbones  # type: ignore

        if not hasattr(backbones, model_name):
            raise ValueError(
                f"Unknown dinov2 model '{model_name}'. "
                f"Expected one of: {', '.join(sorted([k for k in dir(backbones) if k.startswith('dinov2_')]))}"
            )
        model = getattr(backbones, model_name)(pretrained=True)
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name)

    model.eval().to(device)
    return model


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

    model = load_dinov2_model(args.model, args.device, args.dinov2_dir)

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
