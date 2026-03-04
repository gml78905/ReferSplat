"""
Referring Expression Segmentation for Complex Sentences
========================================================
Segments a target object from a 2D image given a complex natural language
description that includes attributes and spatial relationships.

Approach: Decompose the complex sentence into sub-queries, detect each entity
with Grounding DINO, apply spatial constraints, then segment with SAM.

Usage:
    python referring_segmentation_demo.py \
        --image_path <path_to_image> \
        --text_prompt "A metal utensil that is convenient for holding cutlery, located on a wooden table and below the yellow bowl."

Requirements:
    - Grounding DINO checkpoint (groundingdino_swint_ogc.pth)
    - SAM checkpoint (sam_vit_h_4b8939.pth)
    - Or use --use_lisa for LISA-based reasoning segmentation
"""

import argparse
import os
import sys
import re
import json
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Grounded-Segment-Anything"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Grounded-Segment-Anything", "segment_anything"))


# ---------------------------------------------------------------------------
# 1) Sentence Decomposition (rule-based, no external LLM dependency)
# ---------------------------------------------------------------------------

SPATIAL_KEYWORDS = {
    "on": "on",
    "above": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "left of": "left_of",
    "right of": "right_of",
    "to the left of": "left_of",
    "to the right of": "right_of",
    "next to": "next_to",
    "near": "near",
    "behind": "behind",
    "in front of": "in_front_of",
    "inside": "inside",
    "between": "between",
}


def decompose_sentence(sentence: str) -> Dict:
    """
    Rule-based decomposition of a referring expression into:
      - target: the main object noun phrase
      - attributes: descriptive attributes
      - spatial_relations: list of {anchor, relation}

    Example input:
      "A metal utensil that is convenient for holding cutlery,
       located on a wooden table and below the yellow bowl."

    Example output:
      {
        "target": "metal utensil",
        "attributes": ["metal", "convenient for holding cutlery"],
        "spatial_relations": [
          {"anchor": "wooden table", "relation": "on"},
          {"anchor": "yellow bowl", "relation": "below"}
        ]
      }
    """
    sentence = sentence.strip().rstrip(".")
    result = {
        "target": "",
        "attributes": [],
        "spatial_relations": [],
    }

    parts = re.split(r",\s*(?:and\s+)?(?:which\s+is\s+)?(?:located\s+)?|(?:\s+and\s+)", sentence, flags=re.IGNORECASE)

    if parts:
        target_part = parts[0].strip()
        target_part = re.sub(r"^(a|an|the)\s+", "", target_part, flags=re.IGNORECASE)
        target_match = re.match(r"^(.+?)(?:\s+that\s+(?:is|are)\s+(.+))?$", target_part, re.IGNORECASE)
        if target_match:
            result["target"] = target_match.group(1).strip()
            if target_match.group(2):
                result["attributes"].append(target_match.group(2).strip())
        else:
            result["target"] = target_part

    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue

        found_relation = False
        for keyword, relation in sorted(SPATIAL_KEYWORDS.items(), key=lambda x: -len(x[0])):
            pattern = rf"(?:is\s+)?{re.escape(keyword)}\s+(?:the\s+|a\s+|an\s+)?(.+)"
            match = re.search(pattern, part, re.IGNORECASE)
            if match:
                anchor = match.group(1).strip().rstrip(".")
                result["spatial_relations"].append({
                    "anchor": anchor,
                    "relation": relation,
                })
                found_relation = True
                break

        if not found_relation:
            result["attributes"].append(part)

    return result


# ---------------------------------------------------------------------------
# 2) Spatial Relationship Checking
# ---------------------------------------------------------------------------

def box_center(box: torch.Tensor) -> Tuple[float, float]:
    """Returns (cx, cy) of a [x1, y1, x2, y2] box."""
    return ((box[0] + box[2]) / 2).item(), ((box[1] + box[3]) / 2).item()


def box_iou_1d(a_min, a_max, b_min, b_max):
    """1D IoU for overlap checking."""
    inter = max(0, min(a_max, b_max) - max(a_min, b_min))
    union = max(a_max, b_max) - min(a_min, b_min)
    return inter / union if union > 0 else 0


def check_spatial_relation(target_box: torch.Tensor, anchor_box: torch.Tensor,
                           relation: str, margin: float = 0.15) -> float:
    """
    Returns a score [0, 1] for how well the spatial relation holds.
    Boxes are in [x1, y1, x2, y2] format (pixel coordinates).
    """
    t_cx, t_cy = box_center(target_box)
    a_cx, a_cy = box_center(anchor_box)

    img_diag = max(1.0, ((target_box[2] - target_box[0])**2 +
                         (target_box[3] - target_box[1])**2).sqrt().item())

    if relation == "on":
        t_bottom = target_box[3].item()
        a_top = anchor_box[1].item()
        a_bottom = anchor_box[3].item()
        h_overlap = box_iou_1d(target_box[0].item(), target_box[2].item(),
                               anchor_box[0].item(), anchor_box[2].item())
        vertical_ok = a_top <= t_bottom <= a_bottom * 1.2
        return float(vertical_ok) * max(0.3, h_overlap)

    elif relation == "below":
        return max(0, min(1.0, (t_cy - a_cy) / (img_diag * 0.3)))

    elif relation == "above":
        return max(0, min(1.0, (a_cy - t_cy) / (img_diag * 0.3)))

    elif relation == "left_of":
        return max(0, min(1.0, (a_cx - t_cx) / (img_diag * 0.3)))

    elif relation == "right_of":
        return max(0, min(1.0, (t_cx - a_cx) / (img_diag * 0.3)))

    elif relation in ("near", "next_to"):
        dist = ((t_cx - a_cx)**2 + (t_cy - a_cy)**2) ** 0.5
        return max(0, 1 - dist / img_diag)

    elif relation == "inside":
        inside_x = (target_box[0] >= anchor_box[0]) and (target_box[2] <= anchor_box[2])
        inside_y = (target_box[1] >= anchor_box[1]) and (target_box[3] <= anchor_box[3])
        return float(inside_x and inside_y)

    return 0.5


# ---------------------------------------------------------------------------
# 3) Grounding DINO + SAM Pipeline
# ---------------------------------------------------------------------------

def load_grounding_dino(config_path: str, checkpoint_path: str,
                        bert_path: Optional[str], device: str):
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(config_path)
    args.device = device
    if bert_path:
        args.bert_base_uncased_path = bert_path
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def detect_objects(model, image_pil: Image.Image, text_prompt: str,
                   box_threshold: float = 0.3, text_threshold: float = 0.25,
                   device: str = "cuda") -> Tuple[torch.Tensor, List[float]]:
    """
    Run Grounding DINO and return boxes in [x1, y1, x2, y2] pixel coords
    along with confidence scores.
    """
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_tensor, _ = transform(image_pil, None)

    caption = text_prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    W, H = image_pil.size
    scores = []
    for logit in logits_filt:
        scores.append(logit.max().item())

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    return boxes_filt, scores


def segment_with_sam(sam_predictor, image_np: np.ndarray,
                     box: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """Run SAM on a single box and return the best binary mask."""
    sam_predictor.set_image(image_np)
    transformed_box = sam_predictor.transform.apply_boxes_torch(
        box.unsqueeze(0), image_np.shape[:2]
    ).to(device)

    masks, scores, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_box,
        multimask_output=True,
    )

    best_idx = scores.argmax()
    return masks[best_idx].cpu().numpy().squeeze()


# ---------------------------------------------------------------------------
# 4) Main Pipeline: Complex Referring Expression Segmentation
# ---------------------------------------------------------------------------

def referring_segment(image_path: str, complex_sentence: str,
                      grounding_model, sam_predictor,
                      box_threshold: float = 0.25,
                      text_threshold: float = 0.2,
                      device: str = "cuda") -> Tuple[Optional[np.ndarray], Dict]:
    """
    Full pipeline:
      1. Decompose complex sentence
      2. Detect target + anchor objects with Grounding DINO
      3. Score candidates by spatial constraints
      4. Segment best candidate with SAM
    """
    parsed = decompose_sentence(complex_sentence)
    print(f"\n{'='*60}")
    print(f"Input: {complex_sentence}")
    print(f"Parsed: {json.dumps(parsed, indent=2)}")
    print(f"{'='*60}\n")

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    target_query = parsed["target"]
    target_boxes, target_scores = detect_objects(
        grounding_model, image_pil, target_query,
        box_threshold, text_threshold, device
    )
    print(f"[Target] '{target_query}': found {len(target_boxes)} candidates")

    if len(target_boxes) == 0:
        print("No target candidates found. Try lowering thresholds.")
        return None, parsed

    anchor_detections = {}
    for rel in parsed["spatial_relations"]:
        anchor_name = rel["anchor"]
        anchor_boxes, anchor_scores = detect_objects(
            grounding_model, image_pil, anchor_name,
            box_threshold, text_threshold, device
        )
        anchor_detections[anchor_name] = (anchor_boxes, anchor_scores, rel["relation"])
        print(f"[Anchor] '{anchor_name}' ({rel['relation']}): found {len(anchor_boxes)} detections")

    best_score = -1
    best_idx = 0

    for i, (t_box, t_conf) in enumerate(zip(target_boxes, target_scores)):
        spatial_score = 1.0

        for anchor_name, (a_boxes, a_scores, relation) in anchor_detections.items():
            if len(a_boxes) == 0:
                spatial_score *= 0.1
                continue

            best_rel_score = 0
            for a_box in a_boxes:
                rel_score = check_spatial_relation(t_box, a_box, relation)
                best_rel_score = max(best_rel_score, rel_score)

            spatial_score *= best_rel_score

        combined_score = t_conf * 0.4 + spatial_score * 0.6
        print(f"  Candidate {i}: conf={t_conf:.3f}, spatial={spatial_score:.3f}, "
              f"combined={combined_score:.3f}")

        if combined_score > best_score:
            best_score = combined_score
            best_idx = i

    print(f"\nSelected candidate {best_idx} (score={best_score:.3f})")

    best_box = target_boxes[best_idx]
    mask = segment_with_sam(sam_predictor, image_np, best_box, device)

    return mask, parsed


# ---------------------------------------------------------------------------
# 5) Visualization
# ---------------------------------------------------------------------------

def visualize_result(image_path: str, mask: np.ndarray,
                     sentence: str, output_path: str = "output_referring_seg.png"):
    image = np.array(Image.open(image_path).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    color_mask = np.zeros_like(image, dtype=np.float32)
    color_mask[mask > 0] = [255, 50, 50]
    blended = (image * 0.6 + color_mask * 0.4).clip(0, 255).astype(np.uint8)
    axes[1].imshow(blended)
    axes[1].set_title("Segmentation Overlay", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Binary Mask", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(f'Query: "{sentence}"', fontsize=10, y=0.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Result saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Complex Referring Expression Segmentation")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True,
                        help="Complex referring expression")
    parser.add_argument("--output_path", type=str, default="output_referring_seg.png")
    parser.add_argument("--grounding_config", type=str,
                        default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_checkpoint", type=str,
                        default="groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="sam_vit_h_4b8939.pth")
    parser.add_argument("--sam_version", type=str, default="vit_h")
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading Grounding DINO...")
    grounding_model = load_grounding_dino(
        args.grounding_config, args.grounding_checkpoint,
        args.bert_path, args.device
    )

    print("Loading SAM...")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint)
    sam = sam.to(args.device)
    sam_predictor = SamPredictor(sam)

    mask, parsed = referring_segment(
        args.image_path, args.text_prompt,
        grounding_model, sam_predictor,
        args.box_threshold, args.text_threshold,
        args.device
    )

    if mask is not None:
        visualize_result(args.image_path, mask, args.text_prompt, args.output_path)

        mask_save_path = args.output_path.replace(".png", "_mask.png")
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_img.save(mask_save_path)
        print(f"Binary mask saved to {mask_save_path}")
    else:
        print("Failed to segment target object.")


if __name__ == "__main__":
    main()
