"""
mask_and_selection.py
----------------------
Mask generation + score-based selection for DreamMask-style NSS pipeline.

Input dataset structure (after layout-to-image generation, e.g., using LayoutGPT):

    layouts_dataset/
      scene_00000.html
      scene_00000.png        # layout visualization (boxes)
      real_00000.png         # realistic synthesized image
      scene_00000.json       # metadata with bounding boxes
      scene_00001.html
      scene_00001.png
      real_00001.png
      scene_00001.json
      ...

This script:
  1. Loads each scene_XXXXX.json in layouts_dataset.
  2. Reads corresponding real_XXXXX.png.
  3. Uses SAM to generate an instance mask per object bounding box.
  4. Uses CLIP to compute an image-text similarity score per object.
  5. Applies score-based selection (CLIP + mask uncertainty) to keep only
     high-quality object samples.
  6. Writes a processed dataset under processed_dataset/ with:

        processed_dataset/
          images/
            real_00000.png
            real_00001.png
            ...
          masks/
            scene_00000_obj_00.png
            scene_00000_obj_01.png
            ...
          scene_00000.json
          scene_00001.json
          ...

     Each processed scene JSON contains only the selected objects with
     their mask paths, CLIP scores, and mask uncertainty.
"""

import os
import json
import shutil
from typing import Dict, Any, List

import numpy as np
import cv2
from PIL import Image

import torch
import clip
from segment_anything import sam_model_registry, SamPredictor


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

LAYOUTS_DIR = "layouts_dataset"
OUTPUT_DIR = "processed_dataset"

SAM_CHECKPOINT = "./weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"  # 'vit_h', 'vit_l', etc. depending on your checkpoint

CLIP_MODEL_NAME = "ViT-B/32"

# Thresholds for score-based selection
CLIP_THRESHOLD = 0.25         # keep objects with image-text similarity >= this
UNCERTAINTY_THRESHOLD = 0.30  # keep objects with mask_uncertainty <= this


# ------------------------------------------------------------
# Helpers: load SAM, CLIP
# ------------------------------------------------------------

def load_sam(checkpoint_path: str, model_type: str = "vit_h") -> SamPredictor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint_path}. "
            f"Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )
    print(f"[INFO] Loading SAM ({model_type}) from {checkpoint_path} ...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"[INFO] SAM loaded on device: {device}")
    return predictor


def load_clip(model_name: str = "ViT-B/32"):
    print(f"[INFO] Loading CLIP model {model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    print(f"[INFO] CLIP loaded on device: {device}")
    return model, preprocess, device


# ------------------------------------------------------------
# Core functions
# ------------------------------------------------------------

def compute_clip_score_for_object(
    image_rgb: np.ndarray,
    box: Dict[str, Any],
    category_name: str,
    clip_model,
    clip_preprocess,
    device: str,
) -> float:
    """
    Compute a CLIP image-text similarity score for a single object.

    We crop the bounding box region from the real image, feed it to CLIP
    along with the object category name.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    H, W = image_rgb.shape[:2]

    # Clamp box to image boundaries
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    crop = image_rgb[y : y + h, x : x + w, :]  # RGB
    pil = Image.fromarray(crop)

    image_input = clip_preprocess(pil).unsqueeze(0).to(device)
    text_input = clip.tokenize([category_name]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

    return float(similarity)


def generate_mask_with_sam(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box: Dict[str, Any],
) -> (np.ndarray, float):
    """
    Generate an instance mask for a bounding box using SAM.

    Returns:
        mask: np.ndarray of shape (H, W), bool
        mask_uncertainty: float in [0, 1], lower is better
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    H, W = image_rgb.shape[:2]

    # Clamp box to image boundaries
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    box_xyxy = np.array([x, y, x + w, y + h])

    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_xyxy[None, :],
        multimask_output=True,
    )

    # Choose the best mask by score
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]  # bool mask, shape (H, W)

    # A simple proxy for uncertainty: 1 - best_score (scores in [0, 1])
    best_score = float(scores[best_idx])
    mask_uncertainty = 1.0 - best_score

    return best_mask, mask_uncertainty


# ------------------------------------------------------------
# Main processing pipeline
# ------------------------------------------------------------

def process_dataset(
    layouts_dir: str = LAYOUTS_DIR,
    output_dir: str = OUTPUT_DIR,
    sam_checkpoint: str = SAM_CHECKPOINT,
    sam_model_type: str = SAM_MODEL_TYPE,
    clip_model_name: str = CLIP_MODEL_NAME,
    clip_threshold: float = CLIP_THRESHOLD,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> None:
    """
    Run mask generation + score-based selection over all scenes in layouts_dataset.
    """
    if not os.path.isdir(layouts_dir):
        raise FileNotFoundError(
            f"Layouts directory '{layouts_dir}' not found. Make sure it exists and follows the expected structure."
        )

    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Load models
    sam_predictor = load_sam(sam_checkpoint, sam_model_type)
    clip_model, clip_preprocess, device = load_clip(clip_model_name)

    scene_files = sorted([f for f in os.listdir(layouts_dir) if f.endswith(".json")])

    print(f"[INFO] Found {len(scene_files)} scene JSON files in {layouts_dir}.")

    num_scenes_kept = 0
    num_objects_kept = 0

    for json_name in scene_files:
        scene_id = json_name.replace(".json", "")  # e.g., scene_00000
        json_path = os.path.join(layouts_dir, json_name)

        # Expected real image name: real_XXXXX.png
        index_str = scene_id.replace("scene_", "")
        real_name = f"real_{index_str}.png"
        real_path = os.path.join(layouts_dir, real_name)

        if not os.path.exists(real_path):
            print(f"[WARN] Real image not found for {scene_id}: {real_path}. Skipping.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        boxes = meta.get("boxes", [])
        if not boxes:
            print(f"[WARN] No boxes found in {json_path}. Skipping.")
            continue

        # Load real image (RGB)
        bgr = cv2.imread(real_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Failed to read image {real_path}. Skipping.")
            continue
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        selected_objects: List[Dict[str, Any]] = []

        print(f"[INFO] Processing {scene_id} with {len(boxes)} objects...")
        for obj_idx, box in enumerate(boxes):
            category_name = box.get("name", "object")

            # 1) SAM mask
            try:
                mask, mask_uncertainty = generate_mask_with_sam(
                    sam_predictor,
                    image_rgb,
                    box,
                )
            except Exception as e:
                print(f"[WARN] SAM failed on {scene_id} obj {obj_idx}: {e}")
                continue

            # 2) CLIP score
            try:
                clip_score = compute_clip_score_for_object(
                    image_rgb,
                    box,
                    category_name,
                    clip_model,
                    clip_preprocess,
                    device,
                )
            except Exception as e:
                print(f"[WARN] CLIP failed on {scene_id} obj {obj_idx}: {e}")
                continue

            # 3) Score-based selection
            if clip_score < clip_threshold or mask_uncertainty > uncertainty_threshold:
                continue

            # 4) Save mask as PNG (0/255)
            mask_uint8 = (mask.astype("uint8") * 255)
            mask_filename = f"{scene_id}_obj_{obj_idx:02d}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            Image.fromarray(mask_uint8).save(mask_path)

            selected_objects.append(
                {
                    "index": obj_idx,
                    "name": category_name,
                    "box": {
                        "x": box["x"],
                        "y": box["y"],
                        "w": box["w"],
                        "h": box["h"],
                    },
                    "mask_path": os.path.relpath(mask_path, output_dir),
                    "clip_score": clip_score,
                    "mask_uncertainty": mask_uncertainty,
                }
            )

        if not selected_objects:
            print(f"[INFO] No objects passed selection for {scene_id}. Scene skipped.")
            continue

        # Copy real image into processed_dataset/images/
        out_real_name = real_name
        out_real_path = os.path.join(images_dir, out_real_name)
        if not os.path.exists(out_real_path):
            shutil.copy2(real_path, out_real_path)

        # Write processed scene JSON
        processed_scene = {
            "scene_id": scene_id,
            "source_json": os.path.relpath(json_path, output_dir),
            "real_image": os.path.relpath(out_real_path, output_dir),
            "canvas_width": meta.get("canvas_width"),
            "canvas_height": meta.get("canvas_height"),
            "objects": selected_objects,
        }

        out_scene_json = os.path.join(output_dir, f"{scene_id}.json")
        with open(out_scene_json, "w", encoding="utf-8") as f:
            json.dump(processed_scene, f, indent=2)

        num_scenes_kept += 1
        num_objects_kept += len(selected_objects)
        print(
            f"[INFO] Scene {scene_id}: kept {len(selected_objects)} objects. "
            f"Processed scene JSON saved to {out_scene_json}."
        )

    print("""    [INFO] Done.
  Scenes kept:   {}
  Objects kept:  {}
  Output dir:    {}
""".format(num_scenes_kept, num_objects_kept, os.path.abspath(output_dir)))


if __name__ == "__main__":
    process_dataset()
