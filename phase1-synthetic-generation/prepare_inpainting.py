#!/usr/bin/env python3
"""
Inpainting Pair Preparation Module (Updated)
Creates training pairs from healthy images, spatial priors, AND the fruit silhouette mask.
The final mask = (transformed spatial prior) ∩ (fruit mask).
Saves binary masks (0 or 255) and stores region_name in metadata.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import random
import logging
import json
from typing import Tuple
from knowledge_base import KnowledgeBase


MIN_MASK_PIXELS = 500          # minimum number of white pixels in the final mask
MAX_TRANSFORM_ATTEMPTS = 10    # retries per spatial prior mask
MAX_MASK_SWITCHES = 3          # number of different spatial masks to try

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prepare_inpainting.log'),
        logging.StreamHandler()
    ]
)


def load_fruit_mask(healthy_img_path: Path) -> np.ndarray:
    """
    Load the fruit silhouette mask for the given healthy image.
    Returns a binary mask (0 or 255) resized to 512x512.
    If mask not found, returns a white mask (full fruit).
    """
    fruit = healthy_img_path.parent.parent.name
    stem = healthy_img_path.stem
    mask_path = Path(f"data/{fruit}/healthy_fruit_masks/{stem}_fmask.png")
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            # Ensure binary (0 or 255)
            mask = (mask > 128).astype(np.uint8) * 255
            return mask
    logging.warning(f"Fruit mask not found for {healthy_img_path}, using full canvas.")
    return np.ones((512, 512), dtype=np.uint8) * 255


def transform_mask(mask: np.ndarray,
                   rotation_range: float = 30,
                   scale_range: Tuple[float, float] = (0.8, 1.2),
                   translation_range: float = 0.2) -> Tuple[np.ndarray, dict]:
    """
    Apply random rotation, scaling, translation to a binary mask.
    Returns transformed mask (binary 0/255) and transform parameters.
    """
    h, w = mask.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-rotation_range, rotation_range)
    scale = random.uniform(*scale_range)
    tx = random.uniform(-translation_range * w, translation_range * w)
    ty = random.uniform(-translation_range * h, translation_range * h)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    transformed = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    # Soften edges a bit, then threshold to binary
    transformed = cv2.GaussianBlur(transformed, (5, 5), 2)
    transformed = (transformed > 127).astype(np.uint8) * 255

    params = {
        "rotation_degrees": angle,
        "scale_factor": scale,
        "translation_x": tx,
        "translation_y": ty
    }
    return transformed, params

def create_inpainting_pair(healthy_img_path: Path, mask_path: str, output_dir: Path,
                           pair_id: str, kb: KnowledgeBase, mask_entries: list) -> bool:
    """
    Creates one inpainting pair, retrying transformations until the mask area
    is at least MIN_MASK_PIXELS.
    """
    # Load healthy image (unchanged)
    healthy_img = cv2.imread(str(healthy_img_path))
    if healthy_img is None:
        logging.error(f"Could not load image: {healthy_img_path}")
        return False
    healthy_img = cv2.cvtColor(healthy_img, cv2.COLOR_BGR2RGB)
    healthy_img = cv2.resize(healthy_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Load fruit mask once
    fruit_mask = load_fruit_mask(healthy_img_path)

    # Try different spatial prior masks if needed
    for mask_switch in range(MAX_MASK_SWITCHES):
        # If this is not the first attempt, pick a different mask
        if mask_switch > 0:
            # Choose another mask entry (may be the same if only one exists)
            new_entry = random.choice(mask_entries)
            if isinstance(new_entry, dict):
                mask_path = new_entry["path"]
            else:
                mask_path = new_entry
            if not Path(mask_path).exists():
                continue

        # Load base mask
        base_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if base_mask is None:
            logging.error(f"Could not load mask: {mask_path}")
            continue
        base_mask = cv2.resize(base_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        base_mask = (base_mask > 128).astype(np.uint8) * 255

        # Try multiple random transformations on this mask
        for attempt in range(MAX_TRANSFORM_ATTEMPTS):
            transformed_mask, transform_params = transform_mask(base_mask)
            final_mask = cv2.bitwise_and(transformed_mask, fruit_mask)
            area = np.count_nonzero(final_mask)
            

            if area >= MIN_MASK_PIXELS:
                # Success – save pair
                img_out = output_dir / f"{pair_id}_image.png"
                mask_out = output_dir / f"{pair_id}_mask.png"
                cv2.imwrite(str(img_out), cv2.cvtColor(healthy_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_out), final_mask)

                # Determine region name
                region_name = "surface"
                for entry in mask_entries:
                    if isinstance(entry, dict) and entry.get("path") == mask_path:
                        region_name = entry.get("region_name", "surface")
                        break
                    elif isinstance(entry, str) and entry == mask_path:
                        region_name = "surface"
                        break

                metadata = {
                    "source_healthy": str(healthy_img_path),
                    "source_mask": mask_path,
                    "transform_params": transform_params,
                    "fruit_mask_used": bool(np.any(fruit_mask < 255)),
                    "region_name": region_name
                }
                metadata_out = output_dir / f"{pair_id}_metadata.json"
                with open(metadata_out, 'w') as f:
                    json.dump(metadata, f, indent=2)
                return True

            # If area too small, continue loop for another transformation
            # (no logging unless verbose)

        # If all transformations failed for this mask, try a different mask
        logging.warning(f"    Mask {mask_path} never reached minimum area after {MAX_TRANSFORM_ATTEMPTS} attempts, trying another mask.")

    # If all masks and all transformations fail, skip this pair
    logging.warning(f"    Could not create acceptable mask for {healthy_img_path} after {MAX_MASK_SWITCHES} masks and {MAX_TRANSFORM_ATTEMPTS} transforms each. Skipping.")
    return False


def prepare_all_inpainting_pairs():
    logging.info("=" * 70)
    logging.info("Preparing Inpainting Input Pairs (fruit‑surface constrained)")
    logging.info("=" * 70)

    config_path = Path("dataset_config.yaml")
    if not config_path.exists():
        logging.error("dataset_config.yaml not found. Run data_prep.py first.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    kb = KnowledgeBase()
    output_base = Path("inpainting_inputs")
    total_pairs = 0

    for fruit in config["fruits"]:
        logging.info(f"\nProcessing {fruit.upper()}...")
        healthy_images = config["fruits"][fruit].get("healthy", [])
        defect_classes = list(config["fruits"][fruit].get("defect_train", {}).keys())

        if not healthy_images:
            logging.warning(f"  No healthy images found for {fruit}, skipping.")
            continue

        logging.info(f"  Healthy images available: {len(healthy_images)}")

        for defect_class in defect_classes:
            logging.info(f"  Creating pairs for: {defect_class}")
            try:
                kb_data = kb.load_kb(fruit)
                if defect_class not in kb_data:
                    logging.warning(f"    {defect_class} not in knowledge base, skipping")
                    continue

                mask_entries = kb_data[defect_class].get("spatial_prior_mask_paths", [])
                if not mask_entries:
                    logging.warning(f"    No spatial prior masks defined for {fruit}/{defect_class}, skipping")
                    continue

                pair_output_dir = output_base / fruit / defect_class
                pair_output_dir.mkdir(parents=True, exist_ok=True)
                pairs_created = 0

                for img_idx, healthy_img_path in enumerate(healthy_images):
                    healthy_img_path = Path(healthy_img_path)
                    num_variations = 3
                    for var_idx in range(num_variations):
                        # Randomly pick one spatial mask for this pair
                        mask_entry = random.choice(mask_entries)
                        if isinstance(mask_entry, dict):
                            mask_path = mask_entry["path"]
                        else:
                            mask_path = mask_entry

                        if not Path(mask_path).exists():
                            logging.warning(f"    Mask not found: {mask_path}")
                            continue

                        pair_id = f"pair_{img_idx:03d}_{var_idx:02d}"
                        if create_inpainting_pair(healthy_img_path, mask_path, pair_output_dir,
                                                  pair_id, kb, mask_entries):
                            pairs_created += 1
                            total_pairs += 1

                logging.info(f"    Created {pairs_created} inpainting pairs")
            except Exception as e:
                logging.error(f"    Error processing {defect_class}: {e}")
                continue

    logging.info("\n" + "=" * 70)
    logging.info("Inpainting Pair Preparation Complete")
    logging.info("=" * 70)
    logging.info(f"Total pairs created: {total_pairs}")
    logging.info(f"Output location: {output_base}")


if __name__ == "__main__":
    prepare_all_inpainting_pairs()