#!/usr/bin/env python3
"""
prepare_defect_lora_pairs.py
Creates training pairs for LoRA fine‑tuning from few‑shot real defect images.
Masks are expected in a sibling directory named mask_<defect_class>
with filenames: resized_<original_stem>_mask.png
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

FILL_COLOR = (128, 128, 128)  # neutral gray for inpainting void

def create_lora_pairs(fruit, defect_class, img_dir, pair_dst_dir):
    """For each defect image, look for its mask in ../mask_<defect_class>/.
    Mask filename pattern: resized_{stem}_mask.png"""
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    # Skip any pre‑resized images or mask files
    image_files = [f for f in image_files 
                   if not f.name.startswith("resized_") 
                   and not f.name.endswith("_mask.png")]

    mask_dir = img_dir.parent / f"mask_{defect_class}"
    if not mask_dir.exists():
        print(f"  Warning: mask directory {mask_dir} does not exist. Skipping {img_dir}")
        return 0

    pairs_created = 0
    for img_path in image_files:
        # Expected mask name: resized_{original_stem}_mask.png
        mask_path = mask_dir / f"resized_{img_path.stem}_mask.png"
        if not mask_path.exists():
            print(f"  Skipping {img_path.name}: mask not found (expected {mask_path.name})")
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  Failed to load {img_path.name} or its mask")
            continue

        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        masked_img = img.copy()
        masked_img[mask > 0] = FILL_COLOR

        pair_id = img_path.stem
        out_img = pair_dst_dir / f"{pair_id}_image.png"
        out_mask = pair_dst_dir / f"{pair_id}_mask.png"
        out_masked = pair_dst_dir / f"{pair_id}_masked.png"

        cv2.imwrite(str(out_img), img)
        cv2.imwrite(str(out_mask), mask)
        cv2.imwrite(str(out_masked), masked_img)
        pairs_created += 1

    return pairs_created

def main():
    print("Preparing LoRA training pairs from few‑shot defect images...")
    with open("dataset_config.yaml", 'r') as f:
        ds = yaml.safe_load(f)

    base_out = Path("lora_training_pairs")
    total = 0
    for fruit in ds["fruits"]:
        defect_train = ds["fruits"][fruit]["defect_train"]
        for defect_class, paths in defect_train.items():
            if not paths:
                continue
            img_dir = Path(f"data/{fruit}/defect_train/{defect_class}")
            if not img_dir.exists():
                continue
            dst_dir = base_out / fruit / defect_class
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{fruit}/{defect_class}:")
            created = create_lora_pairs(fruit, defect_class, img_dir, dst_dir)
            print(f"  Created {created} pairs")
            total += created
    print(f"\nTotal LoRA training pairs created: {total}")

if __name__ == "__main__":
    main()