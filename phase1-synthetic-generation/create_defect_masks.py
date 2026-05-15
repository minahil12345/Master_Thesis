#!/usr/bin/env python3
"""
Interactive mask creation using Segment Anything Model (SAM).
For each defect image, click points on the defect (left mouse) and
on background (right mouse). Press 'm' to generate mask; 'r' to reset;
'space' to save and move to next image.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from segment_anything import sam_model_registry, SamPredictor
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Create defect masks with SAM")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Folder containing defect images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Folder to save masks (default: same as input)")
    parser.add_argument("--sam-checkpoint", type=str,
                        default="sam_vit_b_01ec64.pth",
                        help="Path to SAM checkpoint")
    parser.add_argument("--model-type", type=str, default="vit_b",
                        help="SAM model type (vit_b, vit_l, vit_h)")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM model on {device}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Collect images (avoid processing existing masks)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    image_files = [f for f in image_files if not f.name.endswith("_mask.png")]
    resized_images = [f for f in image_files if f.name.startswith("resized_")]
    if resized_images:
        print(f"Found {len(resized_images)} already resized images, skipping them.")
        image_files = [f for f in image_files if not f.name.startswith("resized_")]

    for img_path in image_files:
        mask_path = output_dir / f"resized_{img_path.stem}_mask.png"
        if mask_path.exists():
            print(f"Mask already exists for {img_path.name}, skipping.")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read {img_path}, skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        input_points = []
        input_labels = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal input_points, input_labels
            if event == cv2.EVENT_LBUTTONDOWN:
                input_points.append([x, y])
                input_labels.append(1)  # positive
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                input_points.append([x, y])
                input_labels.append(0)  # negative
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

        display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(f"Mask Creator - {img_path.name}")
        cv2.setMouseCallback(f"Mask Creator - {img_path.name}", mouse_callback)

        print(f"\nProcessing: {img_path.name}")
        print("  Left  click = defect (green dot)")
        print("  Right click = background (red dot)")
        print("  Press 'm' to compute mask")
        print("  Press 'r' to reset points")
        print("  Press 'space' to save mask and go to next")
        print("  Press 'q' to quit without saving")

        mask = None
        while True:
            cv2.imshow(f"Mask Creator - {img_path.name}", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                input_points = []
                input_labels = []
                display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mask = None
                print("  Points reset.")

            elif key == ord('m') and len(input_points) > 0:
                points = np.array(input_points)
                labels = np.array(input_labels)
                with torch.no_grad():
                    masks, scores, _ = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=False,  # only best mask
                    )
                mask = masks[0].astype(np.uint8) * 255
                # Overlay mask
                overlay = np.zeros_like(display_img)
                overlay[mask > 0] = [0, 255, 0]  # green overlay
                display_img = cv2.addWeighted(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.7, overlay, 0.3, 0
                )
                print(f"  Mask generated (confidence: {scores[0]:.3f})")

            elif key == ord(' ') and mask is not None:
                cv2.imwrite(str(mask_path), mask)
                print(f"  Mask saved to {mask_path}")
                break

            elif key == ord(' '):
                print("  No mask generated yet. Press 'm' first.")
            elif key == ord('q'):
                print("  Quitting without saving.")
                break

        cv2.destroyWindow(f"Mask Creator - {img_path.name}")
        if key == ord('q'):
            break

    print("\nAll done.")

if __name__ == "__main__":
    main()