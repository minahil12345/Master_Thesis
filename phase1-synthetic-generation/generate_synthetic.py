#!/usr/bin/env python3
"""
generate_synthetic.py – Enhanced generation with defect‑forcing mechanisms.
Implements post‑generation similarity check and automatic retries.
"""

import torch
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline
from peft import PeftModel
import yaml
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json
import random
from knowledge_base import KnowledgeBase

# ----------------------------------------------------------------------
# Defect quality check (enhanced)
# ----------------------------------------------------------------------
def is_defect_region_bad(img_rgb: np.ndarray, mask: np.ndarray,
                         min_brightness: float = 30.0,
                         low_brightness_threshold: float = 50.0,
                         min_saturation: float = 10.0,
                         max_similarity_to_original: float = 5.0) -> tuple[bool, float]:
    """
    Returns (is_bad, difference_score).
    difference_score = mean absolute pixel difference inside mask
    (compared to a hypothetical perfect reconstruction – here we compare
     to the original image? We'll compute later using the original image)
    """
    defect_pixels = img_rgb[mask > 0]
    if len(defect_pixels) == 0:
        return True, 0.0

    mean_brightness = defect_pixels.mean()
    if mean_brightness < min_brightness:
        return True, mean_brightness

    if mean_brightness < low_brightness_threshold:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv_region = hsv[mask > 0]
        mean_sat = hsv_region[:, 1].mean()
        if mean_sat < min_saturation:
            return True, mean_sat

    return False, mean_brightness


def compare_masked_region(original_img: np.ndarray, generated_img: np.ndarray, mask: np.ndarray) -> float:
    """Return mean absolute difference (0‑255) inside the mask."""
    orig_masked = original_img[mask > 0]
    gen_masked = generated_img[mask > 0]
    if len(orig_masked) == 0:
        return 0.0
    return np.abs(orig_masked.astype(np.float32) - gen_masked.astype(np.float32)).mean()


# ----------------------------------------------------------------------
# Main generation function
# ----------------------------------------------------------------------
def generate_synthetic_defects(fruit: str, defect_class: str,
                               use_controlnet: bool = False,
                               device: str = "cuda",
                               lora_dir: str = "loras"):
    kb = KnowledgeBase()

    pair_dir = Path("inpainting_inputs") / fruit / defect_class
    if not pair_dir.exists():
        print(f"No inpainting pairs for {fruit}/{defect_class}, skipping.")
        return 0

    image_files = sorted(pair_dir.glob("pair_*_image.png"))
    if not image_files:
        print(f"No image files in {pair_dir}, skipping.")
        return 0

    # Load pipeline
    print(f"Loading inpainting pipeline for {fruit}/{defect_class}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    if use_controlnet:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=dtype
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)

    # Load LoRA if exists
    lora_path = Path(lora_dir) / f"{fruit}_{defect_class}"
    lora_loaded = False
    if lora_path.exists():
        try:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            # Optional: merge for faster inference (but may reduce quality)
            # pipe.unet = pipe.unet.merge_and_unload()
            pipe.unet.eval()
            lora_loaded = True
            print(f"  ✓ Loaded LoRA from {lora_path}")
            # Check number of trainable parameters
            trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
            print(f"  LoRA trainable params: {trainable_params:,}")
        except Exception as e:
            print(f"  ✗ LoRA load failed ({e}), using base model.")
    else:
        print(f"  No LoRA found at {lora_path}, using base model.")

    output_dir = Path("synthetic") / fruit / defect_class
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generation parameters – more aggressive
    neg_prompt = "blurry, distorted, unrealistic, bad anatomy, low quality, watermark, text, no change, identical"
    num_inference_steps = 50         
    guidance_scale = 9.0              
    if use_controlnet:
        controlnet_conditioning_scale = 0.9

    max_retries = 3
    total_generated = 0

    for img_path in tqdm(image_files, desc=f"  {fruit}/{defect_class}"):
        pair_id = img_path.stem.replace("_image", "")
        mask_path = img_path.parent / f"{pair_id}_mask.png"
        metadata_path = img_path.parent / f"{pair_id}_metadata.json"

        if not mask_path.exists() or not metadata_path.exists():
            continue

        # Load original healthy image and mask
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (512, 512))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask > 128).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_binary)

        # Check mask area
        mask_pixels = np.count_nonzero(mask_binary)
        if mask_pixels < 500:
            print(f"    Skipping {pair_id}: mask area too small ({mask_pixels} pixels)")
            continue

        # Load metadata and build prompt
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        region = meta.get("region_name", "surface")
        prompt = kb.get_prompt(fruit, defect_class, region)

        # Optionally strengthen the prompt for better defect generation
        if not lora_loaded:
            prompt += ", obvious visible defect, distinct from healthy fruit"

        # Try generating
        generated = False
        for attempt in range(max_retries):
            try:
                seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator(device=device).manual_seed(seed)

                # If this is a retry after a low‑difference result, increase guidance further
                current_guidance = guidance_scale + attempt * 1.0

                if use_controlnet:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=Image.fromarray(original_img),
                        mask_image=mask_pil,
                        control_image=mask_pil,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=current_guidance,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        generator=generator
                    ).images[0]
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=Image.fromarray(original_img),
                        mask_image=mask_pil,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=current_guidance,
                        generator=generator
                    ).images[0]

                result_np = np.array(result)

                # Check if defect is actually different from original
                diff_score = compare_masked_region(original_img, result_np, mask_binary)
                bad, _ = is_defect_region_bad(result_np, mask_binary)

                if bad or diff_score < 3.0:   # almost no change
                    print(f"    Attempt {attempt+1}: diff={diff_score:.2f} (too low), retrying...")
                    continue

                # Success
                out_name = f"img_{pair_id}.png"
                result.save(output_dir / out_name)
                cv2.imwrite(str(output_dir / f"img_{pair_id}_mask.png"), mask_binary)

                out_meta = {
                    "source_pair": str(img_path),
                    "prompt": prompt,
                    "region": region,
                    "seed": seed,
                    "attempt": attempt + 1,
                    "lora_used": lora_loaded,
                    "difference_score": float(diff_score)
                }
                with open(output_dir / f"img_{pair_id}_metadata.json", 'w') as f:
                    json.dump(out_meta, f, indent=2)

                total_generated += 1
                generated = True
                break

            except Exception as e:
                print(f"    Error on {pair_id} attempt {attempt+1}: {e}")
                continue

        if not generated:
            print(f"    Failed to generate a valid defect for {pair_id} after {max_retries} attempts.")

    print(f"  ✓ Generated {total_generated} valid images for {fruit}/{defect_class}")
    return total_generated


def generate_all_synthetic(use_controlnet: bool = False, lora_dir: str = "LoRAs"):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    fruits = cfg["fruits"]
    defect_classes = cfg["defect_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = 0
    for fruit in fruits:
        for dc in defect_classes:
            total += generate_synthetic_defects(fruit, dc,
                                                use_controlnet=use_controlnet,
                                                device=device,
                                                lora_dir=lora_dir)
    print(f"\nTotal images generated across all classes: {total}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-controlnet", action="store_true")
    parser.add_argument("--lora-dir", type=str, default="LoRAs")
    args = parser.parse_args()
    generate_all_synthetic(use_controlnet=args.use_controlnet, lora_dir=args.lora_dir)