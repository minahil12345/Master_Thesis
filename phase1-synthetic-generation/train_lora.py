#!/usr/bin/env python3
"""
train_lora.py – Modular LoRA fine‑tuning for Stable Diffusion Inpainting.
Can be used as a CLI tool or imported as a module.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from PIL import Image
import gc
import sys

# --------------------------------
# Dataset (unchanged)
# --------------------------------
class DefectInpaintingDataset(Dataset):
    def __init__(self, pair_dir):
        self.pair_dir = Path(pair_dir)
        self.masked_files = sorted(self.pair_dir.glob("*_masked.png"))

    def __len__(self):
        return len(self.masked_files)

    def __getitem__(self, idx):
        base_name = self.masked_files[idx].stem.replace("_masked", "")
        masked_path = self.pair_dir / f"{base_name}_masked.png"
        image_path = self.pair_dir / f"{base_name}_image.png"
        mask_path = self.pair_dir / f"{base_name}_mask.png"

        masked_img = Image.open(masked_path).convert("RGB")
        target_img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        masked_img = masked_img.resize((512, 512), Image.LANCZOS)
        target_img = target_img.resize((512, 512), Image.LANCZOS)
        mask = mask.resize((512, 512), Image.NEAREST)

        mask = (np.array(mask) > 128).astype(np.float32)

        masked_img = torch.from_numpy(np.array(masked_img)).permute(2,0,1).float() / 255.0
        target_img = torch.from_numpy(np.array(target_img)).permute(2,0,1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        masked_img = masked_img * 2.0 - 1.0
        target_img = target_img * 2.0 - 1.0

        return {"masked": masked_img, "target": target_img, "mask": mask}


# --------------------------------
# Helper: find linear attention modules
# --------------------------------
def find_attention_linear_modules(model):
    """
    Scan a model (usually UNet) and return a list of module names that are
    linear layers belonging to attention blocks.
    """
    target_names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(pattern in name for pattern in ["to_q", "to_k", "to_v", "to_out", "q_proj", "k_proj", "v_proj", "out_proj"]):
            target_names.append(name)
    return target_names


# --------------------------------
# Core training function (reusable)
# --------------------------------
def train_lora(fruit, defect_class, training_config, device="cuda", pairs_root="lora_training_pairs", lora_root="LoRAs"):
    """
    Train a LoRA adapter for a specific fruit and defect class.

    Args:
        fruit (str): e.g., "mango"
        defect_class (str): e.g., "fungal_infection"
        training_config (dict): with keys: lora_r, lora_alpha, lora_dropout,
                                learning_rate, batch_size, gradient_accumulation_steps, max_steps
        device (str): "cuda" or "cpu"
        pairs_root (str): root directory containing fruit/defect_class pairs
        lora_root (str): output directory for LoRA weights
    Returns:
        int: number of training pairs used, or 0 if no data
    """
    training_config = {
        "lora_r": int(training_config.get("lora_r", 8)),
        "lora_alpha": int(training_config.get("lora_alpha", 16)),
        "lora_dropout": float(training_config.get("lora_dropout", 0.0)),
        "learning_rate": float(training_config.get("learning_rate", 5e-6)),
        "batch_size": int(training_config.get("batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 4)),
        "max_steps": int(training_config.get("max_steps", 400)),
    }
    
    pair_dir = Path(pairs_root) / fruit / defect_class
    if not pair_dir.exists() or len(list(pair_dir.glob("*_masked.png"))) == 0:
        print(f"No training pairs for {fruit}/{defect_class}, skipping.")
        return 0

    dataset = DefectInpaintingDataset(pair_dir)
    print(f"Training LoRA for {fruit}/{defect_class} with {len(dataset)} pairs.")

    # Load components
    print("Loading VAE, text encoder, tokenizer, UNet...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="vae", torch_dtype=torch.float16
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="text_encoder", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="unet", torch_dtype=torch.float32
    )

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # LoRA configuration
    target_modules = find_attention_linear_modules(unet)
    if not target_modules:
        raise RuntimeError("No linear attention modules found in UNet. Cannot apply LoRA.")
    print(f"Target LoRA modules: {target_modules}")

    lora_config = LoraConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=training_config["lora_dropout"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Optimizer, dataloader, scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=training_config["learning_rate"])
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    noise_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")

    # Prompt
    try:
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        prompt = kb.get_prompt(fruit, defect_class, "surface")
    except ImportError:
        prompt = f"a high quality photo of a {fruit} with {defect_class.replace('_', ' ')}, sharp focus, natural lighting"
        print(f"Using fallback prompt: {prompt}")

    # Training loop
    unet.train()
    global_step = 0
    max_steps = training_config["max_steps"]
    grad_acc_steps = training_config["gradient_accumulation_steps"]
    progress = tqdm(total=max_steps, desc=f"  {fruit}/{defect_class}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    save_dir = Path(lora_root) / f"{fruit}_{defect_class}"
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad()

    while global_step < max_steps:
        for batch in dataloader:
            masked = batch["masked"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.float32)
            bsz = masked.shape[0]

            # Encode prompt
            text_inputs = tokenizer(
                [prompt] * bsz,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True
            )
            text_ids = text_inputs.input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_ids)[0]

            # Encode images
            with torch.no_grad():
                target_vae = target.to(dtype=torch.float16)
                masked_vae = masked.to(dtype=torch.float16)
                target_latents = vae.encode(target_vae).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(masked_vae).latent_dist.sample() * vae.config.scaling_factor
                target_latents = target_latents.float()
                masked_latents = masked_latents.float()

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

            latent_mask = F.interpolate(mask, size=target_latents.shape[2:], mode="nearest")
            unet_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                noise_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise)

            if torch.isnan(loss):
                print(f"  Warning: NaN loss at step {global_step}, skipping step.")
                continue

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            progress.update(1)
            progress.set_postfix(loss=loss.item())
            global_step += 1

            if global_step % 100 == 0:
                checkpoint_path = save_dir / f"checkpoint_step_{global_step}"
                unet.save_pretrained(checkpoint_path)
                print(f"  Saved checkpoint to {checkpoint_path}")

            if global_step >= max_steps:
                break

    progress.close()
    unet.save_pretrained(save_dir)
    print(f"  Final LoRA weights saved to {save_dir}")

    # Cleanup
    del vae, text_encoder, unet, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return len(dataset)


# --------------------------------
# Batch training function
# --------------------------------
def train_all(config_path="config.yaml", pairs_root="lora_training_pairs", lora_root="LoRAs"):
    """
    Train LoRA adapters for all fruit/defect combinations defined in config.
    """
    default_cfg = {
        "fruits": ["apple", "citrus", "mango"],
        "defect_classes": ["bruise", "fungal_infection", "discoloration", "deformation"],
        "training": {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "learning_rate": 5e-6,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_steps": 400,
        },
    }

    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = default_cfg

    fruits = cfg.get("fruits", default_cfg["fruits"])
    defect_classes = cfg.get("defect_classes", default_cfg["defect_classes"])
    training_cfg = cfg.get("training", default_cfg["training"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    total_pairs = 0
    for fruit in fruits:
        for defect in defect_classes:
            num = train_lora(fruit, defect, training_cfg, device, pairs_root, lora_root)
            total_pairs += num
    print(f"Completed training for all classes. Total pairs processed: {total_pairs}")


# --------------------------------
# CLI entry point (unchanged behaviour)
# --------------------------------
if __name__ == "__main__":
    # Optional: accept config path as argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--pairs-root", type=str, default="lora_training_pairs", help="Root of training pairs")
    parser.add_argument("--lora-root", type=str, default="LoRAs", help="Output directory for LoRAs")
    args = parser.parse_args()
    train_all(config_path=args.config, pairs_root=args.pairs_root, lora_root=args.lora_root)