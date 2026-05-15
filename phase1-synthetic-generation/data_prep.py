#!/usr/bin/env python3
"""
Data Preparation Module - Phase 1
Resize images and create dataset configuration
"""

import yaml
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_prep.log'),
        logging.StreamHandler()
    ]
)

TARGET_SIZE = (512, 512)
FRUITS = ["apple", "citrus", "mango"]
DEFECT_CLASSES = ["bruise", "fungal_infection", "discoloration", "deformation"]

def resize_and_normalize_image(img_path, output_path, size=TARGET_SIZE):
    """Resize image to target size and save.
    
    Args:
        img_path: Input image path
        output_path: Output image path
        size: Target size (width, height)
    
    Returns:
        Path to resized image
    """
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path, quality=95)
        return output_path
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None

def prepare_dataset():
    """Scan data folders, resize images, and create dataset config."""
    
    logging.info("="*70)
    logging.info("Phase 1: Data Preparation")
    logging.info("="*70)
    
    config = {"fruits": {}}
    total_images = 0
    
    for fruit in FRUITS:
        logging.info(f"\nProcessing {fruit.upper()}...")
        config["fruits"][fruit] = {
            "healthy": [],
            "defect_train": {},
            "defect_test": {}
        }
        
        # Process healthy images
        healthy_dir = Path(f"data/{fruit}/healthy")
        if healthy_dir.exists():
            healthy_images = list(healthy_dir.glob("*.jpg")) + \
                           list(healthy_dir.glob("*.png")) + \
                           list(healthy_dir.glob("*.jpeg"))
            
            logging.info(f"  Found {len(healthy_images)} healthy images")
            
            for img_path in tqdm(healthy_images, desc=f"  Resizing healthy {fruit}"):
                if img_path.name.startswith("resized_"):
                    # Already processed
                    config["fruits"][fruit]["healthy"].append(str(img_path))
                    total_images += 1
                else:
                    output_path = img_path.parent / f"resized_{img_path.name}"
                    if resize_and_normalize_image(img_path, output_path):
                        config["fruits"][fruit]["healthy"].append(str(output_path))
                        total_images += 1
        else:
            logging.warning(f"  No healthy directory found for {fruit}")
        
        # Process defect training images
        for defect_class in DEFECT_CLASSES:
            train_dir = Path(f"data/{fruit}/defect_train/{defect_class}")
            
            if train_dir.exists():
                defect_images = list(train_dir.glob("*.jpg")) + \
                              list(train_dir.glob("*.png")) + \
                              list(train_dir.glob("*.jpeg"))
                
                logging.info(f"  Found {len(defect_images)} {defect_class} training images")
                
                if defect_class not in config["fruits"][fruit]["defect_train"]:
                    config["fruits"][fruit]["defect_train"][defect_class] = []
                
                for img_path in tqdm(defect_images, desc=f"  Resizing {defect_class}"):
                    if img_path.name.startswith("resized_"):
                        config["fruits"][fruit]["defect_train"][defect_class].append(str(img_path))
                        total_images += 1
                    else:
                        output_path = img_path.parent / f"resized_{img_path.name}"
                        if resize_and_normalize_image(img_path, output_path):
                            config["fruits"][fruit]["defect_train"][defect_class].append(str(output_path))
                            total_images += 1
        
        # Process defect test images
        for defect_class in DEFECT_CLASSES:
            test_dir = Path(f"data/{fruit}/defect_test/{defect_class}")
            
            if test_dir.exists():
                defect_images = list(test_dir.glob("*.jpg")) + \
                              list(test_dir.glob("*.png")) + \
                              list(test_dir.glob("*.jpeg"))
                
                logging.info(f"  Found {len(defect_images)} {defect_class} test images")
                
                if defect_class not in config["fruits"][fruit]["defect_test"]:
                    config["fruits"][fruit]["defect_test"][defect_class] = []
                
                for img_path in defect_images:
                    if img_path.name.startswith("resized_"):
                        config["fruits"][fruit]["defect_test"][defect_class].append(str(img_path))
                        total_images += 1
                    else:
                        output_path = img_path.parent / f"resized_{img_path.name}"
                        if resize_and_normalize_image(img_path, output_path):
                            config["fruits"][fruit]["defect_test"][defect_class].append(str(output_path))
                            total_images += 1
    
    # Save configuration
    with open("dataset_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logging.info("\n" + "="*70)
    logging.info("Data Preparation Complete")
    logging.info("="*70)
    logging.info(f"Total images processed: {total_images}")
    logging.info("Dataset configuration saved to: dataset_config.yaml")
    
    # Print detailed summary
    logging.info("\nDataset Summary:")
    for fruit in FRUITS:
        logging.info(f"\n{fruit.upper()}:")
        logging.info(f"  Healthy images: {len(config['fruits'][fruit]['healthy'])}")
        
        logging.info(f"  Training defects:")
        for defect in DEFECT_CLASSES:
            count = len(config['fruits'][fruit]['defect_train'].get(defect, []))
            logging.info(f"    - {defect}: {count} images")
        
        logging.info(f"  Test defects:")
        for defect in DEFECT_CLASSES:
            count = len(config['fruits'][fruit]['defect_test'].get(defect, []))
            logging.info(f"    - {defect}: {count} images")
    
    return config

if __name__ == "__main__":
    prepare_dataset()