#!/usr/bin/env python3
"""Post-hoc filtering of generated synthetic images using domain constraints."""

import cv2
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from knowledge_base import KnowledgeBase
import shutil

def filter_synthetic_images():
    """Filter all generated synthetic images using knowledge base validation."""
    
    kb = KnowledgeBase()
    synthetic_dir = Path("synthetic")
    final_dir = Path("final_dataset")
    
    results = []
    stats = {}
    
    print("\nFiltering synthetic images...")
    
    for fruit_dir in sorted(synthetic_dir.iterdir()):
        if not fruit_dir.is_dir():
            continue
        
        fruit = fruit_dir.name
        stats[fruit] = {}
        
        for defect_dir in sorted(fruit_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            
            defect_class = defect_dir.name
            print(f"\n{fruit}/{defect_class}:")
            
            output_dir = final_dir / fruit / defect_class
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect generated images (exclude masks and metadata)
            generated_images = sorted(defect_dir.glob("img_*.png"))
            generated_images = [img for img in generated_images 
                                if not img.name.endswith("_mask.png") 
                                and not img.name.endswith("_metadata.json")]
            
            accepted = 0
            rejected = 0
            
            for img_path in tqdm(generated_images, desc="  Filtering"):
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    rejected += 1
                    results.append({
                        "image_path": str(img_path),
                        "fruit": fruit,
                        "defect_class": defect_class,
                        "passed_filters": False,
                        "filter_details": "Failed to load image"
                    })
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load corresponding mask (saved by generate_synthetic.py with _mask suffix)
                mask_path = img_path.parent / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    rejected += 1
                    results.append({
                        "image_path": str(img_path),
                        "fruit": fruit,
                        "defect_class": defect_class,
                        "passed_filters": False,
                        "filter_details": "Mask file not found"
                    })
                    continue
                
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    rejected += 1
                    results.append({
                        "image_path": str(img_path),
                        "fruit": fruit,
                        "defect_class": defect_class,
                        "passed_filters": False,
                        "filter_details": "Failed to load mask"
                    })
                    continue
                
                # Validate using the knowledge base
                try:
                    valid, reason = kb.validate_defect(fruit, defect_class, img_rgb, mask)
                    if valid:
                        # Copy to final dataset
                        final_img_path = output_dir / img_path.name
                        shutil.copy2(img_path, final_img_path)
                        final_mask_path = output_dir / mask_path.name
                        shutil.copy2(mask_path, final_mask_path)
                        
                        # Also copy metadata if available
                        metadata_path = img_path.parent / f"{img_path.stem}_metadata.json"
                        if metadata_path.exists():
                            final_metadata_path = output_dir / metadata_path.name
                            shutil.copy2(metadata_path, final_metadata_path)
                        
                        accepted += 1
                        filter_details = f"Passed: {reason}"
                    else:
                        rejected += 1
                        filter_details = f"Failed: {reason}"
                    
                    results.append({
                        "image_path": str(final_img_path if valid else img_path),
                        "fruit": fruit,
                        "defect_class": defect_class,
                        "passed_filters": valid,
                        "filter_details": filter_details
                    })
                except Exception as e:
                    rejected += 1
                    results.append({
                        "image_path": str(img_path),
                        "fruit": fruit,
                        "defect_class": defect_class,
                        "passed_filters": False,
                        "filter_details": f"Validation error: {str(e)}"
                    })
            
            total = len(generated_images)
            stats[fruit][defect_class] = {
                "total": total,
                "accepted": accepted,
                "rejected": rejected,
                "acceptance_rate": accepted / total if total > 0 else 0
            }
            if total > 0:
                print(f"    Accepted: {accepted}/{total} ({100*accepted/total:.1f}%)")
    
    # Save manifest and stats
    df = pd.DataFrame(results)
    df.to_csv("dataset_manifest.csv", index=False)
    print(f"\n✓ Dataset manifest saved to dataset_manifest.csv")
    
    with open("filtering_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Filtering statistics saved to filtering_stats.json")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for fruit in stats:
        print(f"\n{fruit.upper()}:")
        for defect_class in stats[fruit]:
            s = stats[fruit][defect_class]
            print(f"  {defect_class}:")
            print(f"    Total generated: {s['total']}")
            print(f"    Accepted: {s['accepted']}")
            print(f"    Rejected: {s['rejected']}")
            print(f"    Acceptance rate: {100*s['acceptance_rate']:.1f}%")
    
    print(f"\n✓ Final dataset available in final_dataset/")

if __name__ == "__main__":
    filter_synthetic_images()