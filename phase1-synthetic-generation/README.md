# Phase 1: Knowledge-Guided Synthetic Fruit Defect Generation

A complete pipeline for generating semantically faithful synthetic fruit defect images using Vision Foundation Models (VFMs) with domain knowledge constraints.

## Overview

This pipeline implements:
- **Domain Knowledge Integration**: Spatial priors, anatomical constraints, and visual feature rules
- **ControlNet-Based Inpainting**: Stable Diffusion with optional ControlNet for controlled defect generation
- **LoRA Fine-Tuning**: Parameter-efficient adaptation for defect-specific generation
- **Post-Hoc Filtering**: Automated validation against biological plausibility rules

## Requirements

- Python 3.10+
- CUDA 12.1+ with GPU (≥16 GB VRAM recommended)
- 50+ GB disk space for models and generated data

## Quick Start

### 1. Setup Environment

```bash
# Clone or download this repository
cd phase1-synthetic-generation

# Run setup (creates venv and installs dependencies)
python3 setup.py

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 2. Prepare Your Data

Organize your data in the following structure:
data/
├── apple/
│   ├── healthy/              # Healthy apple images
│   ├── defect_train/
│   │   ├── bruise/          # 1-5 bruised apple images
│   │   └── scab/            # 1-5 scab images
│   └── defect_test/
│       ├── bruise/          # Test set for evaluation
│       └── scab/
├── citrus/
│   └── ...
└── mango/
└── ...

### 3. Configure Knowledge Base

After first run, the pipeline will generate example knowledge base files:
knowledge_bases/
├── apple_kb.yaml
├── citrus_kb.yaml
└── mango_kb.yaml

**Important**: You must create corresponding mask images referenced in the YAML files:
knowledge_bases/masks/
├── apple_bruise_cheek.png        # Binary mask (white=allowed region)
├── apple_bruise_stem.png
├── apple_forbidden_core.png      # Binary mask (white=forbidden region)
└── ...

**Mask Creation Tips**:
- Masks should be 512×512 PNG images
- Spatial priors: White (255) indicates likely defect locations
- Forbidden regions: White (255) indicates prohibited areas
- Use image editing tools (GIMP, Photoshop, or Python/OpenCV)

### 4. Run the Complete Pipeline

```bash
# Run all steps with default settings
bash run_all.sh

# Custom configuration
bash run_all.sh --fruits "apple citrus" --num-synthetic 20

# Skip LoRA training (faster, uses base model)
bash run_all.sh --skip-train

# Use ControlNet (better quality, requires more VRAM)
bash run_all.sh --use-controlnet
```

## Pipeline Steps

### Step 1: Data Preparation (`data_prep.py`)
- Resizes all images to 512×512
- Creates `dataset_config.yaml` with train/test splits

### Step 2: Knowledge Base Setup (`knowledge_base.py`)
- Generates example YAML files
- Validates domain constraints during generation

### Step 3: Inpainting Pairs (`prepare_inpainting.py`)
- Creates training pairs from healthy images and spatial priors
- Applies random transformations to masks

### Step 4: LoRA Training (`train_lora.py`)
- Fine-tunes Stable Diffusion UNet with LoRA adapters
- Trains separate adapters per fruit-defect combination
- Saves weights to `loras/`

### Step 5: Synthetic Generation (`generate_synthetic.py`)
- Generates N synthetic defects per healthy image
- Uses knowledge-base prompts and spatial guidance
- Saves images with metadata in `synthetic/`

### Step 6: Filtering (`filter_and_finalize.py`)
- Validates each image against domain constraints
- Checks HSV color ranges, shape constraints, forbidden regions
- Moves accepted images to `final_dataset/`

## Output Files
final_dataset/              # Validated synthetic images
├── apple/
│   ├── bruise/
│   │   ├── img_000_000.png
│   │   ├── img_000_000_mask.png
│   │   └── img_000_000_metadata.json
│   └── scab/
dataset_manifest.csv        # Complete dataset inventory
filtering_stats.json        # Acceptance rates per class

## Configuration Options

### Knowledge Base YAML Structure

```yaml
bruise:
  spatial_prior_mask_paths:
    - path: "knowledge_bases/masks/apple_bruise_cheek.png"
      region_name: "cheek"
  forbidden_region_masks:
    - "knowledge_bases/masks/apple_forbidden_core.png"
  colour_hsv:
    hue_min: 0
    hue_max: 30
    sat_min: 40
    sat_max: 200
    val_min: 30
    val_max: 150
  shape:
    min_area: 500
    max_area: 15000
    min_aspect_ratio: 0.5
    max_aspect_ratio: 2.5
  prompt_template: "A fresh apple with a brown bruise on the {location}, realistic product photography"
```

### Generation Parameters

Edit `generate_synthetic.py` to adjust:
- `num_inference_steps`: Diffusion steps (default: 25)
- `guidance_scale`: Prompt adherence (default: 8.0)
- `controlnet_conditioning_scale`: Control strength (default: 0.9)

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size in `train_lora.py`
- Use `--skip-train` to avoid training
- Don't use `--use-controlnet` flag

### Low Acceptance Rates
- Relax constraints in knowledge base YAML files
- Check mask quality and coverage
- Verify HSV ranges match your fruit variety

### Missing Masks
- Ensure all mask paths in YAML files exist
- Create masks using the template structure
- Masks must be binary (0 or 255 values)

## Advanced Usage

### Run Individual Steps

```bash
# Only generate synthetic images (assumes previous steps done)
python generate_synthetic.py --num-per-image 20

# Only filter existing synthetic images
python filter_and_finalize.py

# Re-train LoRA for specific class
# (Edit train_lora.py to specify fruit/defect)
python train_lora.py
```

### Custom Validation Logic

Modify `knowledge_base.py` `validate_defect()` method to add custom checks:
```python
def validate_defect(self, fruit, defect_class, generated_image, mask):
    # Add custom validation logic here
    # Return True/False
```

## Performance Expectations

- **LoRA Training**: ~30-60 min per defect class (on RTX 3090)
- **Generation**: ~3-5 sec per image (25 inference steps)
- **Filtering**: ~0.1 sec per image
- **Total Pipeline**: 2-4 hours for 3 fruits × 2 defects × 10 synthetic/image

## Citation

If you use this pipeline in your research, please cite:
@misc{phase1-synthetic-defects,
title={Knowledge-Guided Synthetic Data Generation for Agricultural Defect Detection},
author={Research Methodology in Computing and Engineering},
year={2025}
}

## License

Academic and research use only. See LICENSE file for details.

## Support

For issues and questions:
1. Check this README and inline code comments
2. Review knowledge base YAML structure
3. Verify data directory structure matches expected format
4. Check GPU memory availability

---

**Next Steps**: After completing Phase 1, proceed to Phase 2 (Few-Shot Adaptation) using the generated synthetic dataset.