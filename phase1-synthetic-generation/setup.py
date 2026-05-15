#!/usr/bin/env python3
"""
Setup script for Phase 1: Knowledge-Guided Synthetic Data Generation Pipeline
Thesis: Bridging the Domain Gap for Agricultural Fruit Quality Inspection
Author: Manahil Faisal (TP090458)
"""

import subprocess
import sys
from pathlib import Path
import platform

def setup_environment():
    """Create virtual environment and install dependencies."""
    
    print("="*70)
    print("Phase 1: Synthetic Data Generation Pipeline - Setup")
    print("="*70)
    print()
    
    # Create virtual environment
    print("Step 1: Creating virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("  Virtual environment already exists. Skipping creation.")
    else:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("  ✓ Virtual environment created")
    
    # Determine pip and python paths
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Upgrade pip
    print("\nStep 2: Upgrading pip...")
    # subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    print("  ✓ Pip upgraded")
    
    # Install requirements
    print("\nStep 3: Installing dependencies...")
    print("  This may take several minutes...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    print("  ✓ Dependencies installed")
    
    # Create directory structure
    print("\nStep 4: Creating directory structure...")
    directories = [
        # Data directories
        "data/apple/healthy",
        "data/apple/defect_train/bruise",
        "data/apple/defect_train/fungal_infection",
        "data/apple/defect_train/discoloration",
        "data/apple/defect_train/deformation",
        "data/apple/defect_test/bruise",
        "data/apple/defect_test/fungal_infection",
        "data/apple/defect_test/discoloration",
        "data/apple/defect_test/deformation",
        
        "data/citrus/healthy",
        "data/citrus/defect_train/bruise",
        "data/citrus/defect_train/fungal_infection",
        "data/citrus/defect_train/discoloration",
        "data/citrus/defect_train/deformation",
        "data/citrus/defect_test/bruise",
        "data/citrus/defect_test/fungal_infection",
        "data/citrus/defect_test/discoloration",
        "data/citrus/defect_test/deformation",
        
        "data/mango/healthy",
        "data/mango/defect_train/bruise",
        "data/mango/defect_train/fungal_infection",
        "data/mango/defect_train/discoloration",
        "data/mango/defect_train/deformation",
        "data/mango/defect_test/bruise",
        "data/mango/defect_test/fungal_infection",
        "data/mango/defect_test/discoloration",
        "data/mango/defect_test/deformation",
        
        # Knowledge base directories
        "knowledge_bases/masks/apple",
        "knowledge_bases/masks/citrus",
        "knowledge_bases/masks/mango",
        
        # Processing directories
        "inpainting_inputs",
        "loras",
        "synthetic",
        "final_dataset",
        "logs",
        "checkpoints"
    ]
    
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"  ✓ Created {len(directories)} directories")
    
    # Create initial configuration
    print("\nStep 5: Creating initial configuration files...")
    create_initial_config()
    print("  ✓ Configuration files created")
    
    print("\n" + "="*70)
    print("Setup Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("     venv\\Scripts\\activate")
    else:
        print("     source venv/bin/activate")
    print("\n  2. Place your fruit images in data/{fruit}/healthy/ and data/{fruit}/defect_train/")
    print("\n  3. Run: python create_knowledge_bases.py")
    print("\n  4. Run: bash run_all.sh (or python run_all.py on Windows)")
    print()

def create_initial_config():
    """Create initial configuration files."""
    
    # Create a sample config file
    config_content = """# Phase 1 Configuration
# Thesis: Bridging the Domain Gap for Agricultural Fruit Quality Inspection

# Target fruits (do not modify - thesis scope)
fruits:
  - apple
  - citrus
  - mango

# Defect classes (do not modify - thesis scope)
defect_classes:
  - bruise              # Mechanical Damage (Bruise/Impact)
  - fungal_infection    # Fungal/Microbial Infection
  - discoloration       # Color Discoloration/Unevenness
  - deformation         # Shape/Structural Deformation

# Generation parameters
generation:
  num_synthetic_per_healthy: 10
  image_size: 512
  num_inference_steps: 25
  guidance_scale: 8.0
  controlnet_conditioning_scale: 0.9

# Training parameters
training:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  learning_rate: 1.0e-5
  batch_size: 2
  gradient_accumulation_steps: 4
  max_steps: 400
  
# Validation thresholds (can be adjusted per fruit)
validation:
  min_acceptance_rate: 0.5  # 50% minimum acceptance rate
"""
    
    with open("config.yaml", "w") as f:
        f.write(config_content)

if __name__ == "__main__":
    setup_environment()