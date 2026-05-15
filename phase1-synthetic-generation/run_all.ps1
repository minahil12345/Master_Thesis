<#
.SYNOPSIS
    Orchestration script for Phase 1 synthetic data generation pipeline.
    Now includes fruit‑mask existence check and uses the corrected generation/filtering.
#>
param(
    [string]$Fruits = "apple citrus mango",
    [int]$NumSynthetic = 5,
    [switch]$UseControlNet = $false,
    [switch]$SkipSetup = $false,
    [switch]$TrainLora = $false,
    [switch]$Help = $false
)

if ($Help) {
    Get-Help $PSCommandPath -Detailed
    exit 0
}

$ErrorActionPreference = 'Stop'

Write-Host "========================================" -ForegroundColor Blue
Write-Host "Phase 1: Synthetic Data Generation" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Fruits: $Fruits"
Write-Host "  Synthetic per healthy: $NumSynthetic"
Write-Host "  Use ControlNet: $UseControlNet"
Write-Host "  Skip setup: $SkipSetup"
Write-Host "  Train LoRA: $TrainLora"
Write-Host ""

# ------------------------------------------------------------
# 0. Virtual environment activation
# ------------------------------------------------------------
$venvPath = ".venv"
if (-not (Test-Path $venvPath)) {
    if (-not $SkipSetup) {
        Write-Host "Virtual environment not found. Running setup..." -ForegroundColor Yellow
        python setup.py
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }
    else {
        Write-Host "Virtual environment not found and -SkipSetup specified. Exiting." -ForegroundColor Red
        exit 1
    }
}
& "$venvPath\Scripts\Activate.ps1"

# ------------------------------------------------------------
# 1. Data Preparation
# ------------------------------------------------------------
Write-Host "[1/8] Data Preparation" -ForegroundColor Blue
if (Test-Path "dataset_config.yaml") {
    Write-Host "dataset_config.yaml exists. Skipping." -ForegroundColor Yellow
}
else {
    python data_prep.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# ------------------------------------------------------------
# 2. Knowledge Base Setup
# ------------------------------------------------------------
Write-Host "[2/8] Knowledge Base Setup" -ForegroundColor Blue
$kbYamls = @("knowledge_bases/apple_kb.yaml", "knowledge_bases/citrus_kb.yaml", "knowledge_bases/mango_kb.yaml")
$kbMissing = ($kbYamls | Where-Object { -not (Test-Path $_) }).Count -gt 0
if ($kbMissing) {
    python create_knowledge_bases.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
else {
    Write-Host "Knowledge bases found." -ForegroundColor Green
}

# ------------------------------------------------------------
# 3. Fruit Silhouette Masks (interactive, must exist before step 4)
# ------------------------------------------------------------
Write-Host "[3/8] Fruit Silhouette Masks" -ForegroundColor Blue
Write-Host "This step creates fruit masks used to keep defects on the fruit surface." -ForegroundColor Yellow
Write-Host "For each fruit in `$Fruits`, you need to run:" -ForegroundColor Yellow
Write-Host "   python create_fruit_masks.py --input-dir data/{fruit}/healthy --output-dir data/{fruit}/healthy_fruit_masks" -ForegroundColor White
Write-Host ""

# Check that at least some fruit masks exist for the specified fruits
$allMasksExist = $true
foreach ($fruit in $Fruits.Split(' ')) {
    $healthyDir = "data/$fruit/healthy"
    if (Test-Path $healthyDir) {
        $images = Get-ChildItem $healthyDir -Filter *.jpg -ErrorAction SilentlyContinue
        $masksDir = "data/$fruit/healthy_fruit_masks"
        if (-not (Test-Path $masksDir)) {
            $allMasksExist = $false
            break
        }
        foreach ($img in $images) {
            $expectedMask = "data/$fruit/healthy_fruit_masks/$($img.BaseName)_fmask.png"
            if (-not (Test-Path $expectedMask)) {
                $allMasksExist = $false
                break
            }
        }
    }
}

if (-not $allMasksExist) {
    Write-Host "Fruit masks missing or incomplete. Please run create_fruit_masks.py for each fruit before continuing." -ForegroundColor Red
    Write-Host "Example: python create_fruit_masks.py --input-dir data/apple/healthy --output-dir data/apple/healthy_fruit_masks" -ForegroundColor White
    exit 1
}
Write-Host "Fruit masks found." -ForegroundColor Green

# ------------------------------------------------------------
# 4. Inpainting Input Pairs (uses fruit masks, spatial priors)
# ------------------------------------------------------------
Write-Host "[4/8] Preparing Inpainting Input Pairs" -ForegroundColor Blue
python prepare_inpainting.py
if ($LASTEXITCODE -ne 0) { exit 1 }

# ------------------------------------------------------------
# 5. Optional LoRA Training Pairs
# ------------------------------------------------------------
if ($TrainLora) {
    Write-Host "[5/8] Preparing LoRA Training Pairs (defect images)" -ForegroundColor Blue
    python prepare_defect_lora_pairs.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
else {
    Write-Host "[5/8] LoRA training pair preparation skipped (use -TrainLora to enable)." -ForegroundColor Yellow
}

# ------------------------------------------------------------
# 6. Optional LoRA Fine‑Tuning
# ------------------------------------------------------------
Write-Host "[6/8] LoRA Fine‑Tuning" -ForegroundColor Blue
if ($TrainLora) {
    python train_lora.py
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
else {
    Write-Host "Skipped (use -TrainLora to train). Will use base model." -ForegroundColor Yellow
}

# ------------------------------------------------------------
# 7. Synthetic Defect Generation (uses fruit masks, updated script)
# ------------------------------------------------------------
Write-Host "[7/8] Generating Synthetic Defect Images" -ForegroundColor Blue
$genArgs = "--num-per-image $NumSynthetic"
if ($UseControlNet) { $genArgs += " --use-controlnet" }
python generate_synthetic.py $genArgs.Split()
if ($LASTEXITCODE -ne 0) { exit 1 }

# ------------------------------------------------------------
# 8. Filtering & Final Dataset
# ------------------------------------------------------------
Write-Host "[8/8] Filtering and Finalizing Dataset" -ForegroundColor Blue
python filter_and_finalize.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Pipeline Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Output: synthetic/, final_dataset/, dataset_manifest.csv"