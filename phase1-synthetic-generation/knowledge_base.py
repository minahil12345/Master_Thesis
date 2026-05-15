#!/usr/bin/env python3
"""
Knowledge Base Module - Domain Knowledge Management
Implements semantic constraints and validation for agricultural defects
"""

import yaml
import cv2
import numpy as np
from pathlib import Path
import random
import logging
from typing import Dict, Tuple, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class KnowledgeBase:
    """
    Manages domain-specific knowledge for fruit defects.
    Implements the knowledge integration module as specified in thesis objectives.
    """
    
    def __init__(self, kb_dir="knowledge_bases"):
        self.kb_dir = Path(kb_dir)
        self.kb_cache = {}
        self.validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "failure_reasons": {}
        }
    
    def load_kb(self, fruit: str) -> Dict:
        """Load knowledge base YAML for a fruit.
        
        Args:
            fruit: Fruit name (apple, citrus, mango)
            
        Returns:
            Knowledge base dictionary
        """
        if fruit in self.kb_cache:
            return self.kb_cache[fruit]
        
        kb_path = self.kb_dir / f"{fruit}_kb.yaml"
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {kb_path}\n"
                f"Please run: python create_knowledge_bases.py"
            )
        
        with open(kb_path, 'r') as f:
            kb = yaml.safe_load(f)
        
        self.kb_cache[fruit] = kb
        logging.info(f"Loaded knowledge base for {fruit}")
        return kb
    
    def get_random_mask(self, fruit: str, defect_class: str) -> Tuple[str, str]:
        """Get a random spatial prior mask for a defect class.
        
        Args:
            fruit: Fruit name
            defect_class: Defect class name
            
        Returns:
            Tuple of (mask_path, region_name)
        """
        kb = self.load_kb(fruit)
        
        if defect_class not in kb:
            raise ValueError(
                f"Defect class '{defect_class}' not found for {fruit}. "
                f"Available classes: {list(kb.keys())}"
            )
        
        masks = kb[defect_class]["spatial_prior_mask_paths"]
        if not masks:
            raise ValueError(
                f"No spatial priors defined for {fruit}/{defect_class}"
            )
        
        # Select random mask
        mask_entry = random.choice(masks)
        
        if isinstance(mask_entry, dict):
            mask_path = mask_entry["path"]
            region_name = mask_entry["region_name"]
        else:
            # Fallback for simple path strings
            mask_path = mask_entry
            region_name = "surface"
        
        # Verify mask exists
        if not Path(mask_path).exists():
            raise FileNotFoundError(
                f"Spatial prior mask not found: {mask_path}\n"
                f"Please run: python create_knowledge_bases.py"
            )
        
        return mask_path, region_name
    
    def get_prompt(self, fruit: str, defect_class: str, location_name: str) -> str:
        """Get defect generation prompt with location filled in.
        
        Args:
            fruit: Fruit name
            defect_class: Defect class name
            location_name: Location name to insert
            
        Returns:
            Formatted prompt string
        """
        kb = self.load_kb(fruit)
        template = kb[defect_class]["prompt_template"]
        return template.format(location=location_name)
    
    def validate_defect(self, 
                       fruit: str, 
                       defect_class: str, 
                       generated_image: np.ndarray, 
                       mask: np.ndarray) -> Tuple[bool, str]:
        """Validate generated defect against domain constraints.
        
        Implements the multi-faceted validation framework specified in thesis.
        
        Args:
            fruit: Fruit name
            defect_class: Defect class name
            generated_image: RGB image array (H, W, 3), range [0, 255]
            mask: Binary mask array (H, W), values 0 or 255
            
        Returns:
            Tuple of (is_valid, failure_reason)
        """
        self.validation_stats["total_validations"] += 1
        
        kb = self.load_kb(fruit)
        rules = kb[defect_class]
        
        # Ensure proper types
        if generated_image.dtype != np.uint8:
            generated_image = (generated_image * 255).astype(np.uint8)
        
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Check 1: Mask coverage (defect must exist)
        mask_coverage = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
        if mask_coverage < 0.005:  # Less than 0.5% coverage
            self._record_failure("insufficient_mask_coverage")
            return False, "Mask coverage too small"
        
        # Check 2: Color HSV constraints
        if "colour_hsv" in rules:
            is_valid, reason = self._validate_color(
                generated_image, mask, rules["colour_hsv"]
            )
            if not is_valid:
                self._record_failure(f"color_constraint_{reason}")
                return False, f"Color constraint failed: {reason}"
        
        # Check 3: Shape constraints
        if "shape" in rules:
            is_valid, reason = self._validate_shape(mask, rules["shape"])
            if not is_valid:
                self._record_failure(f"shape_constraint_{reason}")
                return False, f"Shape constraint failed: {reason}"
        
        # Check 4: Forbidden regions
        if "forbidden_region_masks" in rules:
            is_valid, reason = self._validate_forbidden_regions(
                mask, rules["forbidden_region_masks"]
            )
            if not is_valid:
                self._record_failure("forbidden_region_violation")
                return False, f"Forbidden region violated: {reason}"
        
        # All checks passed
        self.validation_stats["passed"] += 1
        return True, "All constraints satisfied"
    
    def _validate_color(self, 
                       image: np.ndarray, 
                       mask: np.ndarray, 
                       hsv_rules: Dict) -> Tuple[bool, str]:
        """Validate HSV color constraints."""
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract defect region
        masked_region = hsv_image[mask > 127]
        
        if len(masked_region) == 0:
            return False, "empty_defect_region"
        
        # Calculate mean HSV values
        mean_hue = np.mean(masked_region[:, 0])
        mean_sat = np.mean(masked_region[:, 1])
        mean_val = np.mean(masked_region[:, 2])
        
        # Check hue range (with wrap-around for red hues)
        hue_min = hsv_rules["hue_min"]
        hue_max = hsv_rules["hue_max"]
        
        if hue_min <= hue_max:
            hue_valid = hue_min <= mean_hue <= hue_max
        else:
            # Wrap-around case (e.g., red: 170-10)
            hue_valid = mean_hue >= hue_min or mean_hue <= hue_max
        
        if not hue_valid:
            return False, f"hue_out_of_range_{mean_hue:.1f}"
        
        # Check saturation range
        if not (hsv_rules["sat_min"] <= mean_sat <= hsv_rules["sat_max"]):
            return False, f"saturation_out_of_range_{mean_sat:.1f}"
        
        # Check value range
        if not (hsv_rules["val_min"] <= mean_val <= hsv_rules["val_max"]):
            return False, f"value_out_of_range_{mean_val:.1f}"
        
        return True, "color_valid"
    
    def _validate_shape(self, mask: np.ndarray, shape_rules: Dict) -> Tuple[bool, str]:
        """Validate shape constraints."""
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return False, "no_contours_found"
        
        # Take largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check area constraints
        if area < shape_rules["min_area"]:
            return False, f"area_too_small_{area:.0f}"
        
        if area > shape_rules["max_area"]:
            return False, f"area_too_large_{area:.0f}"
        
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if h > 0:
            aspect_ratio = w / h
            
            if aspect_ratio < shape_rules["min_aspect_ratio"]:
                return False, f"aspect_ratio_too_small_{aspect_ratio:.2f}"
            
            if aspect_ratio > shape_rules["max_aspect_ratio"]:
                return False, f"aspect_ratio_too_large_{aspect_ratio:.2f}"
        
        return True, "shape_valid"
    
    def _validate_forbidden_regions(self, 
                                    mask: np.ndarray, 
                                    forbidden_masks: List[str]) -> Tuple[bool, str]:
        """Validate that defect doesn't overlap with forbidden regions."""
        
        for forbidden_mask_path in forbidden_masks:
            forbidden_path = Path(forbidden_mask_path)
            
            if not forbidden_path.exists():
                logging.warning(f"Forbidden mask not found: {forbidden_path}")
                continue
            
            # Load forbidden mask
            forbidden_mask = cv2.imread(str(forbidden_path), cv2.IMREAD_GRAYSCALE)
            
            if forbidden_mask is None:
                logging.warning(f"Could not load forbidden mask: {forbidden_path}")
                continue
            
            # Resize to match
            forbidden_mask = cv2.resize(
                forbidden_mask, 
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Check overlap
            overlap = cv2.bitwise_and(mask, forbidden_mask)
            overlap_pixels = np.sum(overlap > 127)
            
            # Allow very small overlap (up to 2% of defect area)
            defect_pixels = np.sum(mask > 127)
            overlap_ratio = overlap_pixels / defect_pixels if defect_pixels > 0 else 0
            
            if overlap_ratio > 0.02:  # More than 2% overlap
                return False, f"overlaps_forbidden_region_{overlap_ratio:.2%}"
        
        return True, "no_forbidden_overlap"
    
    def _record_failure(self, reason: str):
        """Record validation failure reason for statistics."""
        self.validation_stats["failed"] += 1
        
        if reason not in self.validation_stats["failure_reasons"]:
            self.validation_stats["failure_reasons"][reason] = 0
        
        self.validation_stats["failure_reasons"][reason] += 1
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics summary."""
        total = self.validation_stats["total_validations"]
        
        if total == 0:
            return {"message": "No validations performed yet"}
        
        stats = {
            "total_validations": total,
            "passed": self.validation_stats["passed"],
            "failed": self.validation_stats["failed"],
            "pass_rate": self.validation_stats["passed"] / total,
            "failure_reasons": self.validation_stats["failure_reasons"]
        }
        
        return stats
    
    def print_validation_summary(self):
        """Print validation statistics summary."""
        stats = self.get_validation_stats()
        
        if "message" in stats:
            print(stats["message"])
            return
        
        print("\n" + "="*70)
        print("VALIDATION STATISTICS")
        print("="*70)
        print(f"Total validations: {stats['total_validations']}")
        print(f"Passed: {stats['passed']} ({100*stats['pass_rate']:.1f}%)")
        print(f"Failed: {stats['failed']} ({100*(1-stats['pass_rate']):.1f}%)")
        
        if stats['failure_reasons']:
            print("\nFailure Breakdown:")
            for reason, count in sorted(
                stats['failure_reasons'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                percentage = 100 * count / stats['total_validations']
                print(f"  - {reason}: {count} ({percentage:.1f}%)")
        
        print("="*70)

if __name__ == "__main__":
    # Test knowledge base loading
    kb = KnowledgeBase()
    
    print("Testing Knowledge Base Module...")
    print()
    
    for fruit in ["apple", "citrus", "mango"]:
        try:
            data = kb.load_kb(fruit)
            print(f"✓ {fruit.upper()} knowledge base loaded")
            print(f"  Defect classes: {list(data.keys())}")
        except Exception as e:
            print(f"✗ Error loading {fruit}: {e}")
    
    print("\nKnowledge base module ready!")