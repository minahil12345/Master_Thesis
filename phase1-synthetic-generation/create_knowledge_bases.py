#!/usr/bin/env python3
"""
Enhanced knowledge base with hyper‑realistic prompts and refined colour rules.
Generates YAML definitions and spatial prior masks for apple, citrus, mango.
"""

import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List

class KnowledgeBaseCreator:
    def __init__(self):
        self.kb_dir = Path("knowledge_bases")
        self.mask_dir = self.kb_dir / "masks"
        self.kb_dir.mkdir(exist_ok=True)
        self.mask_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def create_all_knowledge_bases(self):
        print("Creating hyper‑realistic knowledge bases...")
        self.create_apple_kb()
        self.create_citrus_kb()
        self.create_mango_kb()
        self.create_all_masks()
        print("Complete.")

    # ------------------------------------------------------------------
    # Apple knowledge base (YAML)
    # ------------------------------------------------------------------
    def create_apple_kb(self):
        apple_kb = {
            "bruise": {
                "description": "Mechanical impact damage - brown discoloration from cellular breakdown, slightly sunken surface",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/apple/bruise_cheek.png", "region_name": "cheek"},
                    {"path": "knowledge_bases/masks/apple/bruise_stem.png", "region_name": "stem bowl area"},
                    {"path": "knowledge_bases/masks/apple/bruise_bottom.png", "region_name": "bottom calyx end"},
                    {"path": "knowledge_bases/masks/apple/bruise_shoulder.png", "region_name": "shoulder"}
                ],
                "forbidden_region_masks": ["knowledge_bases/masks/apple/forbidden_core.png"],
                "colour_hsv": {"hue_min": 5, "hue_max": 30, "sat_min": 40, "sat_max": 200, "val_min": 20, "val_max": 160},
                "shape": {"min_area": 400, "max_area": 20000, "min_aspect_ratio": 0.4, "max_aspect_ratio": 3.0},
                "prompt_template": "A fresh red apple with a clearly visible brown bruise on the {location}. The bruise is a distinct, well-defined discolored patch with a dark brown center fading to lighter brown edges, slightly sunken into the fruit surface. The damaged area contrasts sharply with the healthy red skin. Natural orchard lighting, macro photography, high detail, realistic fruit inspection."
            },
            "fungal_infection": {
                "description": "Apple scab (Venturia inaequalis) - dark olive to black rough lesions with corky texture",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/apple/fungal_surface.png", "region_name": "upper surface"},
                    {"path": "knowledge_bases/masks/apple/fungal_stem.png", "region_name": "near stem"},
                    {"path": "knowledge_bases/masks/apple/fungal_calyx.png", "region_name": "calyx basin"}
                ],
                "forbidden_region_masks": ["knowledge_bases/masks/apple/forbidden_core.png"],
                "colour_hsv": {"hue_min": 15, "hue_max": 45, "sat_min": 60, "sat_max": 180, "val_min": 15, "val_max": 120},
                "shape": {"min_area": 250, "max_area": 12000, "min_aspect_ratio": 0.5, "max_aspect_ratio": 2.5},
                "prompt_template": "A red apple infected with apple scab disease on the {location}. The infection shows dark olive to black, rough, corky lesions with irregular circular shape. The fungal spots are raised and textured, clearly different from the smooth healthy skin. Sharp focus, daylight illumination, agricultural quality assessment, high resolution."
            },
            "discoloration": {
                "description": "Sunburn or uneven ripening - yellow-brown patches from sun exposure or physiological disorder",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/apple/discolor_cheek.png", "region_name": "sun-exposed cheek"},
                    {"path": "knowledge_bases/masks/apple/discolor_surface.png", "region_name": "outer surface"},
                    {"path": "knowledge_bases/masks/apple/discolor_shoulder.png", "region_name": "shoulder area"}
                ],
                "forbidden_region_masks": [],
                "colour_hsv": {"hue_min": 12, "hue_max": 50, "sat_min": 25, "sat_max": 150, "val_min": 60, "val_max": 220},
                "shape": {"min_area": 800, "max_area": 28000, "min_aspect_ratio": 0.3, "max_aspect_ratio": 5.0},
                "prompt_template": "A red apple with a prominent sunburn discoloration patch on the {location}. The patch is yellow-brown with a soft, gradual transition from the normal red color, covering a large area. The discolored region looks bleached and uneven, typical of sun scald or uneven ripening. Natural sunlight, detailed fruit photography."
            },
            "deformation": {
                "description": "Physical deformation - dents, dimples, or irregular shape from growth or handling",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/apple/deform_side.png", "region_name": "side"},
                    {"path": "knowledge_bases/masks/apple/deform_bottom.png", "region_name": "bottom"},
                    {"path": "knowledge_bases/masks/apple/deform_shoulder.png", "region_name": "shoulder region"}
                ],
                "forbidden_region_masks": [],
                "colour_hsv": {"hue_min": 0, "hue_max": 35, "sat_min": 30, "sat_max": 190, "val_min": 40, "val_max": 210},
                "shape": {"min_area": 500, "max_area": 22000, "min_aspect_ratio": 0.35, "max_aspect_ratio": 3.8},
                "prompt_template": "A red apple with a visible dent and shape deformation on the {location}. The surface shows a clear indentation with soft shadows, creating an irregular contour. The skin remains intact but the fruit structure is depressed. Side lighting emphasizes the depth of the dent. Professional produce grading image."
            }
        }
        self._save_kb("apple_kb.yaml", apple_kb)

    # ------------------------------------------------------------------
    # Citrus knowledge base
    # ------------------------------------------------------------------
    def create_citrus_kb(self):
        citrus_kb = {
            "bruise": {
                "description": "Impact damage - brown soft spots with oil gland rupture in rind",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/citrus/bruise_surface.png", "region_name": "equatorial surface"},
                    {"path": "knowledge_bases/masks/citrus/bruise_side.png", "region_name": "side"},
                    {"path": "knowledge_bases/masks/citrus/bruise_top.png", "region_name": "top pole"}
                ],
                "forbidden_region_masks": ["knowledge_bases/masks/citrus/forbidden_stem.png"],
                "colour_hsv": {"hue_min": 10, "hue_max": 35, "sat_min": 45, "sat_max": 180, "val_min": 35, "val_max": 140},
                "shape": {"min_area": 400, "max_area": 14000, "min_aspect_ratio": 0.5, "max_aspect_ratio": 2.6},
                "prompt_template": "A fresh orange with a distinct brown bruise on the {location}. The bruise is a soft, sunken area with dark brown discoloration and ruptured oil glands visible on the rind. The damage contrasts sharply with the bright orange peel. Natural lighting, macro detail, realistic citrus defect."
            },
            "fungal_infection": {
                "description": "Citrus black spot or green mold - dark sunken lesions with spore formation",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/citrus/fungal_rind.png", "region_name": "rind surface"},
                    {"path": "knowledge_bases/masks/citrus/fungal_spot.png", "region_name": "outer peel"},
                    {"path": "knowledge_bases/masks/citrus/fungal_depression.png", "region_name": "sunken area"}
                ],
                "forbidden_region_masks": ["knowledge_bases/masks/citrus/forbidden_stem.png"],
                "colour_hsv": {"hue_min": 20, "hue_max": 50, "sat_min": 50, "sat_max": 190, "val_min": 20, "val_max": 100},
                "shape": {"min_area": 300, "max_area": 9000, "min_aspect_ratio": 0.6, "max_aspect_ratio": 2.0},
                "prompt_template": "An orange infected with citrus black spot disease on the {location}. The infection shows circular, dark brown to black lesions with a hard, sunken center and raised borders. The spots have a rough, scabby texture, clearly distinguishable from the healthy rind. Daylight, high-resolution agricultural photography."
            },
            "discoloration": {
                "description": "Rind blemish - uneven coloring from environmental stress or physiological disorder",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/citrus/discolor_surface.png", "region_name": "outer surface"},
                    {"path": "knowledge_bases/masks/citrus/discolor_large.png", "region_name": "large patch"},
                    {"path": "knowledge_bases/masks/citrus/discolor_equator.png", "region_name": "equatorial band"}
                ],
                "forbidden_region_masks": [],
                "colour_hsv": {"hue_min": 8, "hue_max": 48, "sat_min": 30, "sat_max": 150, "val_min": 75, "val_max": 220},
                "shape": {"min_area": 900, "max_area": 26000, "min_aspect_ratio": 0.3, "max_aspect_ratio": 4.2},
                "prompt_template": "An orange with uneven rind discoloration on the {location}. The patch is greenish-yellow to pale orange, covering a large area with soft, irregular edges. The discolored zone lacks the vibrant orange hue of the rest of the fruit, indicating a color break disorder. Diffused natural light, detailed grading image."
            },
            "deformation": {
                "description": "Shape irregularity - dents, flattened areas, or misshapen fruit",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/citrus/deform_side.png", "region_name": "side wall"},
                    {"path": "knowledge_bases/masks/citrus/deform_surface.png", "region_name": "surface"},
                    {"path": "knowledge_bases/masks/citrus/deform_pole.png", "region_name": "pole area"}
                ],
                "forbidden_region_masks": [],
                "colour_hsv": {"hue_min": 10, "hue_max": 40, "sat_min": 35, "sat_max": 190, "val_min": 60, "val_max": 215},
                "shape": {"min_area": 550, "max_area": 17000, "min_aspect_ratio": 0.4, "max_aspect_ratio": 3.2},
                "prompt_template": "An orange with a visible dent and shape deformation on the {location}. The fruit surface has a flat, indented area with subtle shadowing, creating an asymmetrical contour. The rind follows the depression, giving a misshapen appearance. Studio lighting to emphasize depth and texture."
            }
        }
        self._save_kb("citrus_kb.yaml", citrus_kb)

    # ------------------------------------------------------------------
    # Mango knowledge base
    # ------------------------------------------------------------------
    def create_mango_kb(self):
        mango_kb = {
            "bruise": {
                "description": "Impact bruise - brown to black discoloration from mechanical damage, flesh breakdown beneath skin",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/mango/bruise_cheek.png", "region_name": "cheek"},
                    {"path": "knowledge_bases/masks/mango/bruise_shoulder.png", "region_name": "shoulder"},
                    {"path": "knowledge_bases/masks/mango/bruise_belly.png", "region_name": "belly area"}
                ],
                "forbidden_region_masks": ["knowledge_bases/masks/mango/forbidden_seed.png"],
                "colour_hsv": {"hue_min": 12, "hue_max": 35, "sat_min": 50, "sat_max": 200, "val_min": 30, "val_max": 135},
                "shape": {"min_area": 500, "max_area": 18000, "min_aspect_ratio": 0.45, "max_aspect_ratio": 3.0},
                "prompt_template": "A ripe yellow mango with a prominent brown impact bruise on the {location}. The bruise is a dark brown, slightly depressed area with soft, collapsed tissue beneath the skin, contrasting sharply with the smooth yellow peel. The damaged region has a dull, matte appearance. Natural tropical light, macro detail, realistic post-harvest defect."
            },
            "fungal_infection": {
                "description": "Anthracnose (Colletotrichum gloeosporioides) - dark sunken lesions with concentric rings",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/mango/fungal_skin.png", "region_name": "skin surface"},
                    {"path": "knowledge_bases/masks/mango/fungal_spot.png", "region_name": "upper surface"},
                    {"path": "knowledge_bases/masks/mango/fungal_shoulder.png", "region_name": "shoulder area"}
                ],
                "forbidden_region_masks": ["knowledge_bases/masks/mango/forbidden_seed.png"],
                "colour_hsv": {"hue_min": 10, "hue_max": 38, "sat_min": 60, "sat_max": 185, "val_min": 20, "val_max": 105},
                "shape": {"min_area": 350, "max_area": 11000, "min_aspect_ratio": 0.5, "max_aspect_ratio": 2.4},
                "prompt_template": "A mango infected with anthracnose disease on the {location}. The infection shows dark brown to black sunken lesions with concentric rings and a velvety texture. The spots are clearly defined, spreading along the fruit surface, with cracked margins. Bright daylight, high-resolution pathology photography."
            },
            "discoloration": {
                "description": "Uneven ripening or sap burn - irregular color patches from physiological disorder",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/mango/discolor_surface.png", "region_name": "outer surface"},
                    {"path": "knowledge_bases/masks/mango/discolor_cheek.png", "region_name": "cheek area"},
                    {"path": "knowledge_bases/masks/mango/discolor_shoulder.png", "region_name": "shoulder"}
                ],
                "forbidden_region_masks": [],
                "colour_hsv": {"hue_min": 18, "hue_max": 55, "sat_min": 25, "sat_max": 160, "val_min": 70, "val_max": 225},
                "shape": {"min_area": 850, "max_area": 28000, "min_aspect_ratio": 0.3, "max_aspect_ratio": 4.5},
                "prompt_template": "A mango with uneven ripening discoloration on the {location}. The patch is green-yellow to pale orange, with irregular, blotchy edges and soft transitions. The discolored area lacks the rich golden hue of the ripe mango, showing sap burn or ripening disorder. Diffused natural lighting, detailed fruit quality image."
            },
            "deformation": {
                "description": "Shape irregularity - dents, flat spots, or asymmetrical growth",
                "spatial_prior_mask_paths": [
                    {"path": "knowledge_bases/masks/mango/deform_side.png", "region_name": "side"},
                    {"path": "knowledge_bases/masks/mango/deform_tip.png", "region_name": "beak tip"},
                    {"path": "knowledge_bases/masks/mango/deform_belly.png", "region_name": "belly"}
                ],
                "forbidden_region_masks": [],
                "colour_hsv": {"hue_min": 15, "hue_max": 50, "sat_min": 30, "sat_max": 180, "val_min": 55, "val_max": 215},
                "shape": {"min_area": 600, "max_area": 20000, "min_aspect_ratio": 0.35, "max_aspect_ratio": 3.8},
                "prompt_template": "A mango with a visible dent and shape deformation on the {location}. The fruit shows a flattened, indented area with gentle shadowing, distorting the natural oval contour. The skin follows the depression without breaking. Professional lighting emphasizing surface irregularities, realistic fruit grading."
            }
        }
        self._save_kb("mango_kb.yaml", mango_kb)

    # ------------------------------------------------------------------
    # Helper: save YAML
    # ------------------------------------------------------------------
    def _save_kb(self, filename: str, data: dict):
        path = self.kb_dir / filename
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  Created {filename}")

    # ------------------------------------------------------------------
    # Mask generation (ellipses, polygons, etc.)
    # ------------------------------------------------------------------
    def create_all_masks(self):
        """Create all spatial prior masks and forbidden masks for each fruit."""
        # Define mask shapes for apple
        apple_masks = {
            "bruise_cheek": (300, 200, 45),         # (width, height, rotation_degrees)
            "bruise_stem": (180, 150, -30),
            "bruise_bottom": (220, 180, 10),
            "bruise_shoulder": (250, 160, 60),
            "fungal_surface": (280, 220, 0),
            "fungal_stem": (190, 170, -45),
            "fungal_calyx": (200, 160, 30),
            "discolor_cheek": (400, 300, 10),
            "discolor_surface": (380, 280, 0),
            "discolor_shoulder": (350, 250, 45),
            "deform_side": (260, 190, 20),
            "deform_bottom": (240, 180, -15),
            "deform_shoulder": (270, 200, 55),
            "forbidden_core": (120, 120, 0)          # central core – never mask here
        }

        citrus_masks = {
            "bruise_surface": (350, 250, 30),
            "bruise_side": (320, 240, 0),
            "bruise_top": (200, 180, 90),
            "fungal_rind": (280, 220, 15),
            "fungal_spot": (250, 200, -20),
            "fungal_depression": (260, 210, 40),
            "discolor_surface": (420, 320, 10),
            "discolor_large": (400, 350, 0),
            "discolor_equator": (450, 100, 0),       # long horizontal band
            "deform_side": (300, 220, 5),
            "deform_surface": (280, 260, 45),
            "deform_pole": (180, 160, 0),
            "forbidden_stem": (80, 80, 0)            # small stem area
        }

        mango_masks = {
            "bruise_cheek": (320, 240, 20),
            "bruise_shoulder": (280, 200, -30),
            "bruise_belly": (300, 260, 10),
            "fungal_skin": (290, 230, 0),
            "fungal_spot": (260, 210, 45),
            "fungal_shoulder": (270, 200, -15),
            "discolor_surface": (380, 300, 5),
            "discolor_cheek": (340, 280, 25),
            "discolor_shoulder": (320, 250, 60),
            "deform_side": (280, 220, 0),
            "deform_tip": (150, 130, 90),
            "deform_belly": (310, 250, -10),
            "forbidden_seed": (100, 100, 0)
        }

        # Generate each set of masks
        self._generate_fruit_masks("apple", apple_masks)
        self._generate_fruit_masks("citrus", citrus_masks)
        self._generate_fruit_masks("mango", mango_masks)

    def _generate_fruit_masks(self, fruit: str, mask_dict: dict):
        """Create all mask images for a given fruit."""
        fruit_dir = self.mask_dir / fruit
        fruit_dir.mkdir(exist_ok=True)
        print(f"  Generating masks for {fruit}...")
        for name, (w, h, angle) in mask_dict.items():
            # Create ellipse mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            center = (256, 256)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            # For very elongated masks (e.g., equatorial band), we may want a rectangle
            if "equator" in name:
                # Horizontal strip
                mask = np.zeros((512, 512), dtype=np.uint8)
                y_center = 256
                half_height = h // 2
                cv2.rectangle(mask, (0, y_center - half_height), (512, y_center + half_height), 255, -1)
            if "forbidden" in name:
                # Forbidden masks should be small and central
                mask = np.zeros((512, 512), dtype=np.uint8)
                cv2.ellipse(mask, center, (w//2, h//2), 0, 0, 360, 255, -1)
            # Save
            out_path = fruit_dir / f"{name}.png"
            cv2.imwrite(str(out_path), mask)
        print(f"    Generated {len(mask_dict)} masks for {fruit}.")


# ------------------------------------------------------------------
# Run the creator
# ------------------------------------------------------------------
if __name__ == "__main__":
    creator = KnowledgeBaseCreator()
    creator.create_all_knowledge_bases()