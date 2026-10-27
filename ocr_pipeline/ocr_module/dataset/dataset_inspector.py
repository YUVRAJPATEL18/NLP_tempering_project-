"""
Automated dataset locator, structure inspector, and statistics report generator.
"""

import os
import glob
import json
import csv
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from ..preprocessing.image_loader import ImageLoader


class DatasetInspector:
    """Discovers, inspects, and analyzes document datasets."""

    def __init__(self, search_dir: str = "."):
        self.search_dir = os.path.abspath(search_dir)

    def locate_dataset(self) -> List[str]:
        """
        Scans search directory for dataset candidates containing image files or annotation CSV/JSON files.
        """
        candidates = []
        for root, dirs, files in os.walk(self.search_dir):
            if "venv" in root or ".git" in root or "__pycache__" in root:
                continue
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.jng'))]
            annot_files = [f for f in files if f.lower().endswith(('.csv', '.json', '.xml', '.txt')) and f.lower() != 'requirements.txt']

            if image_files or annot_files:
                candidates.append(root)
        return candidates

    def inspect(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Inspects dataset at path or auto-located dataset directory.
        """
        target_dir = dataset_path or self.search_dir
        if not os.path.exists(target_dir):
            return {
                "total_images": 0,
                "image_formats": [],
                "average_width": 0,
                "average_height": 0,
                "has_annotations": False,
                "annotation_format": "none",
                "has_full_text": False,
                "has_bounding_boxes": False,
                "document_types": [],
                "recommended_use": "No valid dataset directory found.",
                "case_classification": "CASE_B"
            }

        # Find all images
        img_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.jng']
        image_paths = []
        for ext in img_extensions:
            image_paths.extend(glob.glob(os.path.join(target_dir, f"**/*{ext}"), recursive=True))
            image_paths.extend(glob.glob(os.path.join(target_dir, f"**/*{ext.upper()}"), recursive=True))

        # Filter out venv/preprocessed outputs
        image_paths = [p for p in image_paths if "venv" not in p and "preprocessed_out" not in p]

        # Find annotations
        csv_files = glob.glob(os.path.join(target_dir, "**/*.csv"), recursive=True)
        json_files = glob.glob(os.path.join(target_dir, "**/*.json"), recursive=True)

        has_annotations = False
        annotation_format = "none"
        has_full_text = False
        has_bounding_boxes = False
        sample_annotations = {}

        if csv_files:
            has_annotations = True
            annotation_format = "csv"
            for csv_p in csv_files:
                try:
                    with open(csv_p, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.DictReader(f, delimiter=';')
                        fieldnames = reader.fieldnames or []
                        if 'text' in fieldnames or 'ground_truth' in fieldnames:
                            has_full_text = True
                        if 'bbox' in fieldnames or 'box' in fieldnames:
                            has_bounding_boxes = True
                except Exception:
                    pass

        elif json_files:
            has_annotations = True
            annotation_format = "json"

        formats_found = sorted(list(set(os.path.splitext(p)[1].lower() for p in image_paths)))

        widths, heights = [], []
        blurry_count = 0
        rotated_count = 0

        sample_paths = image_paths[:20]  # Sample first 20 images for fast metadata
        for p in sample_paths:
            try:
                img, metrics = ImageLoader.load_image(p)
                widths.append(metrics["width"])
                heights.append(metrics["height"])
                if metrics["is_blurry"]:
                    blurry_count += 1
            except Exception:
                continue

        avg_w = int(np.mean(widths)) if widths else 0
        avg_h = int(np.mean(heights)) if heights else 0

        # Determine Case A, B, C, or D
        if has_annotations and (has_full_text or has_bounding_boxes):
            case = "CASE_A"
            recommended = "Full document images with ground-truth labels. Ready for OCR training and evaluation."
        elif len(image_paths) > 0 and not has_annotations:
            case = "CASE_B"
            recommended = "Full document images without ground truth. Suitable for baseline inference and template creation."
        elif len(image_paths) == 0:
            case = "CASE_D"
            recommended = "No image dataset found. Use synthetic document generator for testing."
        else:
            case = "CASE_B"
            recommended = "Document images identified. Proceeding with baseline inference."

        return {
            "total_images": len(image_paths),
            "image_formats": formats_found,
            "average_width": avg_w,
            "average_height": avg_h,
            "has_annotations": has_annotations,
            "annotation_format": annotation_format,
            "has_full_text": has_full_text,
            "has_bounding_boxes": has_bounding_boxes,
            "document_types": ["passport", "id_card"],
            "recommended_use": recommended,
            "case_classification": case,
            "sample_image_paths": image_paths[:10]
        }
