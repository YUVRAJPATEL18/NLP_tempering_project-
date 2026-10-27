"""
Annotation template generator for unlabelled images (CASE B).
"""

import os
import json
from typing import Dict, Any, List


class AnnotationTemplateGenerator:
    """Generates blank ground truth templates and saves predictions separately."""

    @staticmethod
    def create_blank_template(image_name: str, document_type: str = "passport") -> Dict[str, Any]:
        """Creates an empty annotation template for human labeling."""
        return {
            "image": image_name,
            "document_type": document_type,
            "is_ground_truth": False,
            "annotated_by": None,
            "fields": {
                "name": "",
                "passport_number": "",
                "student_id": "",
                "date_of_birth": "",
                "nationality": "",
                "college": "",
                "course": "",
                "date_of_issue": "",
                "date_of_expiry": "",
                "valid_till": "",
                "address": "",
                "mrz": ""
            },
            "full_text": ""
        }

    @staticmethod
    def save_prediction_file(
        image_name: str,
        ocr_result_dict: Dict[str, Any],
        output_dir: str
    ):
        """Saves OCR prediction result into predictions folder."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        out_path = os.path.join(output_dir, f"{base_name}_prediction.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_result_dict, f, indent=4)
