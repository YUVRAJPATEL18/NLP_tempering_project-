"""
Annotation converter for unifying dataset label formats into normalized internal schemas.
"""

import os
import json
import csv
from typing import Dict, Any, List


class AnnotationConverter:
    """Converts CSV/JSON ground truth files into IDGuard standard annotation format."""

    @staticmethod
    def create_standard_annotation(
        image_name: str,
        document_type: str = "passport",
        fields: Dict[str, str] = None,
        full_text: str = ""
    ) -> Dict[str, Any]:
        """Creates unified annotation object."""
        return {
            "image": image_name,
            "document_type": document_type,
            "fields": fields or {},
            "full_text": full_text
        }

    @staticmethod
    def save_annotation_file(annotation_data: Dict[str, Any], output_json_path: str):
        """Saves annotation object to JSON file."""
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=4)
