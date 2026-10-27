"""
Ground-truth label loader for evaluation dataset matching.
"""

import os
import json
import csv
from typing import Dict, Any, Optional


class GroundTruthLoader:
    """Loads and validates ground truth JSON/CSV labels."""

    @staticmethod
    def load_annotation(file_path: str) -> Optional[Dict[str, Any]]:
        """Loads ground truth JSON file."""
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception:
            return None
