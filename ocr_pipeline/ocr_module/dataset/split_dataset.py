"""
Train / Validation / Test dataset splitter.
"""

import os
import random
from typing import List, Dict, Any


class DatasetSplitter:
    """Handles train/val/test splitting for OCR dataset files."""

    @staticmethod
    def split(
        image_paths: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        """
        Splits list of image paths into train, val, and test subsets.
        """
        if not image_paths:
            return {"train": [], "val": [], "test": []}

        # Deduplicate paths
        unique_paths = list(set(image_paths))
        random.seed(seed)
        random.shuffle(unique_paths)

        n = len(unique_paths)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return {
            "train": unique_paths[:train_end],
            "val": unique_paths[train_end:val_end],
            "test": unique_paths[val_end:]
        }
