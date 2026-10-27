"""
Duplicate line and duplicate text phrase remover.
"""

from typing import List
from ..schemas import OCRLine


class DuplicateRemover:
    """Eliminates repeated text lines produced by multi-pass OCR or overlapping bounding boxes."""

    @staticmethod
    def remove_duplicates(lines: List[OCRLine]) -> List[OCRLine]:
        """
        Deduplicates lines based on normalized string matching while preserving spatial line order.
        """
        seen_texts = set()
        unique_lines: List[OCRLine] = []

        for line in lines:
            norm = line.text.strip().lower()
            if not norm or norm in seen_texts:
                continue
            seen_texts.add(norm)
            unique_lines.append(line)

        return unique_lines
