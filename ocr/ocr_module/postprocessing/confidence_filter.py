"""
Confidence-based line and line-token filter.
"""

from typing import List
from ..schemas import OCRLine
from ..config import settings


class ConfidenceFilter:
    """Filters OCR predictions below confidence thresholds."""

    @staticmethod
    def filter_lines(lines: List[OCRLine], min_confidence: float = settings.min_confidence) -> List[OCRLine]:
        """
        Returns list of OCRLine objects whose confidence meets or exceeds min_confidence.
        If all lines fall below threshold, returns top 3 highest-confidence lines as safety fallback.
        """
        if not lines:
            return []

        filtered = [line for line in lines if line.confidence >= min_confidence]
        if not filtered:
            # Fallback: retain top lines to avoid total data loss
            sorted_lines = sorted(lines, key=lambda l: l.confidence, reverse=True)
            filtered = sorted_lines[:max(1, len(lines) // 2)]

        return filtered
