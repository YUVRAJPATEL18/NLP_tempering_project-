"""
Postprocessing pipeline orchestrator.
"""

from typing import Tuple
from .text_cleaner import TextCleaner
from .confidence_filter import ConfidenceFilter
from .duplicate_remover import DuplicateRemover
from .line_merger import LineMerger
from ..schemas import OCRResult, OCRLine


class TextPostprocessor:
    """Orchestrates postprocessing steps on an OCRResult object."""

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence

    def process(self, ocr_result: OCRResult) -> Tuple[OCRResult, str]:
        """
        Executes confidence filtering, deduplication, spatial merging, and cleaning.

        Returns:
            (updated_ocr_result, cleaned_text_string)
        """
        if not ocr_result.lines:
            return ocr_result, ""

        # Step 1: Filter low confidence predictions
        lines = ConfidenceFilter.filter_lines(ocr_result.lines, min_confidence=self.min_confidence)

        # Step 2: Remove duplicate text lines
        lines = DuplicateRemover.remove_duplicates(lines)

        # Step 3: Spatial sorting & adjacent line merging
        lines = LineMerger.merge_adjacent_lines(lines)

        # Step 4: Clean line text values
        cleaned_lines = []
        for line in lines:
            cleaned_t = TextCleaner.clean_text(line.text)
            if cleaned_t:
                line.text = cleaned_t
                cleaned_lines.append(line)

        raw_text = ocr_result.raw_text
        cleaned_text = "\n".join([l.text for l in cleaned_lines])

        updated_result = OCRResult(
            document_id=ocr_result.document_id,
            raw_text=raw_text,
            lines=cleaned_lines,
            average_confidence=ocr_result.average_confidence,
            image_width=ocr_result.image_width,
            image_height=ocr_result.image_height,
            processing_time_ms=ocr_result.processing_time_ms,
            preprocessing_variant=ocr_result.preprocessing_variant
        )

        return updated_result, cleaned_text
