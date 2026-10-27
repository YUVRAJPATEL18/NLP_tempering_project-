"""
Unit tests for text normalization and context-aware OCR confusion corrections.
"""

import pytest

from ocr_module.nlp.normalization import TextNormalizer
from ocr_module.nlp.spelling_correction import ContextualOCRCorrector


def test_text_normalizer():
    raw = "  Computer   Science  &  Engg.\n "
    res = TextNormalizer.generate_comparison_versions(raw)
    assert res["normalized_text"] == "computer science & engg."
    assert res["comparison_text"] == "computerscienceengg"


def test_ocr_confusion_numeric_context():
    # In numeric ID: 'O' becomes '0', 'I' becomes '1', 'S' becomes '5'
    corrected, reasons = ContextualOCRCorrector.correct_numeric_field("O123S8")
    assert corrected == "012358"
    assert len(reasons) > 0


def test_ocr_confusion_name_context():
    # In name context: 'O' remains 'O', but digit noise '0' becomes 'O', '5' becomes 'S'
    corrected, reasons = ContextualOCRCorrector.correct_name_field("YUVRA0 PAT3L")
    assert "YUVRAO" in corrected


def test_passport_number_correction():
    # Passport: Alpha prefix, numeric suffix
    corrected, _ = ContextualOCRCorrector.correct_passport_number("01234567")
    assert corrected.startswith("O")  # First pos becomes letter 'O'
    assert corrected[1:] == "1234567"  # Trailing positions numeric
