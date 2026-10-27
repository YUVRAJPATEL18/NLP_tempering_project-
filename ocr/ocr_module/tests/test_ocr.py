"""
Unit tests for OCR engines, result parsing, and low confidence handling.
"""

import pytest
import numpy as np

from ocr_module.engine.result_parser import ResultParser
from ocr_module.engine.paddle_ocr_engine import FallbackOCREngine
from ocr_module.schemas import OCRResult


def test_result_parser_valid_paddle_output():
    raw_paddle = [
        [
            [[[10, 10], [100, 10], [100, 40], [10, 40]], ("PASSPORT NO: A1234567", 0.95)],
            [[[10, 50], [100, 50], [100, 80], [10, 80]], ("NAME: YUVRAJ PATEL", 0.92)]
        ]
    ]
    res = ResultParser.parse_paddle_output(
        raw_output=raw_paddle,
        image_shape=(800, 600, 3),
        processing_time_ms=120.0,
        variant_name="clahe"
    )
    assert len(res.lines) == 2
    assert "A1234567" in res.raw_text
    assert res.average_confidence > 0.90


def test_result_parser_empty_paddle_output():
    res = ResultParser.parse_paddle_output(
        raw_output=[],
        image_shape=(800, 600, 3),
        processing_time_ms=50.0
    )
    assert len(res.lines) == 0
    assert res.average_confidence == 0.0
    assert res.raw_text == ""


def test_fallback_ocr_engine():
    engine = FallbackOCREngine()
    assert engine.is_available() is True
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    res = engine.recognize(img)
    assert isinstance(res, OCRResult)
    assert res.lines == []
    assert res.raw_text == ""
    assert res.average_confidence == 0.0
