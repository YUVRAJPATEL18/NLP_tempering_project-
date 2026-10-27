"""
OCR Engine subpackage for IDGuard.
"""

from .base_ocr import BaseOCREngine
from .paddle_ocr_engine import PaddleOCREngine, FallbackOCREngine
from .result_parser import ResultParser

__all__ = ["BaseOCREngine", "PaddleOCREngine", "FallbackOCREngine", "ResultParser"]
