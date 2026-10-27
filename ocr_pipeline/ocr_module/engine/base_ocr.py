"""
Abstract base OCR engine interface.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from ..schemas import OCRResult


class BaseOCREngine(ABC):
    """
    Abstract interface for all OCR engine implementations (PaddleOCR, EasyOCR, Tesseract, TrOCR, etc.).
    """

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        document_id: Optional[str] = None,
        variant_name: str = "original"
    ) -> OCRResult:
        """
        Executes text detection and recognition on an input image array.

        Args:
            image: Input image as BGR numpy array.
            document_id: Unique identifier for tracking request.
            variant_name: Name of preprocessing variant applied.

        Returns:
            OCRResult object with detected lines, bounding boxes, confidence, and metadata.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the underlying OCR engine dependencies and weights are available."""
        pass
