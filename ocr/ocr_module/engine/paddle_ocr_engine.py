"""
PaddleOCR Engine implementation and robust fallback OCR engine.
"""

import time
import uuid
import logging
from typing import Optional, Union, List, Any
import cv2
import numpy as np

from .base_ocr import BaseOCREngine
from .result_parser import ResultParser
from ..schemas import OCRResult, OCRLine
from ..config import settings
from ..exceptions import OCRError, EngineInitializationError

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR Engine implementation supporting angle classification, CPU/GPU execution.
    """

    def __init__(
        self,
        lang: str = settings.paddle_lang,
        use_gpu: bool = settings.use_gpu,
        min_confidence: float = settings.min_confidence
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self._engine = None
        self._initialized = False
        self._init_engine()

    def _init_engine(self):
        try:
            from paddleocr import PaddleOCR
            try:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu)
            except Exception:
                try:
                    self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang)
                except Exception:
                    self._engine = PaddleOCR(lang=self.lang)

            self._initialized = True
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {e}. Engine will fall back if invoked.")
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self._engine is not None

    def recognize(
        self,
        image: np.ndarray,
        document_id: Optional[str] = None,
        variant_name: str = "original"
    ) -> OCRResult:
        if not self.is_available():
            raise EngineInitializationError("PaddleOCR engine is not initialized or unavailable.")

        start_time = time.time()
        try:
            # PaddleOCR expects 3-channel BGR numpy array
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image

            # Execute PaddleOCR inference
            try:
                predictions = self._engine.ocr(image_bgr)
            except TypeError:
                predictions = self._engine.ocr(image_bgr, cls=True)
            elapsed_ms = (time.time() - start_time) * 1000.0

            return ResultParser.parse_paddle_output(
                raw_output=predictions,
                image_shape=image.shape,
                processing_time_ms=elapsed_ms,
                document_id=document_id,
                variant_name=variant_name
            )

        except Exception as e:
            raise OCRError(f"PaddleOCR inference failed: {str(e)}")


class FallbackOCREngine(BaseOCREngine):
    """
    Lightweight computer-vision fallback OCR engine for test environments or lightweight execution.
    Detects text line contours and extracts structural features.
    """

    def is_available(self) -> bool:
        return True

    def recognize(
        self,
        image: np.ndarray,
        document_id: Optional[str] = None,
        variant_name: str = "original"
    ) -> OCRResult:
        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # Sort top-to-bottom

        lines: List[OCRLine] = []
        confidences: List[float] = []

        line_num = 1
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 40 or bh < 10:
                continue

            bbox = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
            # This detector finds text-like regions only. It has no recognizer, so
            # it must not emit placeholder OCR text or fabricated confidence.
            line_num += 1

        elapsed_ms = (time.time() - start_time) * 1000.0
        avg_conf = 0.0
        raw_text = "\n".join([l.text for l in lines])

        return OCRResult(
            document_id=doc_id,
            raw_text=raw_text,
            lines=lines,
            average_confidence=avg_conf,
            image_width=w,
            image_height=h,
            processing_time_ms=round(elapsed_ms, 2),
            preprocessing_variant=variant_name
        )
