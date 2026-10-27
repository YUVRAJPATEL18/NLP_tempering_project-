"""
End-to-end OCR benchmark and evaluation runner.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np

from .metrics import MetricsCalculator
from ..schemas import EvaluationReport
from ..engine.paddle_ocr_engine import PaddleOCREngine, FallbackOCREngine
from ..preprocessing.pipeline import PreprocessingPipeline


class OCREvaluator:
    """Evaluates OCR prediction quality against ground truth targets."""

    def __init__(self, engine=None):
        paddle = PaddleOCREngine()
        self.engine = paddle if paddle.is_available() else FallbackOCREngine()
        self.pipeline = PreprocessingPipeline()

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]]
    ) -> EvaluationReport:
        """
        Runs evaluation on a list of sample dicts containing 'image_path' and optional 'ground_truth'.
        """
        if not samples:
            return EvaluationReport(
                cer=0.0,
                wer=0.0,
                exact_field_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                average_confidence=0.0,
                processing_time_ms=0.0,
                success_rate=0.0,
                total_samples=0
            )

        cers, wers = [], []
        confidences = []
        times = []
        successes = 0

        for sample in samples:
            img_p = sample.get("image_path")
            gt = sample.get("ground_truth", {})
            gt_text = gt.get("full_text", "")

            start_t = time.time()
            try:
                processed_img, _ = self.pipeline.process_single(img_p, variant="enhanced")
                result = self.engine.recognize(processed_img)
                elapsed = (time.time() - start_t) * 1000.0

                times.append(elapsed)
                confidences.append(result.average_confidence)
                successes += 1

                if gt_text:
                    cer = MetricsCalculator.calculate_cer(gt_text, result.raw_text)
                    wer = MetricsCalculator.calculate_wer(gt_text, result.raw_text)
                    cers.append(cer)
                    wers.append(wer)

            except Exception:
                times.append(0.0)

        avg_cer = float(np.mean(cers)) if cers else 0.0
        avg_wer = float(np.mean(wers)) if wers else 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        avg_time = float(np.mean(times)) if times else 0.0
        succ_rate = (successes / float(len(samples))) if samples else 0.0

        return EvaluationReport(
            cer=round(avg_cer, 4),
            wer=round(avg_wer, 4),
            exact_field_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            average_confidence=round(avg_conf, 4),
            processing_time_ms=round(avg_time, 2),
            success_rate=round(succ_rate, 4),
            total_samples=len(samples)
        )
