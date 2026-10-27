"""
OCR evaluation metrics and dataset benchmarking subpackage.
"""

from .metrics import MetricsCalculator
from .ground_truth_loader import GroundTruthLoader
from .evaluate_ocr import OCREvaluator
from .generate_report import ReportGenerator

__all__ = [
    "MetricsCalculator",
    "GroundTruthLoader",
    "OCREvaluator",
    "ReportGenerator"
]
