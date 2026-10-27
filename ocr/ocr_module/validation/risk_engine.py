"""Explainable review-priority scoring from available validation signals.

This heuristic is not calibrated and must not be interpreted as authenticity probability.
"""

from typing import Dict, List, Optional, Tuple
from ..schemas import FieldExtractionValue


class RiskEngine:
    """Combine present OCR and format signals; omit unavailable evidence."""

    def __init__(self, low_confidence_threshold: float = 0.40, review_threshold: float = 0.50):
        self.low_confidence_threshold = low_confidence_threshold
        self.review_threshold = review_threshold

    def score(self, fields: Dict[str, FieldExtractionValue]) -> Tuple[Optional[float], str, List[str]]:
        if not fields:
            return None, "unavailable", ["No extracted fields are available for risk assessment."]
        signals = []
        reasons = []
        for name, field in fields.items():
            if field.confidence < self.low_confidence_threshold:
                signals.append(1.0 - field.confidence)
                reasons.append(f"{name}: low OCR confidence ({field.confidence:.2f}).")
            if field.validation_status == "invalid":
                signals.append(1.0)
                reasons.append(f"{name}: field format or logical validation failed.")
        # Mean over the signals present; absence of a signal is never treated as success.
        value = sum(signals) / len(signals) if signals else 0.0
        category = "review" if value >= self.review_threshold else "no_triggering_signals"
        if not signals:
            reasons.append("No risk-triggering OCR or validation signals were available.")
        return round(value, 4), category, reasons
