"""Confidence-aware decision layer.

Keeps three signals strictly separate — image quality, OCR reliability and
forensic tampering evidence — and maps them onto IDGuard's four outcomes:

    NO_TAMPERING_DETECTED  — no supporting evidence of tampering (NOT a claim
                             that the document is authentic)
    POTENTIAL_TAMPERING    — forensic probability above the calibrated threshold
    RESCAN_REQUIRED        — image quality too poor to judge; recapture
    MANUAL_REVIEW          — conflicting or borderline evidence for a human

Principles enforced here (see docs/tampering_model_audit.md):
  1. Low OCR confidence alone never establishes fraud.
  2. Unusable image quality triggers recapture, not a verdict; degraded quality
     on a readable image escalates forensic evidence to human review rather than
     an automatic fraud flag.
  3. Strong forensic evidence is never silently downgraded to "clean".
  4. Conflicting evidence -> review, not an unsupported verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

NO_TAMPERING_DETECTED = "NO_TAMPERING_DETECTED"
POTENTIAL_TAMPERING = "POTENTIAL_TAMPERING"
RESCAN_REQUIRED = "RESCAN_REQUIRED"
MANUAL_REVIEW = "MANUAL_REVIEW"

ALL_DECISIONS = (NO_TAMPERING_DETECTED, POTENTIAL_TAMPERING, RESCAN_REQUIRED, MANUAL_REVIEW)
FRAUD_FLAG = POTENTIAL_TAMPERING  # the only outcome that alleges tampering


@dataclass
class DecisionResult:
    decision: str
    reasons: List[str]

    def to_dict(self) -> Dict:
        return {"decision": self.decision, "decision_reasons": list(self.reasons)}


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def decide(tampering_probability: float, quality_score: float,
           ocr_reliability: Optional[float], thresholds: Dict[str, float],
           gate_enabled: bool = True) -> DecisionResult:
    """Map calibrated forensic probability + quality + OCR onto a decision.

    `thresholds` keys: tamper_high, tamper_low, rescan_below, ocr_min.
    """
    p = float(tampering_probability)
    q = float(quality_score)
    high = float(thresholds["tamper_high"])
    low = float(thresholds["tamper_low"])
    rescan_below = float(thresholds["rescan_below"])
    ocr_min = float(thresholds.get("ocr_min", 0.55))
    # `quality_degraded` (the dataset-calibrated degraded threshold) is distinct
    # from `rescan_below` (the absolute unusable-capture floor). When it is
    # supplied, a degraded-but-readable capture can no longer produce an
    # automatic fraud flag on forensic evidence alone.
    degraded_below = float(thresholds.get("quality_degraded", rescan_below))
    if low > high:
        raise ValueError("tamper_low must be <= tamper_high")

    if not gate_enabled:
        # Gate disabled: quality-unaware two-state threshold (the comparison baseline)
        if p >= high:
            return DecisionResult(POTENTIAL_TAMPERING,
                                  [f"tampering probability {_fmt(p)} >= threshold {_fmt(high)} "
                                   "(confidence gate disabled: quality ignored)"])
        return DecisionResult(NO_TAMPERING_DETECTED,
                              [f"tampering probability {_fmt(p)} < threshold {_fmt(high)} "
                               "(confidence gate disabled: quality ignored)"])

    reasons: List[str] = []

    # --- unusable image quality -------------------------------------------
    if q < rescan_below:
        if p >= high:
            reasons.append(f"strong forensic evidence (p={_fmt(p)} >= {_fmt(high)}) "
                           f"but image quality is unusable (q={_fmt(q)} < {_fmt(rescan_below)})")
            reasons.append("conflicting evidence: human inspection required rather than "
                           "an automatic fraud flag or an endless rescan loop")
            if ocr_reliability is not None and ocr_reliability < ocr_min:
                reasons.append(f"OCR reliability {_fmt(ocr_reliability)} is low; "
                               "OCR failure alone is not treated as fraud")
            return DecisionResult(MANUAL_REVIEW, reasons)
        reasons.append(f"image quality too poor to judge (q={_fmt(q)} < {_fmt(rescan_below)}); "
                       f"recapture required (forensic probability {_fmt(p)} is unreliable here)")
        return DecisionResult(RESCAN_REQUIRED, reasons)

    # --- usable image ------------------------------------------------------
    if p >= high:
        if quality_score_is_degraded(q, degraded_below):
            reasons.append(f"forensic tampering probability {_fmt(p)} >= calibrated threshold "
                           f"{_fmt(high)}, but the capture is degraded "
                           f"(q={_fmt(q)} < {_fmt(degraded_below)})")
            reasons.append("conflicting evidence: image quality lowers the reliability of the "
                           "forensic signal, so this goes to human review instead of an "
                           "automatic fraud flag")
            return DecisionResult(MANUAL_REVIEW, reasons)
        reasons.append(f"forensic tampering probability {_fmt(p)} >= calibrated threshold {_fmt(high)}")
        return DecisionResult(POTENTIAL_TAMPERING, reasons)

    if p >= low:
        reasons.append(f"forensic evidence is borderline (p={_fmt(p)} within "
                       f"[{_fmt(low)}, {_fmt(high)})); conflicting/uncertain signals -> review")
        if ocr_reliability is not None and ocr_reliability < ocr_min:
            reasons.append(f"OCR reliability {_fmt(ocr_reliability)} < {_fmt(ocr_min)} "
                           "(recorded as a separate signal, not as fraud)")
        return DecisionResult(MANUAL_REVIEW, reasons)

    reasons.append(f"forensic tampering probability {_fmt(p)} < {_fmt(low)} "
                   "with usable image quality")
    if quality_score_is_degraded(q, degraded_below):
        reasons.append(f"note: capture quality is degraded (q={_fmt(q)} < {_fmt(degraded_below)}), "
                       "so this clearance is less reliable than one from a clean capture")
    if ocr_reliability is not None and ocr_reliability < ocr_min:
        reasons.append(f"note: OCR reliability {_fmt(ocr_reliability)} is low — "
                       "an OCR problem, never by itself evidence of tampering")
    return DecisionResult(NO_TAMPERING_DETECTED, reasons)


def quality_score_is_degraded(q: float, degraded_below: float) -> bool:
    return q < degraded_below
