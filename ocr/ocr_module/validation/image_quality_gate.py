"""Early image quality gate; thresholds are provisional, configurable heuristics."""

from typing import Any, Dict, List
from ..config import settings


class ImageQualityGate:
    """Classify upload quality before preprocessing or OCR begins."""

    @staticmethod
    def assess(measurements: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        reject: List[str] = []
        short_edge = min(measurements["width"], measurements["height"])

        if short_edge < settings.quality_min_short_edge:
            issues.append("LOW_TEXT_RESOLUTION")
            reject.append("LOW_TEXT_RESOLUTION")
        blur = measurements["blur_score"]
        if blur < settings.quality_reject_blur_score:
            issues.append("POSSIBLE_BLUR")
            reject.append("POSSIBLE_BLUR")
        elif blur < settings.quality_warning_blur_score:
            issues.append("POSSIBLE_BLUR")

        if measurements["brightness"] < settings.quality_reject_brightness:
            issues.append("UNDEREXPOSED")
            reject.append("UNDEREXPOSED")
        elif measurements["brightness"] < settings.quality_warning_brightness:
            issues.append("LOW_BRIGHTNESS")

        # Global contrast can be low for a mostly-white card with a small amount
        # of crisp text. Reject only near-uniform images; otherwise warn.
        if measurements["contrast"] < 8:
            issues.append("LOW_CONTRAST")
            reject.append("LOW_CONTRAST")
        elif measurements["contrast"] < 35:
            issues.append("LOW_CONTRAST")

        glare = measurements.get("glare_fraction", 0.0)
        if glare >= settings.quality_reject_glare_fraction and measurements["contrast"] < 8:
            issues.append("SEVERE_GLARE_OR_HIGHLIGHTS")
            reject.append("SEVERE_GLARE_OR_HIGHLIGHTS")
        elif glare >= settings.quality_warning_glare_fraction:
            issues.append("POSSIBLE_GLARE")

        poor = bool(reject)
        return {
            "status": "poor" if poor else ("warning" if issues else "good"),
            "issues": issues,
            "measurements": measurements,
            "accepted": not poor,
            "message": (
                "The image quality is insufficient for reliable processing. Please retake the image."
                if poor else "Image quality is acceptable for OCR."
            ),
        }
