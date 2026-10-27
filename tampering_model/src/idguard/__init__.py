"""IDGuard image-tampering detection package.

Public contract:
    TamperingModel, get_default_model   — model (src/idguard/model.py)
    TamperingPredictor, TamperingPrediction — inference (src/idguard/inference.py)
    decide, DecisionResult              — decision layer (src/idguard/decision_engine.py)
"""
from .decision_engine import (ALL_DECISIONS, FRAUD_FLAG, MANUAL_REVIEW,
                              NO_TAMPERING_DETECTED, POTENTIAL_TAMPERING,
                              RESCAN_REQUIRED, DecisionResult, decide)
from .model import TamperingModel, get_default_model
from .inference import InvalidImageError, TamperingPrediction, TamperingPredictor

__all__ = [
    "TamperingModel", "get_default_model",
    "TamperingPredictor", "TamperingPrediction", "InvalidImageError",
    "decide", "DecisionResult", "ALL_DECISIONS", "FRAUD_FLAG",
    "NO_TAMPERING_DETECTED", "POTENTIAL_TAMPERING", "RESCAN_REQUIRED", "MANUAL_REVIEW",
]
