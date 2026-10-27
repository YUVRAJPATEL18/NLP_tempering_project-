"""Decision-engine tests covering IDGuard's decision principles:
OCR never establishes fraud, poor quality recaptures, strong evidence survives,
conflicting signals go to review."""
from __future__ import annotations

import pytest

from idguard.decision_engine import (ALL_DECISIONS, FRAUD_FLAG, MANUAL_REVIEW,
                                     NO_TAMPERING_DETECTED, POTENTIAL_TAMPERING,
                                     RESCAN_REQUIRED, decide)

THR = {"tamper_high": 0.50, "tamper_low": 0.20, "rescan_below": 0.30, "ocr_min": 0.55}
# Same thresholds plus the dataset-calibrated degraded-quality threshold, as the
# production predictor supplies it.
THR_Q = dict(THR, quality_degraded=0.80)


def test_clear_genuine():
    r = decide(0.05, 0.90, None, THR)
    assert r.decision == NO_TAMPERING_DETECTED


def test_clear_tampered():
    r = decide(0.95, 0.90, 0.9, THR)
    assert r.decision == POTENTIAL_TAMPERING
    assert any("0.950" in x for x in r.reasons)


def test_borderline_goes_to_review():
    r = decide(0.35, 0.90, None, THR)
    assert r.decision == MANUAL_REVIEW


def test_unusable_quality_triggers_rescan_not_verdict():
    r = decide(0.05, 0.10, None, THR)
    assert r.decision == RESCAN_REQUIRED
    assert any("quality" in x.lower() for x in r.reasons)


def test_unusable_quality_with_strong_evidence_goes_to_review():
    r = decide(0.97, 0.10, None, THR)
    assert r.decision == MANUAL_REVIEW          # not an automatic fraud flag
    assert r.decision != FRAUD_FLAG
    assert any("strong forensic evidence" in x for x in r.reasons)


def test_low_ocr_alone_never_flags_fraud():
    r = decide(0.02, 0.90, 0.10, THR)
    assert r.decision == NO_TAMPERING_DETECTED
    assert r.decision != POTENTIAL_TAMPERING
    assert any("OCR" in x for x in r.reasons)   # noted, but not fraud


def test_strong_forensic_evidence_survives_low_ocr():
    r = decide(0.92, 0.85, 0.10, THR)
    assert r.decision == POTENTIAL_TAMPERING


def test_low_ocr_with_borderline_forensics_stays_review():
    r = decide(0.30, 0.85, 0.10, THR)
    assert r.decision == MANUAL_REVIEW


def test_degraded_capture_with_strong_evidence_goes_to_review_not_fraud_flag():
    r = decide(0.95, 0.70, None, THR_Q)
    assert r.decision == MANUAL_REVIEW
    assert r.decision != FRAUD_FLAG
    assert any("degraded" in x for x in r.reasons)


def test_clean_capture_with_strong_evidence_is_flagged():
    r = decide(0.95, 0.90, None, THR_Q)
    assert r.decision == POTENTIAL_TAMPERING


def test_degraded_genuine_clearance_records_the_quality_caveat():
    r = decide(0.05, 0.70, None, THR_Q)
    assert r.decision == NO_TAMPERING_DETECTED
    assert any("degraded" in x for x in r.reasons)


def test_degraded_gate_off_still_flags_for_comparison():
    on = decide(0.95, 0.70, None, THR_Q, gate_enabled=True)
    off = decide(0.95, 0.70, None, THR_Q, gate_enabled=False)
    assert off.decision == POTENTIAL_TAMPERING
    assert on.decision == MANUAL_REVIEW


def test_gate_off_ignores_quality():
    on = decide(0.60, 0.10, None, THR, gate_enabled=True)
    off = decide(0.60, 0.10, None, THR, gate_enabled=False)
    assert off.decision == POTENTIAL_TAMPERING   # quality-blind comparison mode
    assert on.decision in (RESCAN_REQUIRED, MANUAL_REVIEW)
    gate_off_clear = decide(0.10, 0.10, None, THR, gate_enabled=False)
    assert gate_off_clear.decision == NO_TAMPERING_DETECTED


def test_decisions_always_valid_and_reasoned():
    for p in (0.0, 0.1, 0.35, 0.7, 1.0):
        for q in (0.05, 0.5, 0.95):
            for ocr in (None, 0.2, 0.9):
                r = decide(p, q, ocr, THR)
                assert r.decision in ALL_DECISIONS
                assert r.reasons and all(isinstance(x, str) for x in r.reasons)
                d = r.to_dict()
                assert set(d) == {"decision", "decision_reasons"}


def test_invalid_threshold_order_rejected():
    bad = dict(THR, tamper_low=0.9)
    with pytest.raises(ValueError):
        decide(0.5, 0.9, None, bad)
