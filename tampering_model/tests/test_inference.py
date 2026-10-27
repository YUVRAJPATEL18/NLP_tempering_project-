"""Inference tests: structured output schema, invalid input handling,
checkpoint loading and heatmap alignment."""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from idguard.decision_engine import ALL_DECISIONS
from idguard.inference import InvalidImageError, TamperingPrediction
from idguard.model import model_from_checkpoint

from conftest import make_document_bytes

REQUIRED_FIELDS = {
    "tampering_probability", "tampering_detected", "quality_score",
    "ocr_reliability", "predicted_regions", "decision", "decision_reasons",
    "model_version",
}


def test_prediction_schema(predictor, sharp_image_bytes):
    pred = predictor.predict(sharp_image_bytes, ocr_reliability=0.9)
    assert isinstance(pred, TamperingPrediction)
    d = pred.to_dict()
    assert REQUIRED_FIELDS <= set(d)
    assert 0.0 <= d["tampering_probability"] <= 1.0
    assert 0.0 <= d["quality_score"] <= 1.0
    assert isinstance(d["tampering_detected"], bool)
    assert d["decision"] in ALL_DECISIONS
    assert isinstance(d["decision_reasons"], list) and d["decision_reasons"]
    assert d["ocr_reliability"] == 0.9
    assert d["model_version"] == "test-tiny-random"
    assert pred.tampering_detected == (pred.decision == "POTENTIAL_TAMPERING")


def test_no_internal_paths_in_response(predictor, sharp_image_bytes):
    d = predictor.predict(sharp_image_bytes).to_dict()
    blob = str(d)
    for forbidden in ("datasets/", "checkpoints/", "/Users/", "config.yaml"):
        assert forbidden not in blob


def test_missing_ocr_is_none_not_zero(predictor, sharp_image_bytes):
    pred = predictor.predict(sharp_image_bytes)
    assert pred.ocr_reliability is None
    assert pred.to_dict()["ocr_reliability"] is None


def test_quality_signal_reflects_blur(predictor, sharp_image_bytes, blurred_image_bytes):
    sharp = predictor.predict(sharp_image_bytes)
    blurred = predictor.predict(blurred_image_bytes)
    assert sharp.quality_score > blurred.quality_score
    # blur must NOT by itself make an image tampered
    assert blurred.tampering_probability < 1.0


def test_heatmap_alignment(predictor, sharp_image_bytes):
    pred = predictor.predict(sharp_image_bytes, include_heatmap=True)
    assert pred.heatmap is not None
    size = predictor.cfg["data"]["image_size"]
    assert pred.heatmap.shape == (size, size)
    assert float(pred.heatmap.min()) >= 0.0 and float(pred.heatmap.max()) <= 1.0 + 1e-6
    # overlay keeps the source dimensions
    pred2, overlay = predictor.predict_with_overlay(sharp_image_bytes)
    assert overlay.size == Image.open(io.BytesIO(sharp_image_bytes)).size
    assert overlay.mode == "RGB"


def test_predicted_regions_normalized(predictor, sharp_image_bytes):
    pred = predictor.predict(sharp_image_bytes, include_heatmap=True)
    if pred.predicted_regions:
        for box in pred.predicted_regions:
            assert len(box) == 4
            x0, y0, x1, y1 = box
            assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def test_timings_recorded(predictor, sharp_image_bytes):
    pred = predictor.predict(sharp_image_bytes)
    keys = {"decode", "quality", "preprocess", "inference", "postprocess", "total"}
    assert keys <= set(pred.timings_ms)
    assert all(v >= 0 for v in pred.timings_ms.values())


@pytest.mark.parametrize("bad", [
    "not-an-image",
    b"\x00\x01\x02 not jpeg",
    b"",
    12345,
    None,
])
def test_invalid_inputs_rejected(predictor, bad):
    with pytest.raises(InvalidImageError):
        predictor.predict(bad)


def test_float_array_rejected(predictor):
    with pytest.raises(InvalidImageError):
        predictor.predict(np.zeros((64, 64, 3), dtype=np.float32))


def test_uint8_array_accepted(predictor):
    arr = np.asarray(Image.open(io.BytesIO(make_document_bytes())).convert("RGB"))
    pred = predictor.predict(arr)
    assert pred.decision in ALL_DECISIONS


def test_checkpoint_roundtrip(tiny_checkpoint):
    model, ckpt = model_from_checkpoint(tiny_checkpoint, device="cpu")
    assert ckpt["model_version"] == "test-tiny-random"
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_checkpoint_rejects_foreign_format(tmp_path):
    import torch
    p = tmp_path / "foreign.pt"
    torch.save({"weights": torch.zeros(3)}, p)
    with pytest.raises(ValueError, match="not an IDGuard"):
        model_from_checkpoint(p)
