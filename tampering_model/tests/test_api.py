"""API integration tests: the FastAPI surface and its error handling.

Uses the tiny random-weight checkpoint, so no dataset or trained model is needed.
"""
from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from api.app import create_app  # noqa: E402

from conftest import make_document_bytes  # noqa: E402


@pytest.fixture(scope="module")
def client(tiny_checkpoint):
    app = create_app(checkpoint=str(tiny_checkpoint))
    with TestClient(app) as c:
        yield c


def _files(data: bytes, name: str = "doc.jpg", mime: str = "image/jpeg"):
    return {"image": (name, io.BytesIO(data), mime)}


def test_health_reports_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["model_loaded"] is True
    assert d["model_version"] == "test-tiny-random"


def test_analyze_returns_prediction_schema(client):
    r = client.post("/v1/tampering/analyze", files=_files(make_document_bytes()))
    assert r.status_code == 200
    d = r.json()
    for key in ("tampering_probability", "tampering_detected", "quality_score",
                "ocr_reliability", "predicted_regions", "decision",
                "decision_reasons", "model_version"):
        assert key in d
    assert d["decision"] in {"NO_TAMPERING_DETECTED", "POTENTIAL_TAMPERING",
                             "RESCAN_REQUIRED", "MANUAL_REVIEW"}
    assert d["model_version"] == "test-tiny-random"


def test_analyze_accepts_ocr_and_gate_form_fields(client):
    r = client.post("/v1/tampering/analyze",
                    files=_files(make_document_bytes(seed=3)),
                    data={"ocr_reliability": "0.42", "gate_enabled": "true"})
    assert r.status_code == 200
    assert r.json()["ocr_reliability"] == 0.42


def test_heatmap_is_returned_on_request(client):
    r = client.post("/v1/tampering/analyze",
                    files=_files(make_document_bytes(seed=4)),
                    data={"include_heatmap": "true"})
    assert r.status_code == 200
    png = r.json().get("heatmap_png_base64")
    assert png and len(png) > 100


def test_out_of_range_ocr_rejected(client):
    r = client.post("/v1/tampering/analyze", files=_files(make_document_bytes()),
                    data={"ocr_reliability": "1.7"})
    assert r.status_code == 422


def test_corrupt_image_is_400(client):
    r = client.post("/v1/tampering/analyze", files=_files(b"not an image at all"))
    assert r.status_code == 400


def test_empty_upload_is_400(client):
    r = client.post("/v1/tampering/analyze", files=_files(b""))
    assert r.status_code == 400


def test_missing_file_field_is_422(client):
    r = client.post("/v1/tampering/analyze", data={"ocr_reliability": "0.5"})
    assert r.status_code == 422


def test_response_has_no_internal_paths(client):
    r = client.post("/v1/tampering/analyze", files=_files(make_document_bytes(seed=5)))
    body = r.text
    for forbidden in ("datasets/", "checkpoints/", "/Users/", "config.yaml"):
        assert forbidden not in body


def test_missing_checkpoint_fails_loudly(tmp_path):
    """A bad model path must not silently serve an untrained head."""
    from idguard.inference import TamperingPredictor

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        TamperingPredictor(checkpoint=str(tmp_path / "does_not_exist.pt"))
