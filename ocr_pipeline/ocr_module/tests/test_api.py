"""
Unit tests for FastAPI endpoints: /ocr/health and /ocr/extract.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import numpy as np
import cv2
import io

from ocr_module.api.routes import router
import ocr_module.api.routes as routes
from ocr_module.schemas import OCRResult

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/ocr/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "supported_formats" in data


def test_extract_ocr_endpoint():
    # Generate synthetic image in memory
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "PASSPORT NO: A1234567", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "NAME: YUVRAJ PATEL", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    response = client.post(
        "/ocr/extract",
        files={"file": ("test_passport.png", img_bytes, "image/png")},
        data={"document_type": "passport", "preprocessing_mode": "auto"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "raw_ocr_result" in data
    assert "fields" in data
    assert "risk_score" in data
    assert "risk_category" in data
    assert "field_forensics" in data


def test_rejects_unsupported_media_type():
    response = client.post(
        "/ocr/extract",
        files={"file": ("document.png", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_rejects_corrupt_image_bytes():
    response = client.post(
        "/ocr/extract",
        files={"file": ("document.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400


def test_blurry_upload_is_rejected_before_preprocessing_or_ocr(monkeypatch):
    image = np.full((700, 900, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    def unexpected(*args, **kwargs):
        raise AssertionError("Expensive OCR/preprocessing must not run after quality rejection")

    monkeypatch.setattr(routes.pipeline, "process_single", unexpected)
    monkeypatch.setattr(routes.engine, "recognize", unexpected)
    response = client.post(
        "/ocr/extract",
        files={"file": ("blurry.png", encoded.tobytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RESCAN_REQUIRED"
    assert data["ocr"]["executed"] is False
    assert data["downstream_checks"]["face_matching"] == "skipped"
    assert data["downstream_checks"]["database_verification"] == "skipped"
    assert data["downstream_checks"]["forensic_detection"] == "skipped"


def test_low_confidence_ocr_defers_risk_scoring(monkeypatch):
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(image, "CLEAR TEXT", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    def uncertain_ocr(image, document_id=None, variant_name="original"):
        return OCRResult(
            document_id=document_id or "test", raw_text="", lines=[], average_confidence=0.1,
            image_width=600, image_height=400, processing_time_ms=1,
            preprocessing_variant=variant_name,
        )

    def unexpected_score(*args, **kwargs):
        raise AssertionError("Risk scoring must be deferred when OCR is uncertain")

    monkeypatch.setattr(routes.engine, "recognize", uncertain_ocr)
    monkeypatch.setattr(routes.risk_engine, "score", unexpected_score)
    response = client.post(
        "/ocr/extract",
        files={"file": ("clear.png", encoded.tobytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OCR_UNCERTAIN"
    assert data["risk_score"] is None
