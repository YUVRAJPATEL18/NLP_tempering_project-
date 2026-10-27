from fastapi.testclient import TestClient

from app import app


def test_standalone_app_exposes_ocr_health_endpoint():
    response = TestClient(app).get("/ocr/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
