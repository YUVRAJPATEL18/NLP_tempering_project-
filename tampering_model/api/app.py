"""IDGuard tampering-detection FastAPI service.

    .venv/bin/uvicorn api.app:app --port 8080

Endpoints
    GET  /health                       — service + model status
    POST /v1/tampering/analyze         — multipart image upload -> TamperingPrediction JSON

The model is loaded once at startup and reused for every request (eval mode,
inference-only). Responses never contain internal file paths.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from idguard.inference import InvalidImageError, TamperingPredictor  # noqa: E402

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SERVICE_VERSION = "tampering-api-1.0"


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    model_loaded: bool


def create_app(config_path: Optional[str] = None,
               checkpoint: Optional[str] = None) -> FastAPI:
    state: dict = {"predictor": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["predictor"] = TamperingPredictor(config_path=config_path,
                                                checkpoint=checkpoint)
        yield
        state["predictor"] = None

    app = FastAPI(title="IDGuard Tampering Detection API",
                  version=SERVICE_VERSION, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p: Optional[TamperingPredictor] = state["predictor"]
        return HealthResponse(
            status="ok",
            service=SERVICE_VERSION,
            model_version=p.model_version if p else "not-loaded",
            model_loaded=p is not None,
        )

    @app.post("/v1/tampering/analyze")
    async def analyze(
        image: UploadFile = File(..., description="Document image (JPEG/PNG/WEBP)"),
        ocr_reliability: Optional[float] = Form(None, ge=0.0, le=1.0),
        include_heatmap: bool = Form(False),
        gate_enabled: Optional[bool] = Form(None),
    ):
        p: Optional[TamperingPredictor] = state["predictor"]
        if p is None:
            raise HTTPException(status_code=503, detail="model not loaded yet")
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"image exceeds {MAX_UPLOAD_BYTES} bytes")
        try:
            pred = p.predict(data, ocr_reliability=ocr_reliability,
                             include_heatmap=include_heatmap,
                             gate_enabled=gate_enabled)
        except InvalidImageError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return pred.to_dict(include_heatmap=include_heatmap)

    return app


app = create_app()
