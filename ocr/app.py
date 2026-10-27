"""Standalone FastAPI application for the ID-Guard OCR service."""

from fastapi import FastAPI

from ocr_module.api.routes import router

app = FastAPI(
    title="ID-Guard OCR",
    description="Standalone OCR and early image-quality assessment API.",
    version="0.2.0",
)
app.include_router(router)
