"""
Configuration settings for IDGuard OCR module using Pydantic Settings.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRConfig(BaseSettings):
    """System-wide configuration settings."""

    # Core OCR Engine settings
    ocr_engine: str = Field(default="paddle", description="OCR engine: paddle, easyocr, tesseract, mock")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration if available")
    paddle_lang: str = Field(default="en", description="Language model for PaddleOCR")
    min_confidence: float = Field(default=0.40, description="Minimum line confidence threshold")
    risk_review_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Heuristic review-priority cutoff; not calibrated")

    # Early upload quality gate. Provisional starting values; calibrate on reviewed documents.
    quality_min_short_edge: int = Field(default=360, ge=1, description="Minimum shorter image dimension before OCR")
    quality_reject_blur_score: float = Field(default=100.0, ge=0.0, description="Provisional Laplacian variance rejection cutoff")
    quality_warning_blur_score: float = Field(default=180.0, ge=0.0, description="Provisional Laplacian variance warning cutoff")
    quality_reject_brightness: float = Field(default=25.0, ge=0.0, le=255.0)
    quality_warning_brightness: float = Field(default=40.0, ge=0.0, le=255.0)
    quality_reject_glare_fraction: float = Field(default=0.98, ge=0.0, le=1.0)
    quality_warning_glare_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    ocr_uncertain_threshold: float = Field(default=0.40, ge=0.0, le=1.0, description="Minimum mean OCR confidence for downstream risk scoring")

    # Image Preprocessing settings
    max_image_dimension: int = Field(default=2048, description="Max dimension for resizing large images")
    min_image_dimension: int = Field(default=600, description="Min dimension for upscaling low-res images")
    default_preprocessing_variant: str = Field(default="auto", description="Variant or 'auto' multi-variant selection")
    save_intermediate_images: bool = Field(default=False, description="Whether to save preprocessed images")

    # API and File upload limits
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024, description="10MB file size limit")
    allowed_mime_types: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )

    # Privacy and security
    mask_sensitive_logs: bool = Field(default=True, description="Mask sensitive fields (passport/DOB) in logs")
    temp_dir_retention_seconds: int = Field(default=300, description="Temporary file retention limit")

    model_config = SettingsConfigDict(
        env_prefix="IDGUARD_",
        env_file=".env",
        extra="ignore"
    )


settings = OCRConfig()
