"""
FastAPI REST API routes for IDGuard OCR system.
"""

import time
import os
import uuid
import logging
from typing import Optional, Union
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from ..schemas import ExtractionResponse, HealthResponse, QualityGateResponse
from ..config import settings
from ..preprocessing.pipeline import PreprocessingPipeline, PreprocessingVariant
from ..engine.paddle_ocr_engine import PaddleOCREngine, FallbackOCREngine
from ..postprocessing.text_postprocessor import TextPostprocessor
from ..nlp.nlp_pipeline import NLPPipeline
from ..validation.document_validator import DocumentValidator
from ..exceptions import InvalidImageError, OCRError
from ..validation.risk_engine import RiskEngine
from ..preprocessing.image_loader import ImageLoader
from ..validation.image_quality_gate import ImageQualityGate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

# Initialize singletons
_paddle = PaddleOCREngine()
engine = _paddle if _paddle.is_available() else FallbackOCREngine()
pipeline = PreprocessingPipeline()
postprocessor = TextPostprocessor()
nlp_pipeline = NLPPipeline()
risk_engine = RiskEngine(
    low_confidence_threshold=settings.min_confidence,
    review_threshold=settings.risk_review_threshold,
)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns OCR engine status, model availability, supported formats, and CPU/GPU execution mode.
    """
    return HealthResponse(
        status="healthy",
        engine_name=engine.__class__.__name__,
        model_available=engine.is_available(),
        supported_formats=["image/jpeg", "image/png", "image/webp"],
        execution_mode="GPU" if settings.use_gpu else "CPU",
        version="0.1.0"
    )


@router.post("/extract", response_model=Union[ExtractionResponse, QualityGateResponse])
async def extract_ocr(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form("unknown"),
    preprocessing_mode: Optional[str] = Form("auto")
):
    """
    Upload document image for complete OCR + NLP processing and field extraction.
    """
    start_time = time.time()
    doc_id = str(uuid.uuid4())

    # 1. MIME and Extension Validation
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {settings.allowed_extensions}"
        )

    if file.content_type not in settings.allowed_mime_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image media type.")
    content = await file.read(settings.max_file_size_bytes + 1)
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {settings.max_file_size_bytes // (1024*1024)}MB."
        )

    try:
        # Decode and assess original pixels before resizing/enhancement or model work.
        original_img, quality_metrics = ImageLoader.load_image(content)
        quality = ImageQualityGate.assess(quality_metrics)
        if not quality["accepted"]:
            return QualityGateResponse(
                status="RESCAN_REQUIRED",
                message=quality["message"],
                image_quality={"status": quality["status"], "issues": quality["issues"], "measurements": quality_metrics},
                ocr={"executed": False},
                downstream_checks={
                    "face_matching": "skipped", "database_verification": "skipped",
                    "forensic_detection": "skipped", "risk_scoring": "skipped",
                },
                retry_required=True,
                recommendation="Retake the image in focus with even lighting and the entire document visible.",
            )

        # Only accepted originals proceed to preprocessing and OCR.
        target_variant = preprocessing_mode if preprocessing_mode != "auto" else "enhanced"
        processed_img, metrics = pipeline.process_single(original_img, variant=target_variant)

        # 4. OCR Inference
        ocr_res = engine.recognize(processed_img, document_id=doc_id, variant_name=metrics["applied_variant"])

        # 5. Postprocessing
        updated_ocr_res, cleaned_text = postprocessor.process(ocr_res)

        # 6. NLP Processing & Field Extraction
        nlp_out = nlp_pipeline.process(
            raw_text=updated_ocr_res.raw_text,
            cleaned_text=cleaned_text,
            ocr_result=updated_ocr_res
        )

        if updated_ocr_res.average_confidence < settings.ocr_uncertain_threshold:
            elapsed_ms = (time.time() - start_time) * 1000.0
            return ExtractionResponse(
                document_id=doc_id,
                document_type=document_type or "unknown",
                raw_ocr_result=updated_ocr_res,
                cleaned_text=cleaned_text,
                normalized_text=nlp_out["normalized_text"],
                fields={},
                fuzzy_matches={},
                warnings=["OCR confidence is below the configured threshold; downstream verification was deferred."],
                errors=[],
                processing_time_ms=round(elapsed_ms, 2),
                preprocessing_variant_used=metrics["applied_variant"],
                risk_score=None,
                risk_category="unavailable",
                reasons=["OCR_UNCERTAIN: OCR output is not reliable enough for downstream risk scoring."],
                field_forensics={},
                status="OCR_UNCERTAIN",
                image_quality={"status": quality["status"], "issues": quality["issues"], "measurements": quality_metrics},
            )

        # 7. Document Field Validation & Sensitivity Masking
        validated_fields, warnings = DocumentValidator.validate_extracted_fields(nlp_out["fields"])
        warnings.extend(nlp_out["warnings"])
        for field_name, field_value in validated_fields.items():
            if field_value.confidence < settings.min_confidence:
                warnings.append(f"Field '{field_name}' has low OCR confidence; review its transcription.")
        risk_score, risk_category, reasons = risk_engine.score(validated_fields)
        warnings.append("No tampering or forensic model is present; no authenticity finding was produced.")
        if isinstance(engine, FallbackOCREngine):
            warnings.append("PaddleOCR is unavailable; the fallback detects text-like regions only and does not recognize text.")

        elapsed_ms = (time.time() - start_time) * 1000.0

        return ExtractionResponse(
            document_id=doc_id,
            document_type=document_type or "unknown",
            raw_ocr_result=updated_ocr_res,
            cleaned_text=cleaned_text,
            normalized_text=nlp_out["normalized_text"],
            fields=validated_fields,
            fuzzy_matches={},
            warnings=warnings,
            errors=[],
            processing_time_ms=round(elapsed_ms, 2),
            preprocessing_variant_used=metrics["applied_variant"],
            risk_score=risk_score,
            risk_category=risk_category,
            reasons=reasons,
            field_forensics={},
            status="QUALITY_WARNING" if quality["issues"] else "PROCESSING_CONTINUES",
            image_quality={"status": quality["status"], "issues": quality["issues"], "measurements": quality_metrics}
        )

    except InvalidImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except OCRError as e:
        logger.error("OCR engine failure for request %s (%s)", doc_id, type(e).__name__)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OCR engine is unavailable or failed.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected OCR failure for request %s (%s)", doc_id, type(e).__name__)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OCR processing failed.")
