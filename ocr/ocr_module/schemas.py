"""
Pydantic schema definitions for IDGuard OCR system.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class BBoxPoint(BaseModel):
    x: int
    y: int


class OCRLine(BaseModel):
    text: str = Field(..., description="Recognized line text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: List[List[int]] = Field(..., description="Bounding box coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    line_number: int = Field(..., ge=1, description="Line number in document")


class OCRResult(BaseModel):
    document_id: str = Field(..., description="Unique ID for document request")
    raw_text: str = Field(..., description="Raw aggregated OCR text")
    lines: List[OCRLine] = Field(default_factory=list, description="Structured OCR lines")
    average_confidence: float = Field(default=0.0, description="Average line confidence score")
    image_width: int = Field(..., ge=1)
    image_height: int = Field(..., ge=1)
    processing_time_ms: float = Field(..., ge=0.0)
    preprocessing_variant: str = Field(default="original")


class CleanedTextResult(BaseModel):
    raw_text: str
    cleaned_text: str
    normalized_text: str
    tokens: List[str] = Field(default_factory=list)
    character_tokens: List[str] = Field(default_factory=list)
    comparison_text: str


class FieldExtractionValue(BaseModel):
    value: str
    raw_text: Optional[str] = Field(default=None, description="Unmodified OCR text captured for this field")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bbox: Optional[List[List[int]]] = Field(default=None, description="Field polygon in processed image coordinates")
    source_lines: List[int] = Field(default_factory=list)
    validation_status: str = Field(default="unknown", description="valid, invalid, warning, or unknown")
    validation_message: Optional[str] = None
    normalized_value: Optional[str] = None


class ExtractionResponse(BaseModel):
    document_id: str
    document_type: str = Field(default="unknown")
    raw_ocr_result: OCRResult
    cleaned_text: str
    normalized_text: str
    fields: Dict[str, FieldExtractionValue] = Field(default_factory=dict)
    fuzzy_matches: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float
    preprocessing_variant_used: str
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Heuristic review-priority score; not authenticity confidence")
    risk_category: str = "unavailable"
    reasons: List[str] = Field(default_factory=list)
    field_forensics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    status: str = "PROCESSING_CONTINUES"
    image_quality: Dict[str, Any] = Field(default_factory=dict)


class QualityGateResponse(BaseModel):
    status: str
    message: str
    image_quality: Dict[str, Any]
    ocr: Dict[str, Any]
    downstream_checks: Dict[str, str]
    retry_required: bool
    recommendation: str


class HealthResponse(BaseModel):
    status: str
    engine_name: str
    model_available: bool
    supported_formats: List[str] = Field(default_factory=list)
    execution_mode: str = Field(default="CPU")
    version: str = Field(default="0.1.0")


class EvaluationReport(BaseModel):
    cer: float = Field(..., description="Character Error Rate")
    wer: float = Field(..., description="Word Error Rate")
    exact_field_accuracy: float = Field(..., description="Exact field match ratio")
    precision: float = Field(..., description="Field-level precision")
    recall: float = Field(..., description="Field-level recall")
    f1_score: float = Field(..., description="Field-level F1 score")
    average_confidence: float
    processing_time_ms: float
    success_rate: float
    total_samples: int
