"""
Unit tests for rule-based and label-variant field extraction.
"""

import pytest

from ocr_module.nlp.field_extraction import FieldExtractor
from ocr_module.schemas import OCRResult, OCRLine


def test_field_extraction_passport_and_dob():
    lines = [
        OCRLine(text="PASSPORT NO: J1234567", confidence=0.96, bbox=[[0,0],[1,0],[1,1],[0,1]], line_number=1),
        OCRLine(text="FULL NAME: YUVRAJ PATEL", confidence=0.94, bbox=[[0,2],[1,2],[1,3],[0,3]], line_number=2),
        OCRLine(text="DATE OF BIRTH: 01/01/2004", confidence=0.95, bbox=[[0,4],[1,4],[1,5],[0,5]], line_number=3)
    ]
    ocr_res = OCRResult(
        document_id="test-123",
        raw_text="\n".join([l.text for l in lines]),
        lines=lines,
        average_confidence=0.95,
        image_width=600,
        image_height=800,
        processing_time_ms=100.0
    )

    fields = FieldExtractor.extract_fields(ocr_res)
    assert "passport_number" in fields
    assert fields["passport_number"].value == "J1234567"
    assert "name" in fields
    assert "YUVRAJ" in fields["name"].value
    assert "date_of_birth" in fields
    assert fields["date_of_birth"].value == "01/01/2004"
    assert fields["passport_number"].raw_text == "J1234567"
    assert fields["passport_number"].confidence == 0.96
    assert fields["passport_number"].bbox == [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_field_extraction_ocr_label_variants():
    # Misread label variants: N4ME, D0B, Exp1ry
    lines = [
        OCRLine(text="N4ME: ALEX SMITH", confidence=0.90, bbox=[[0,0],[1,0],[1,1],[0,1]], line_number=1),
        OCRLine(text="D0B: 15-05-1998", confidence=0.91, bbox=[[0,2],[1,2],[1,3],[0,3]], line_number=2),
        OCRLine(text="Exp1ry: 2030-05-15", confidence=0.89, bbox=[[0,4],[1,4],[1,5],[0,5]], line_number=3)
    ]
    ocr_res = OCRResult(
        document_id="test-variant",
        raw_text="\n".join([l.text for l in lines]),
        lines=lines,
        average_confidence=0.90,
        image_width=600,
        image_height=800,
        processing_time_ms=100.0
    )

    fields = FieldExtractor.extract_fields(ocr_res)
    assert "name" in fields
    assert fields["name"].value == "ALEX SMITH"
    assert "date_of_birth" in fields
    assert fields["date_of_birth"].value == "15-05-1998"
    assert "date_of_expiry" in fields
    assert fields["date_of_expiry"].value == "2030-05-15"


def test_unlabelled_aggregate_match_has_no_fabricated_confidence():
    line = OCRLine(text="noise", confidence=0.72, bbox=[[0,0],[5,0],[5,5],[0,5]], line_number=1)
    result = OCRResult(document_id="x", raw_text="DATE OF BIRTH: 2004-01-01", lines=[line],
                       average_confidence=0.72, image_width=10, image_height=10, processing_time_ms=1)
    field = FieldExtractor.extract_fields(result)["date_of_birth"]
    assert field.confidence == 0.0
    assert field.source_lines == []
    assert field.validation_status == "ocr_uncertain"
