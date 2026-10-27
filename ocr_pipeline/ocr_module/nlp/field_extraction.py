"""
Rule-based and fuzzy-label field extraction system.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional
from ..schemas import OCRResult, OCRLine, FieldExtractionValue
from .spelling_correction import ContextualOCRCorrector


class FieldExtractor:
    """Extracts identity document fields using spatial bounding box lines and regex rules."""

    LABEL_PATTERNS = {
        "name": [
            r"(?:full\s*name|given\s*name|surname|name|n4me)\s*[:\-]?\s*([A-Za-z\s]{3,40})",
            r"name\s*of\s*holder\s*[:\-]?\s*([A-Za-z\s]{3,40})"
        ],
        "passport_number": [
            r"(?:passport\s*no|passport\s*number|passport\s*n0|doc\s*no)\s*[:\-]?\s*([A-Z0-9]{6,10})",
            r"\b([A-Z]{1,2}[0-9]{6,8})\b"
        ],
        "student_id": [
            r"(?:student\s*id|roll\s*no|reg\s*no|enrollment\s*no|id\s*no|student\s*n0)\s*[:\-]?\s*([A-Z0-9\-\/]{4,20})",
            r"\b([0-9]{7,12})\b"
        ],
        "date_of_birth": [
            r"(?:date\s*of\s*birth|dob|birth|born|d0b)\s*[:\-]?\s*(\d{1,4}[./\s\-][a-zA-Z0-9]{1,3}[./\s\-]\d{1,4})",
            r"\b(\d{2}[./\-]\d{2}[./\-]\d{4})\b"
        ],
        "date_of_issue": [
            r"(?:date\s*of\s*issue|issue\s*date|doi|issued)\s*[:\-]?\s*(\d{1,4}[./\s\-][a-zA-Z0-9]{1,3}[./\s\-]\d{1,4})"
        ],
        "date_of_expiry": [
            r"(?:date\s*of\s*expiry|expiry\s*date|expiry|exp1ry|valid\s*till|validity)\s*[:\-]?\s*(\d{1,4}[./\s\-][a-zA-Z0-9]{1,3}[./\s\-]\d{1,4})"
        ],
        "nationality": [
            r"(?:nationality|nation|citizenship)\s*[:\-]?\s*([A-Za-z]{3,20})",
            r"\b(INDIAN|AMERICAN|BRITISH|CANADIAN|GERMAN|FRENCH|AUSTRALIAN)\b"
        ],
        "college": [
            r"(?:college|institute|university|un1versity|school)\s*[:\-]?\s*([A-Za-z0-9\s,&.\-]{4,60})"
        ],
        "course": [
            r"(?:course|branch|program|degree)\s*[:\-]?\s*([A-Za-z0-9\s,&.\-]{3,50})"
        ],
        "address": [
            r"(?:address|residence|addr)\s*[:\-]?\s*([A-Za-z0-9\s,#.\-\n]{10,100})"
        ]
    }

    @staticmethod
    def extract_mrz(lines: List[OCRLine]) -> Optional[Dict[str, Any]]:
        """Extracts and parses Machine Readable Zone (MRZ) lines if present."""
        mrz_lines = []
        source_line_nums = []

        for line in lines:
            t = line.text.replace(" ", "").upper()
            if "<<" in t or (len(t) >= 28 and re.search(r"^[A-Z0-9<]{28,44}$", t)):
                mrz_lines.append(t)
                source_line_nums.append(line.line_number)

        if mrz_lines:
            combined_mrz = "\n".join(mrz_lines)
            return {
                "value": combined_mrz,
                "raw_text": combined_mrz,
                "confidence": sum(line.confidence for line in lines if line.line_number in source_line_nums) / len(source_line_nums),
                "source_lines": source_line_nums,
                "bbox": FieldExtractor._combined_bbox([line for line in lines if line.line_number in source_line_nums]),
                "validation_status": "valid"
            }
        return None

    @staticmethod
    def _combined_bbox(lines: List[OCRLine]):
        points = [point for line in lines for point in line.bbox]
        if not points:
            return None
        left, top = min(p[0] for p in points), min(p[1] for p in points)
        right, bottom = max(p[0] for p in points), max(p[1] for p in points)
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    @staticmethod
    def extract_fields(ocr_result: OCRResult) -> Dict[str, FieldExtractionValue]:
        """
        Extracts structured fields from lines in OCRResult.
        """
        extracted: Dict[str, FieldExtractionValue] = {}
        full_text = ocr_result.raw_text
        lines = ocr_result.lines

        # Check MRZ first
        mrz_data = FieldExtractor.extract_mrz(lines)
        if mrz_data:
            extracted["mrz"] = FieldExtractionValue(**mrz_data)

        # Extract labeled fields line by line
        for field_name, patterns in FieldExtractor.LABEL_PATTERNS.items():
            if field_name in extracted:
                continue

            found_val = None
            source_nums = []
            conf = None
            source_line = None

            for line in lines:
                text = line.text
                for pat in patterns:
                    match = re.search(pat, text, re.IGNORECASE)
                    if match:
                        found_val = match.group(1).strip()
                        source_nums = [line.line_number]
                        conf = line.confidence
                        source_line = line
                        break
                if found_val:
                    break

            # If not found line by line, try multi-line regex on full text
            if not found_val:
                for pat in patterns:
                    match = re.search(pat, full_text, re.IGNORECASE)
                    if match:
                        found_val = match.group(1).strip()
                        source_nums = []
                        break

            if found_val:
                # Apply contextual OCR corrections if needed
                raw_value = unicodedata.normalize("NFKC", found_val)
                raw_value = " ".join(raw_value.split())
                normalized_value = raw_value
                if field_name == "passport_number":
                    normalized_value, _ = ContextualOCRCorrector.correct_passport_number(raw_value)
                elif field_name == "name":
                    normalized_value, _ = ContextualOCRCorrector.correct_name_field(raw_value)

                extracted[field_name] = FieldExtractionValue(
                    value=raw_value,
                    raw_text=raw_value,
                    confidence=round(conf, 4) if conf is not None else 0.0,
                    source_lines=source_nums,
                    bbox=FieldExtractor._combined_bbox([source_line] if source_line else []),
                    normalized_value=normalized_value if normalized_value != raw_value else None,
                    validation_status="unknown" if conf is not None else "ocr_uncertain",
                    validation_message=None if conf is not None else "Field was matched in aggregated OCR text without a reliable source line/confidence."
                )

        return extracted
