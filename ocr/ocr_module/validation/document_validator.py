"""
Comprehensive document validation engine and privacy log masking utility.
"""

import logging
from typing import Dict, Any, Tuple
import datetime
from ..schemas import FieldExtractionValue
from .date_validator import DateValidator
from .id_validator import IDValidator
from ..config import settings

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validates extracted document fields and masks sensitive identity information in logs."""

    @staticmethod
    def mask_sensitive_value(field_name: str, value: str) -> str:
        """Masks sensitive values (passport numbers, DOBs, student IDs) for safe logging."""
        if not settings.mask_sensitive_logs or not value:
            return value

        f_lower = field_name.lower()
        if "passport" in f_lower or "id" in f_lower:
            if len(value) > 4:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
            return "****"
        if "birth" in f_lower or "dob" in f_lower:
            return "****-**-**"
        return value

    @staticmethod
    def validate_extracted_fields(fields: Dict[str, FieldExtractionValue]) -> Tuple[Dict[str, FieldExtractionValue], list]:
        """
        Executes date, ID, and expiry validation across extracted document fields.
        """
        warnings = []
        today = datetime.date.today()

        for field_name, field_obj in fields.items():
            # Validate the corrected/normalized candidate when available, while
            # retaining field_obj.raw_text/value as the observed OCR transcription.
            val = field_obj.normalized_value or field_obj.value
            f_lower = field_name.lower()

            # Date Fields
            if any(k in f_lower for k in ["date", "dob", "expiry", "issue", "till"]):
                is_valid, iso_date, msg = DateValidator.validate_and_normalize(val)
                if is_valid and iso_date:
                    field_obj.validation_status = "valid"
                    field_obj.normalized_value = iso_date
                    if msg:
                        field_obj.validation_status = "warning"
                        field_obj.validation_message = msg
                        warnings.append(msg)

                    # Expiry Check
                    if "expiry" in f_lower or "till" in f_lower:
                        try:
                            exp_dt = datetime.date.fromisoformat(iso_date)
                            if exp_dt < today:
                                field_obj.validation_status = "invalid"
                                field_obj.validation_message = f"Document expired on {iso_date}"
                                warnings.append(f"Field '{field_name}' indicates document is expired ({iso_date}).")
                        except ValueError:
                            pass
                else:
                    field_obj.validation_status = "invalid"
                    field_obj.validation_message = msg or "Invalid date format."

            # Passport Number
            elif "passport" in f_lower:
                status, msg = IDValidator.validate_passport(val)
                field_obj.validation_status = status
                field_obj.validation_message = msg
                if status == "invalid" and msg:
                    warnings.append(msg)

            # Student ID
            elif "student" in f_lower or "id" in f_lower:
                status, msg = IDValidator.validate_student_id(val)
                field_obj.validation_status = status
                field_obj.validation_message = msg

            # MRZ
            elif "mrz" in f_lower:
                status, msg = IDValidator.validate_mrz_checksum(val)
                field_obj.validation_status = status
                field_obj.validation_message = msg

            # General Text
            else:
                if val:
                    field_obj.validation_status = "valid"

            # Log sanitized field status
            masked_val = DocumentValidator.mask_sensitive_value(field_name, val)
            logger.info(f"Validated field '{field_name}': status={field_obj.validation_status}, value={masked_val}")

        return fields, warnings
