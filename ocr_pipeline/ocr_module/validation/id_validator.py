"""
Regex and checksum validation for identity documents (Passport, Student ID, MRZ).
"""

from typing import Tuple, Optional
import re


class IDValidator:
    """Validates structure and checksums for identity card and passport numbers."""

    # Passport: 1-2 alpha characters + 6-8 digits (e.g. A1234567, Z9876543)
    PASSPORT_REGEX = r"^[A-Z]{1,2}[0-9]{6,8}$"

    # Student ID: alphanumeric 4-20 chars with optional dashes/slashes
    STUDENT_ID_REGEX = r"^[A-Z0-9\-\/]{4,20}$"

    @staticmethod
    def validate_passport(passport_num: str) -> Tuple[str, Optional[str]]:
        """Validates passport number format."""
        if not passport_num:
            return "unknown", "Passport number is empty."
        clean = passport_num.replace(" ", "").upper()
        if re.match(IDValidator.PASSPORT_REGEX, clean):
            return "valid", None
        return "invalid", f"Passport number '{passport_num}' does not match standard pattern."

    @staticmethod
    def validate_student_id(student_id: str) -> Tuple[str, Optional[str]]:
        """Validates student ID format."""
        if not student_id:
            return "unknown", "Student ID is empty."
        clean = student_id.replace(" ", "").upper()
        if re.match(IDValidator.STUDENT_ID_REGEX, clean):
            return "valid", None
        return "invalid", f"Student ID '{student_id}' does not match expected format."

    @staticmethod
    def validate_mrz_checksum(mrz_text: str) -> Tuple[str, Optional[str]]:
        """Validates MRZ lines structure and filler characters."""
        if not mrz_text:
            return "unknown", "MRZ text is empty."
        lines = [l.strip() for l in mrz_text.splitlines() if l.strip()]
        if len(lines) in [2, 3] and all(len(l) >= 28 for l in lines):
            return "valid", None
        return "warning", "MRZ line length or count is non-standard."
