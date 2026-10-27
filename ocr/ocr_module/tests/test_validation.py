"""
Unit tests for date validators, ID validators, and privacy masking.
"""

import pytest

from ocr_module.validation.date_validator import DateValidator
from ocr_module.validation.id_validator import IDValidator
from ocr_module.validation.document_validator import DocumentValidator


def test_date_validator_iso_conversion():
    valid, iso, warning = DateValidator.validate_and_normalize("2024-08-15")
    assert valid is True
    assert iso == "2024-08-15"
    assert warning is None


def test_date_validator_ambiguous_warning():
    valid, iso, warning = DateValidator.validate_and_normalize("05/06/2022")
    assert valid is True
    assert warning is not None
    assert "Ambiguous" in warning


def test_passport_validation():
    status, msg = IDValidator.validate_passport("A1234567")
    assert status == "valid"
    assert msg is None

    status_bad, msg_bad = IDValidator.validate_passport("INVALID_123456789")
    assert status_bad == "invalid"


def test_sensitivity_masking():
    masked_p = DocumentValidator.mask_sensitive_value("passport_number", "A1234567")
    assert "A1****67" in masked_p or "*" in masked_p
    assert "A1234567" not in masked_p

    masked_dob = DocumentValidator.mask_sensitive_value("date_of_birth", "2004-01-01")
    assert "****-**-**" in masked_dob
