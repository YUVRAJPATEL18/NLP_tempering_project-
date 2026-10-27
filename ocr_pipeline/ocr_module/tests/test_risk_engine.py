from ocr_module.schemas import FieldExtractionValue
from ocr_module.validation.risk_engine import RiskEngine


def test_risk_engine_reports_unavailable_without_fields():
    score, category, reasons = RiskEngine().score({})
    assert score is None
    assert category == "unavailable"
    assert reasons


def test_risk_engine_uses_only_present_signals_and_explains_them():
    fields = {
        "date_of_birth": FieldExtractionValue(value="bad", confidence=0.2, validation_status="invalid")
    }
    score, category, reasons = RiskEngine().score(fields)
    assert score == 0.9
    assert category == "review"
    assert len(reasons) == 2
