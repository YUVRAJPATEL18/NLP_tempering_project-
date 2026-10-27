import numpy as np

from ocr_module.config import settings
from ocr_module.preprocessing.image_loader import ImageLoader
from ocr_module.validation.image_quality_gate import ImageQualityGate


def test_low_resolution_is_rejected_before_resize():
    _, measurements = ImageLoader.load_image(np.full((240, 320, 3), 200, dtype=np.uint8))
    result = ImageQualityGate.assess(measurements)
    assert result["accepted"] is False
    assert "LOW_TEXT_RESOLUTION" in result["issues"]


def test_clear_textured_image_at_different_resolution_passes_gate():
    image = np.full((800, 1000, 3), 245, dtype=np.uint8)
    # A deterministic grid provides sharp edges without invoking OCR.
    for y in range(30, 770, 40):
        image[y:y + 3, 80:920] = 20
    _, measurements = ImageLoader.load_image(image)
    result = ImageQualityGate.assess(measurements)
    assert result["accepted"] is True


def test_dark_low_contrast_is_rejected():
    _, measurements = ImageLoader.load_image(np.full((700, 900, 3), 10, dtype=np.uint8))
    result = ImageQualityGate.assess(measurements)
    assert result["accepted"] is False
    assert "UNDEREXPOSED" in result["issues"]


def test_resolution_cutoff_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "quality_min_short_edge", 700)
    _, measurements = ImageLoader.load_image(np.full((650, 900, 3), 210, dtype=np.uint8))
    assert "LOW_TEXT_RESOLUTION" in ImageQualityGate.assess(measurements)["issues"]
