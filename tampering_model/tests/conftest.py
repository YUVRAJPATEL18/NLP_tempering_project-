"""Shared fixtures: synthetic images and a tiny random-weight checkpoint.

No network access is required by any test (pretrained=False everywhere).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "config.yaml"


def make_document_bytes(size=(320, 240), sharp: bool = True, seed: int = 0) -> bytes:
    """Synthetic 'document-like' RGB JPEG: light background, dark text blocks."""
    rng = np.random.default_rng(seed)
    arr = np.full((size[1], size[0], 3), 235, dtype=np.float32)
    arr += rng.normal(0, 4, arr.shape)              # mild sensor-like noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=(30, 30, 30), width=2)
    d.rectangle([35, 55, 115, 150], fill=(60, 70, 120))          # "photo"
    for i in range(6):                                            # "text lines"
        y = 60 + i * 22
        d.rectangle([135, y, size[0] - 40, y + 9], fill=(25, 25, 25))
    d.text((140, 200), "IDGUARD TEST DOC 123", fill=(10, 10, 10))
    if not sharp:
        img = img.filter(ImageFilter.GaussianBlur(6))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sharp_image_bytes() -> bytes:
    return make_document_bytes(sharp=True, seed=1)


@pytest.fixture(scope="session")
def blurred_image_bytes() -> bytes:
    return make_document_bytes(sharp=False, seed=2)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A structurally-valid checkpoint with random weights (no download)."""
    from idguard.model import TamperingModel, save_checkpoint

    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = TamperingModel(pretrained=False, dropout=0.1)
    save_checkpoint(
        path, model,
        model_version="test-tiny-random",
        temperature=1.0,
        run_name="test",
        seed=0,
        image_size=256,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        threshold={"tamper_high": 0.5, "tamper_low": 0.2},
        thresholds_decision={"tamper_high": 0.5, "tamper_low": 0.2,
                             "rescan_below": 0.3, "ocr_min": 0.55},
        val_metrics={"val_auc": 0.5},
    )
    return path


@pytest.fixture(scope="session")
def tiny_patch_checkpoint(tmp_path_factory) -> Path:
    """Structurally-valid *patch-task* checkpoint: carries its tiling geometry."""
    from idguard.model import TamperingModel, save_checkpoint

    path = tmp_path_factory.mktemp("ckpt_patch") / "best.pt"
    model = TamperingModel(pretrained=False, dropout=0.1, freeze_backbone=True)
    save_checkpoint(
        path, model,
        model_version="test-tiny-patch",
        task="patch",
        patch={"patch_size": 256, "stride": 192},
        temperature=1.0,
        run_name="test-patch",
        seed=0,
        image_size=256,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        threshold={"tamper_high": 0.5, "tamper_low": 0.2},
        thresholds_decision={"tamper_high": 0.5, "tamper_low": 0.2,
                             "rescan_below": 0.3, "ocr_min": 0.55},
        val_document_metrics={"val_auc": 0.5},
    )
    return path


@pytest.fixture(scope="session")
def predictor(tiny_checkpoint):
    from idguard.inference import TamperingPredictor

    return TamperingPredictor(config_path=CONFIG, checkpoint=tiny_checkpoint,
                              device="cpu")


@pytest.fixture(scope="session")
def patch_predictor(tiny_patch_checkpoint):
    from idguard.inference import TamperingPredictor

    return TamperingPredictor(config_path=CONFIG, checkpoint=tiny_patch_checkpoint,
                              device="cpu")
