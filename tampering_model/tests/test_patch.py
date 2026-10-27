"""Patch (native-resolution) pipeline tests: tiling, supervised crops, document
scoring and the tiled inference path."""
from __future__ import annotations

import io
import random

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import make_document_bytes
from idguard.forensic import rasterize_region
from idguard.model import model_from_checkpoint
from idguard.patch import (PatchDataset, PatchSample, build_patch_samples,
                           far_crop_box, region_crop_box, regions_from_tiles,
                           score_document, tile_boxes)

CFG_DATA = {
    "image_size": 256,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------
def test_tile_boxes_cover_every_pixel():
    w, h, patch, stride = 700, 460, 256, 192
    boxes = tile_boxes(w, h, patch, stride)
    cover = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in boxes:
        assert (bw, bh) == (patch, patch)
        assert 0 <= x <= w - patch and 0 <= y <= h - patch
        cover[y:y + bh, x:x + bw] += 1
    assert cover.min() >= 1, "tiling leaves part of the document unscored"


def test_tile_boxes_handles_small_images():
    assert tile_boxes(100, 80, 256, 192) == [(0, 0, 256, 256)]


# ---------------------------------------------------------------------------
# supervised crops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("region", [(120, 80, 90, 30), (1600, 900, 300, 120)])
def test_region_crop_contains_the_region(region):
    w, h = 2167, 1360
    x, y, bw, bh = region_crop_box(region, w, h, 256, rng=random.Random(0))
    rx, ry, rw, rh = region
    assert x <= rx and y <= ry and x + bw >= rx + rw and y + bh >= ry + rh


def test_far_crop_avoids_the_region():
    rng = random.Random(0)
    region = (1000, 600, 200, 80)
    for _ in range(20):
        x, y, bw, bh = far_crop_box(region, 2167, 1360, 256, rng)
        ix = max(0, min(x + bw, region[0] + region[2]) - max(x, region[0]))
        iy = max(0, min(y + bh, region[1] + region[3]) - max(y, region[1]))
        assert ix * iy <= 0.10 * bw * bh


def test_patch_index_labels_and_skips(tmp_path):
    """Tampered rows get 1 positive + 1 same-image negative; genuine rows get
    negatives only; tampered rows without an annotation are skipped, never
    labelled on a guess."""
    from idguard.dataset import Sample

    img = Image.open(io.BytesIO(make_document_bytes((640, 480)))).convert("RGB")
    (tmp_path / "imgs").mkdir()
    img.save(tmp_path / "imgs" / "a_00.jpg", quality=95)
    img.save(tmp_path / "imgs" / "a_00_fake_6_1.jpg", quality=95)
    ann = tmp_path / "ann"
    ann.mkdir()

    rows = [
        Sample("imgs/a_00.jpg", 0, "reals", "a_00"),
        Sample("imgs/a_00_fake_6_1.jpg", 1, "fakes", "a_00",
               manipulation_type="Inpaint_and_Rewrite",
               tamper_field="name", src_image="imgs/a_00.jpg"),
    ]
    samples, skipped = build_patch_samples(tmp_path, rows, ann, patch=256, seed=0)
    # no annotation file -> fake is skipped, genuine falls back to one random crop
    assert skipped["tampered_without_region_annotation"] == 1
    assert [s.label for s in samples] == [0]

    # with VIA annotations the fake yields one positive and field-matched negatives
    via = {"_via_img_metadata": {"x": {
        "filename": "00.jpg",
        "regions": [
            {"shape_attributes": {"name": "rect", "x": 40, "y": 40,
                                  "width": 120, "height": 30},
             "region_attributes": {"field_name": "name"}},
            {"shape_attributes": {"name": "rect", "x": 300, "y": 200,
                                  "width": 150, "height": 30},
             "region_attributes": {"field_name": "expiry_date"}},
        ]}}}
    (ann / "a_id.json").write_text(__import__("json").dumps(via))
    rows[1].src_image = "imgs/a_id_00.jpg"
    samples, skipped = build_patch_samples(tmp_path, rows, ann, patch=256, seed=0)
    kinds = [s.kind for s in samples]
    assert kinds.count("region") == 1
    assert kinds.count("far") >= 1, "fake must contribute field-matched negatives"
    assert skipped["tampered_without_region_annotation"] == 0
    # every negative crop of the fake comes from a *different* annotated field
    fake_negs = [s for s in samples if s.label == 0 and s.image_path.endswith("fake_6_1.jpg")]
    assert fake_negs and all(s.kind == "far" for s in fake_negs)


def test_patch_dataset_returns_tensors(tmp_path):
    from idguard.dataset import build_transforms

    (tmp_path / "imgs").mkdir()
    Image.open(io.BytesIO(make_document_bytes((640, 480)))).save(
        tmp_path / "imgs" / "a_00.jpg")
    ds = PatchDataset(tmp_path, [PatchSample("imgs/a_00.jpg", 1, "region",
                                             (10, 10, 200, 200))],
                      build_transforms(CFG_DATA, augment=False))
    item = ds[0]
    assert item["image"].shape == (3, 256, 256)
    assert item["label"] == 1 and item["kind"] == "region"


# ---------------------------------------------------------------------------
# document scoring
# ---------------------------------------------------------------------------
def test_score_document_shapes(tiny_patch_checkpoint):
    from idguard.dataset import build_transforms

    model, _ = model_from_checkpoint(tiny_patch_checkpoint, device="cpu")
    pil = Image.open(io.BytesIO(make_document_bytes((640, 480)))).convert("RGB")
    out = score_document(model, pil, build_transforms(CFG_DATA, augment=False),
                         patch=256, stride=192)
    assert out["heatmap"].shape == (480, 640)
    assert 0.0 <= out["probability"] <= 1.0
    assert len(out["boxes"]) == len(tile_boxes(640, 480, 256, 192))
    assert np.allclose(out["heatmap"].max(), out["tile_probs"].max(), atol=1e-6)


def test_regions_from_tiles_normalized():
    boxes = [(0, 0, 256, 256), (500, 500, 256, 256), (900, 700, 256, 256)]
    probs = [0.9, 0.8, 0.1]
    regions = regions_from_tiles(boxes, probs, 1200, 1000, threshold=0.5)
    assert len(regions) == 2
    for x0, y0, x1, y1 in regions:
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    # highest-probability tile first
    assert regions[0][0] == 0.0 and regions[0][1] == 0.0


# ---------------------------------------------------------------------------
# inference through the tiled path
# ---------------------------------------------------------------------------
def test_patch_predictor_uses_tiling(patch_predictor, sharp_image_bytes):
    from idguard.decision_engine import ALL_DECISIONS

    assert patch_predictor.patch_cfg == {"patch_size": 256, "stride": 192}
    pred = patch_predictor.predict(sharp_image_bytes, include_heatmap=True)
    assert pred.decision in ALL_DECISIONS
    assert 0.0 <= pred.tampering_probability <= 1.0
    # patch heatmap is aligned to the *source* image, not the network input
    src_size = Image.open(io.BytesIO(sharp_image_bytes)).size  # (w, h)
    assert pred.heatmap.shape == (src_size[1], src_size[0])
    assert float(pred.heatmap.max()) <= 1.0 + 1e-6
    assert set(pred.timings_ms) >= {"decode", "quality", "preprocess",
                                    "inference", "postprocess", "total"}
    for box in pred.predicted_regions or []:
        x0, y0, x1, y1 = box
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def test_patch_predictor_overlay_keeps_source_size(patch_predictor, sharp_image_bytes):
    pred, overlay = patch_predictor.predict_with_overlay(sharp_image_bytes)
    src = Image.open(io.BytesIO(sharp_image_bytes))
    assert overlay.size == src.size and overlay.mode == "RGB"
    assert pred.tampering_detected == (pred.decision == "POTENTIAL_TAMPERING")


def test_mask_rasterization_matches_region():
    mask = rasterize_region((10, 20, 30, 40), (100, 100))
    assert mask.sum() == 30 * 40
    assert mask[20:60, 10:40].all()
