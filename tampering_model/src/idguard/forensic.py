"""Non-trainable forensic evidence helpers.

These are *complementary* signals. Per IDGuard policy an ELA map or quality metric
alone never proves tampering — they feed the calibrated decision layer and the
report/heatmap pipeline only.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

# --- quality scoring -------------------------------------------------------
# quality_score = 0.6 * sharpness + 0.4 * contrast, each clamped to [0, 1].
# Rationale (documented, fixed before evaluation):
#   * sharpness — variance of the Laplacian on grayscale, log-scaled. Blur from
#     defocus/motion/low resolution is the dominant failure mode for OCR on
#     document captures.
#   * contrast — std-dev of luminance; under/over-exposed captures collapse it.
# Noise is recorded separately for analysis but not folded into the score,
# because sensor noise varies by capture device and is a tamper cue, not a
# quality defect.
SHARPNESS_WEIGHT = 0.6
CONTRAST_WEIGHT = 0.4
SHARPNESS_SATURATION = 2000.0  # Laplacian variance that counts as fully sharp
CONTRAST_SATURATION = 70.0     # luminance std-dev that counts as full contrast


def _to_gray(img: Image.Image) -> Image.Image:
    return img.convert("L")


def laplacian_variance(gray: Image.Image) -> float:
    """Variance of the 3x3 Laplacian response (classic focus measure)."""
    lap = np.asarray(gray.filter(ImageFilter.Kernel(
        (3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)), dtype=np.float64)
    return float(lap.var())


def noise_estimate(gray: Image.Image) -> float:
    """Median absolute deviation of the high-pass residual (noise proxy)."""
    blurred = np.asarray(gray.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float64)
    res = np.asarray(gray, dtype=np.float64) - blurred
    mad = float(np.median(np.abs(res - np.median(res))))
    return mad * 1.4826


def quality_metrics(img: Image.Image) -> Dict[str, float]:
    """Compute quality metrics for an RGB/gray PIL image."""
    gray = _to_gray(img)
    arr = np.asarray(gray, dtype=np.float64)
    lap_var = laplacian_variance(gray)
    sharp = min(max(np.log1p(lap_var) / np.log1p(SHARPNESS_SATURATION), 0.0), 1.0)
    contrast = min(max(float(arr.std()) / CONTRAST_SATURATION, 0.0), 1.0)
    score = SHARPNESS_WEIGHT * sharp + CONTRAST_WEIGHT * contrast
    return {
        "quality_score": round(float(score), 4),
        "sharpness": round(float(sharp), 4),
        "contrast": round(float(contrast), 4),
        "laplacian_var": round(lap_var, 2),
        "noise": round(noise_estimate(gray), 3),
    }


# --- error level analysis --------------------------------------------------
def ela_map(img: Image.Image, quality: int = 90, scale: float = 10.0) -> np.ndarray:
    """Error Level Analysis map in [0, 1] at the image's own resolution.

    Recompress at `quality`, take the amplified absolute difference, normalize.
    Regions edited at a different compression history stand out. Diagnostic
    evidence only — never proof on its own.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(img.convert("RGB"), dtype=np.float32)
                  - np.asarray(recompressed, dtype=np.float32))
    ela = diff.max(axis=2) * scale
    return np.clip(ela / 255.0, 0.0, 1.0)


# --- annotation-derived tamper regions ------------------------------------
# manifest tamper_field values use shortened names for a few VIA field names
FIELD_ALIASES = {"number": "id_number", "surname": "name", "firstname": "name"}


def field_rects(src_image: str, field_annotations_dir: str) -> Dict[str, Tuple[int, int, int, int]]:
    """All annotated field rectangles of one genuine template image.

    Returns {field_name: (x, y, w, h)} in source pixels, or {} when the
    annotation file/entry cannot be resolved. Fakes are pixel-aligned edits of
    their source template, so these rectangles apply to the fake as well — which
    is what makes *field-matched* negative crops possible (same field, same
    position, same compression: the only difference is the manipulation).
    """
    if not src_image:
        return {}
    path, idx = _via_file_for(src_image, field_annotations_dir)
    if path is None:
        return {}
    with open(path) as f:
        via = json.load(f)
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for entry in via.get("_via_img_metadata", {}).values():
        if _entry_index(entry) != idx:
            continue
        for region in entry.get("regions", []):
            sa = region.get("shape_attributes", {})
            ra = region.get("region_attributes", {})
            name = ra.get("field_name")
            if sa.get("name") == "rect" and name and name not in out:
                out[name] = (int(sa["x"]), int(sa["y"]),
                             int(sa["width"]), int(sa["height"]))
    return out


def _entry_index(entry: Dict) -> Optional[str]:
    """VIA entries carry the image index in `filename` (``00.jpg`` -> ``00``)."""
    fn_stem = os.path.basename(entry.get("filename", "")).rsplit(".", 1)[0]
    try:
        return str(int(re.sub(r"\D", "", fn_stem) or "-1"))
    except ValueError:
        return None


def _via_file_for(src_image: str, annotations_dir: str) -> Tuple[Optional[str], str]:
    """Locate the VIA json + entry index for a template image path."""
    src_stem = os.path.basename(src_image).rsplit(".", 1)[0]
    m = re.search(r"(\d+)$", src_stem)
    if not m:
        return None, ""
    try:
        idx = str(int(m.group(1)))
    except ValueError:
        return None, ""
    prefix = src_stem[:m.start()].rstrip("_")          # alb_id_00 -> alb_id

    path = os.path.join(annotations_dir, f"{prefix}.json")
    if not os.path.exists(path):
        nation = prefix.split("_")[0]
        cands = sorted(f for f in os.listdir(annotations_dir)
                       if f.startswith(nation) and f.endswith(".json"))
        if not cands:
            return None, ""
        path = os.path.join(annotations_dir, cands[0])
    return path, idx


def derived_region_for_fake(manifest_row: Dict, field_annotations_dir: str) -> Optional[Tuple[int, int, int, int]]:
    """Resolve the manipulated-field rectangle for a fake, from official annotations.

    SIDTD provides, per fake, `src_image` (the genuine template it was derived
    from) and `tamper_field` (which document field was altered), plus VIA
    rectangle annotations of every field on the genuine templates. Fakes are
    pixel-aligned edits of their source, so (src_image, tamper_field) resolves
    to a rectangle on the fake.

    Returns (x, y, w, h) in source-image pixels, or None when no matching
    annotation exists. This yields *derived region masks* — documented as such,
    never as official pixel-level ground truth.

    Annotation files are named after the document prefix (``alb_id.json``) and
    their entries carry ``filename`` = the document's numeric index (``00.jpg``).
    """
    if not manifest_row.get("src_image") or not manifest_row.get("tamper_field"):
        return None
    want_field = FIELD_ALIASES.get(manifest_row["tamper_field"],
                                   manifest_row["tamper_field"])
    return field_rects(manifest_row["src_image"], field_annotations_dir).get(want_field)


def rasterize_region(region: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
    """Rasterize (x, y, w, h) to a boolean mask of shape `size` (h, w)."""
    mask = np.zeros((size[1], size[0]), dtype=bool)
    x, y, w, h = region
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, size[0]), min(y + h, size[1])
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask
