"""Native-resolution patch analysis for identity-document tampering.

Why patches: SIDTD documents are 2167x1360. Resizing a whole card to 256 px
(the whole-image baseline recipe) leaves rewritten text fields ~2 px tall, so
the manipulation is erased before the network sees it. Measured on this
dataset, whole-document classification sits near chance (val ROC-AUC ~0.57-0.62),
while frozen ImageNet features on **native-resolution 256 px patches** separate
a manipulated-field patch from an untouched patch of the *same* image with
~0.98 accuracy (`experiments/tampering/patch_probe.py`, within-image control).

This module provides the pieces shared by training, evaluation and inference:

* `region_crop_box` / `far_crop_box` — supervised crops from the annotation-
  derived tamper regions (positive) and untouched crops of the same document
  (hard negative, same compression history).
* `tile_boxes` — deterministic overlapping tiling used to score a full document.
* `score_document` — max-pooled document score plus a tile-resolution
  probability map (the patch heatmap) and thresholded regions.

The patch heatmap is a coarse, tile-resolution localisation map (stride 192 px),
not a pixel-accurate mask.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import Sample
from .forensic import FIELD_ALIASES, derived_region_for_fake, field_rects

DEFAULT_PATCH = 256
DEFAULT_STRIDE = 192
MAX_REGION_SIDE = 448          # cap for context crops around large regions
Box = Tuple[int, int, int, int]


def _square(cx: float, cy: float, side: int, w: int, h: int) -> Box:
    side = int(max(1, min(side, w, h)))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, w - side))
    y0 = max(0, min(y0, h - side))
    return (x0, y0, side, side)


def region_crop_box(region: Box, w: int, h: int, patch: int = DEFAULT_PATCH,
                    expand: float = 2.2, jitter: float = 0.0,
                    rng: Optional[random.Random] = None,
                    max_side: int = MAX_REGION_SIDE) -> Box:
    """Square context crop centred on an annotation-derived tamper region.

    `jitter` (fraction of the crop side) shifts the centre so the model sees the
    region at varying positions instead of always dead-centre.
    """
    x, y, rw, rh = region
    side = int(min(max_side, max(patch, max(rw, rh) * expand)))
    cx, cy = x + rw / 2.0, y + rh / 2.0
    if jitter and rng is not None:
        cx += rng.uniform(-jitter, jitter) * side
        cy += rng.uniform(-jitter, jitter) * side
    return _square(cx, cy, side, w, h)


def far_crop_box(region: Optional[Box], w: int, h: int, patch: int = DEFAULT_PATCH,
                 rng: Optional[random.Random] = None,
                 max_overlap: float = 0.10) -> Box:
    """Random patch that barely overlaps `region` (same-image hard negative).

    Overlap is measured in pixels against `max_overlap * patch^2`; if no such
    patch exists (region covering most of the card) the last draw is returned.
    """
    rng = rng or random.Random(0)
    px = py = 0
    for _ in range(50):
        px = rng.randint(0, max(w - patch, 0))
        py = rng.randint(0, max(h - patch, 0))
        if region is None:
            break
        rx, ry, rw, rh = region
        ix = max(0, min(px + patch, rx + rw) - max(px, rx))
        iy = max(0, min(py + patch, ry + rh) - max(py, ry))
        if ix * iy <= max_overlap * patch * patch:
            break
    return (px, py, patch, patch)


def tile_boxes(w: int, h: int, patch: int = DEFAULT_PATCH,
               stride: int = DEFAULT_STRIDE) -> List[Box]:
    """Overlapping tiling that covers the whole image exactly at the borders."""
    def axis(total: int) -> List[int]:
        if total <= patch:
            return [0]
        pos = list(range(0, total - patch + 1, stride))
        if pos[-1] != total - patch:
            pos.append(total - patch)
        return pos

    return [(x, y, patch, patch) for y in axis(h) for x in axis(w)]


# ---------------------------------------------------------------------------
# training samples
# ---------------------------------------------------------------------------
@dataclass
class PatchSample:
    image_path: str
    label: int
    kind: str                 # "region" (manipulated) | "far" | "genuine"
    box: Box


def build_patch_samples(root: Path, rows: Sequence[Sample], ann_dir: Path,
                        patch: int = DEFAULT_PATCH, seed: int = 42,
                        genuine_per_image: int = 2,
                        max_side: int = MAX_REGION_SIDE
                        ) -> Tuple[List[PatchSample], Dict[str, int]]:
    """Supervised patch index for one split — **field-matched** negatives.

    * tampered image -> one crop of the manipulated field (label 1) and
      `genuine_per_image` crops of *other annotated fields* of the same
      document (label 0).
    * genuine image -> `genuine_per_image` crops of its own annotated fields
      (label 0).

    Why field-matched: an earlier version drew negatives from random positions.
    Positives then always sat on a text field while negatives often landed on
    empty template areas, so the head learned "text-dense vs empty" (patch AUC
    0.94, but document AUC 0.55 and a within-image accuracy of only 0.67 once
    negatives were field-matched — see `experiments/tampering/patch_probe2.py`).
    Sampling both classes from annotated fields removes that confound.

    Fakes whose field annotation cannot be resolved are skipped and counted;
    their pixels are never labelled tampered on a guess.
    """
    rng = random.Random(seed)
    samples: List[PatchSample] = []
    skipped = {"tampered_without_region_annotation": 0}
    for s in rows:
        with Image.open(Path(root) / s.image_path) as im:
            w, h = im.size
        ref = s.src_image if s.label == 1 else s.image_path
        rects = field_rects(ref, str(ann_dir))
        if s.label == 0:
            chosen = rng.sample(sorted(rects), min(genuine_per_image, len(rects))) if rects else []
            for field in chosen:
                samples.append(PatchSample(s.image_path, 0, "genuine",
                                           region_crop_box(rects[field], w, h, patch,
                                                           jitter=0.30, rng=rng,
                                                           max_side=max_side)))
            if not chosen:   # no annotation: fall back to a random crop
                samples.append(PatchSample(s.image_path, 0, "genuine",
                                           far_crop_box(None, w, h, patch, rng)))
            continue

        region = derived_region_for_fake(
            {"src_image": s.src_image, "tamper_field": s.tamper_field}, str(ann_dir))
        if region is None:
            skipped["tampered_without_region_annotation"] += 1
            continue
        samples.append(PatchSample(s.image_path, 1, "region",
                                   region_crop_box(region, w, h, patch,
                                                   jitter=0.30, rng=rng,
                                                   max_side=max_side)))
        want = FIELD_ALIASES.get(s.tamper_field, s.tamper_field)
        others = [f for f in sorted(rects) if f != want]
        rng.shuffle(others)
        for field in others[:max(1, genuine_per_image)]:
            samples.append(PatchSample(s.image_path, 0, "far",
                                       region_crop_box(rects[field], w, h, patch,
                                                       jitter=0.30, rng=rng,
                                                       max_side=max_side)))
    return samples, skipped


class PatchDataset(Dataset):
    """Crops one patch per item. Returns tensors ready for `TamperingModel`."""

    def __init__(self, root: Path, samples: Sequence[PatchSample], image_tf):
        self.root = Path(root)
        self.samples = list(samples)
        self.tf = image_tf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict:
        s = self.samples[i]
        x, y, w, h = s.box
        with Image.open(self.root / s.image_path) as im:
            tile = im.convert("RGB").crop((x, y, x + w, y + h))
        return {"image": self.tf(tile), "label": s.label, "kind": s.kind, "index": i}

    @property
    def labels(self) -> List[int]:
        return [s.label for s in self.samples]


# ---------------------------------------------------------------------------
# document scoring
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


@torch.no_grad()
def score_document(model, pil: Image.Image, image_tf, device: str = "cpu",
                   patch: int = DEFAULT_PATCH, stride: int = DEFAULT_STRIDE,
                   batch: int = 16, temperature: float = 1.0,
                   heatmap: bool = True) -> Dict:
    """Score every tile of `pil`; the document score is the max tile logit.

    Returns dict with `logit` (raw max), `probability` (calibrated), `boxes`,
    `tile_probs`, and — when `heatmap=True` — `heatmap`, a float32 HxW
    probability map at tile resolution, pixel-aligned with the source image.
    """
    model.eval()
    w, h = pil.size
    boxes = tile_boxes(w, h, patch, stride)
    logits: List[np.ndarray] = []
    for i in range(0, len(boxes), batch):
        tiles = torch.stack([image_tf(pil.crop((x, y, x + bw, y + bh)))
                             for x, y, bw, bh in boxes[i:i + batch]])
        logits.append(model(tiles.to(device)).reshape(-1).float().cpu().numpy())
    logits = np.concatenate(logits) if logits else np.zeros(0, dtype=np.float32)
    tile_probs = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60, 60)))

    out = {
        "logit": float(logits.max()) if logits.size else 0.0,
        "probability": float(tile_probs.max()) if tile_probs.size else 0.0,
        "boxes": boxes,
        "tile_probs": tile_probs,
    }
    if heatmap:
        heat = np.zeros((h, w), dtype=np.float32)
        for (x, y, bw, bh), p in zip(boxes, tile_probs):
            heat[y:y + bh, x:x + bw] = np.maximum(heat[y:y + bh, x:x + bw], p)
        out["heatmap"] = heat
    return out


def regions_from_tiles(boxes: Sequence[Box], probs: Sequence[float], w: int, h: int,
                       threshold: float = 0.5, max_regions: int = 5,
                       iou_merge: float = 0.30) -> List[List[float]]:
    """Thresholded tiles -> normalized [x0,y0,x1,y1] boxes, best first."""
    order = np.argsort(-np.asarray(probs)) if len(probs) else []
    out: List[List[float]] = []
    kept: List[Tuple[float, float, float, float]] = []
    for i in order:
        if float(probs[i]) < threshold:
            break
        x, y, bw, bh = boxes[i]
        # tiles can overhang a document smaller than the patch: clip to the image
        cand = (max(0.0, x / w), max(0.0, y / h),
                min(1.0, (x + bw) / w), min(1.0, (y + bh) / h))
        if any(_iou(cand, k) > iou_merge for k in kept):
            continue
        kept.append(cand)
        out.append([round(v, 4) for v in cand])
        if len(out) >= max_regions:
            break
    return out


def _iou(a, b) -> float:  # normalized boxes
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)
