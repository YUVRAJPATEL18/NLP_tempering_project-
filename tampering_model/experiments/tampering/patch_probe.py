#!/usr/bin/env python3
"""Diagnostic probe: is there a *local* manipulation signal at native resolution?

Whole-document classification on SIDTD (all images are 2167x1360) resizes the
entire card to 256 px, where rewritten text fields are a few pixels tall. This
probe asks whether native-resolution 256x256 patches contain learnable evidence
once the image-level compression shortcut (ELA differs systematically between
SIDTD reals and fakes) is factored out.

Two numbers matter:

* cross-image patch AUC  — separates tampered-field patches from genuine patches.
  Inflated by the reals/fakes compression difference, so it is an upper bound.
* within-image accuracy  — for each fake image, does the patch over the
  manipulated field score higher than an untouched patch of the *same* image?
  Compression quality is identical inside one image, so this is the honest
  measure of local tampering evidence.

Run:  PYTHONPATH=src .venv/bin/python experiments/tampering/patch_probe.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from idguard.dataset import load_manifest, load_splits, samples_for_split  # noqa: E402
from idguard.forensic import derived_region_for_fake  # noqa: E402

SIZE = 256
DATA = ROOT / "datasets" / "sidtd"
ANN = DATA / "templates" / "Annotations" / "reals"
OUT = ROOT / "experiments" / "tampering" / "results" / "patch_probe.json"


def pick_far_patch(w: int, h: int, box, rng: random.Random):
    """Random patch that does not overlap `box` (x, y, w, h) by more than 10%."""
    for _ in range(40):
        x = rng.randint(0, max(w - SIZE, 0))
        y = rng.randint(0, max(h - SIZE, 0))
        if box is None:
            return x, y
        bx, by, bw, bh = box
        ix = max(0, min(x + SIZE, bx + bw) - max(x, bx))
        iy = max(0, min(y + SIZE, by + bh) - max(y, by))
        if ix * iy < 0.10 * SIZE * SIZE:
            return x, y
    return rng.randint(0, max(w - SIZE, 0)), rng.randint(0, max(h - SIZE, 0))


def region_patch(w: int, h: int, box, rng: random.Random):
    """Patch centred on the manipulated field, expanded 2.2x and clamped."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(SIZE, int(max(bw, bh) * 2.2))
    x0 = int(np.clip(cx - side / 2, 0, max(w - side, 0)))
    y0 = int(np.clip(cy - side / 2, 0, max(h - side, 0)))
    return x0, y0, min(side, w), min(side, h)


def main() -> int:
    torch.manual_seed(42)
    rng = random.Random(42)
    samples = load_manifest(DATA / "manifest.csv")
    splits = load_splits(DATA / "splits.json")

    backbone = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
    backbone.fc = nn.Identity()
    backbone.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone.to(device)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def feats(patches):
        out = []
        with torch.no_grad():
            for i in range(0, len(patches), 32):
                batch = torch.stack([tf(p) for p in patches[i:i + 32]]).to(device)
                out.append(backbone(batch).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def build(split: str):
        pos, neg_genuine, neg_same = [], [], []   # patches
        meta_pos = []                              # keys of within-image pairs
        for s in samples_for_split(samples, splits, split):
            with Image.open(DATA / s.image_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if s.label == 0:
                    for _ in range(2):
                        x, y = pick_far_patch(w, h, None, rng)
                        neg_genuine.append(im.crop((x, y, x + SIZE, y + SIZE)))
                else:
                    region = derived_region_for_fake(
                        {"src_image": s.src_image, "tamper_field": s.tamper_field}, str(ANN))
                    if region is None:
                        continue
                    x, y, sw, sh = region_patch(w, h, region, rng)
                    pos.append(im.crop((x, y, x + sw, y + sh)).resize(
                        (SIZE, SIZE), Image.BILINEAR))
                    fx, fy = pick_far_patch(w, h, region, rng)
                    neg_same.append(im.crop((fx, fy, fx + SIZE, fy + SIZE)))
                    meta_pos.append(s.image_path)
        print(f"[{split}] tampered patches={len(pos)} genuine patches={len(neg_genuine)} "
              f"same-image negatives={len(neg_same)}")
        return pos, neg_genuine, neg_same

    tr_pos, tr_gen, tr_same = build("train")
    va_pos, va_gen, va_same = build("val")

    Xtr = feats(tr_pos + tr_gen + tr_same)
    ytr = np.array([1] * len(tr_pos) + [0] * (len(tr_gen) + len(tr_same)))
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    clf.fit(scaler.transform(Xtr), ytr)

    def score(patches):
        if not patches:
            return np.zeros(0)
        return clf.predict_proba(scaler.transform(feats(patches)))[:, 1]

    s_pos, s_gen, s_same = score(va_pos), score(va_gen), score(va_same)
    cross_auc = roc_auc_score([1] * len(s_pos) + [0] * len(s_gen),
                              np.concatenate([s_pos, s_gen]))
    within = float(np.mean(s_pos > s_same)) if len(s_pos) else float("nan")
    rng_shift = float(np.mean(s_pos - s_same))
    res = {
        "probe": "frozen ImageNet ResNet-18 features + logistic regression on 256px native patches",
        "split": "val",
        "n_pos": len(s_pos), "n_genuine": len(s_gen),
        "cross_image_patch_auc": round(float(cross_auc), 4),
        "within_image_pair_accuracy": round(within, 4),
        "within_image_score_delta_mean": round(rng_shift, 4),
        "note": "cross-image AUC is inflated by the reals/fakes compression difference; "
                "within-image accuracy controls for it",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
