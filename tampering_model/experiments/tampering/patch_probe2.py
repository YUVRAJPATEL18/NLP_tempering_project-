#!/usr/bin/env python3
"""Probe #2: field-matched crops, scored at *document* level.

Probe #1 (patch_probe.py) reported 0.98 within-image accuracy, but its positive
crops always landed on a text field while many negative crops landed on empty
template areas — so it measured "text-dense vs empty", not "manipulated vs
genuine". The trained head that used the same sampling reached 0.94 patch AUC
yet only ~0.55 document AUC on validation: the confound confirmed.

Here negatives are crops of *other annotated fields* of the same document, so
positives and negatives share position, text density, and compression history.
Evaluation is document-level: max tile probability over the same 256/192 tiling
used at inference.

Run:  PYTHONPATH=src .venv/bin/python experiments/tampering/patch_probe2.py
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
from idguard.forensic import FIELD_ALIASES, field_rects  # noqa: E402
from idguard.patch import region_crop_box, tile_boxes  # noqa: E402

SIZE = 256
DATA = ROOT / "datasets" / "sidtd"
ANN = DATA / "templates" / "Annotations" / "reals"
OUT = ROOT / "experiments" / "tampering" / "results" / "patch_probe2.json"


def crops_for(sample, rng: random.Random, n_neg: int = 2):
    """(positives, field-matched negatives) crop boxes for one document."""
    src = sample.src_image if sample.label == 1 else sample.image_path
    rects = field_rects(src, str(ANN))
    if not rects:
        return [], []
    if sample.label == 1:
        want = FIELD_ALIASES.get(sample.tamper_field, sample.tamper_field)
        if want not in rects:
            return [], []
        others = [f for f in rects if f != want]
        rng.shuffle(others)
        negs = [rects[f] for f in others[:n_neg]]
        return [rects[want]], negs
    fields = list(rects.values())
    rng.shuffle(fields)
    return [], fields[:n_neg]


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
        transforms.Resize((SIZE, SIZE)),   # region crops have variable side (256..448)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def feats(tiles):
        out = []
        with torch.no_grad():
            for i in range(0, len(tiles), 32):
                batch = torch.stack([tf(p) for p in tiles[i:i + 32]]).to(device)
                out.append(backbone(batch).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    # ---- training crops ---------------------------------------------------
    X, y, pairs = [], [], []
    for s in samples_for_split(samples, splits, "train"):
        with Image.open(DATA / s.image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            pos, negs = crops_for(s, rng)
            pos_imgs = ([im.crop(_crop(region_crop_box(pos[0], w, h, SIZE, jitter=0.3, rng=rng)))]
                        if pos else [])
            neg_imgs = [im.crop(_crop(region_crop_box(r, w, h, SIZE, jitter=0.3, rng=rng)))
                        for r in negs]
        X += pos_imgs + neg_imgs
        y += [1] * len(pos_imgs) + [0] * len(neg_imgs)
        pairs.append((len(pos_imgs), len(neg_imgs)))
    print(f"[train] crops={len(X)} positives={sum(y)}", flush=True)

    F = feats(X)
    scaler = StandardScaler().fit(F)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.05)
    clf.fit(scaler.transform(F), np.array(y))

    # ---- document-level validation ---------------------------------------
    docs, labels, within_pos, within_neg = [], [], [], []
    for s in samples_for_split(samples, splits, "val"):
        with Image.open(DATA / s.image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            boxes = tile_boxes(w, h, SIZE, 192)
            tiles = [im.crop((x, y_, x + bw, y_ + bh)) for x, y_, bw, bh in boxes]
            pos, negs = crops_for(s, rng)
            if s.label == 1 and pos:
                within_pos.append(im.crop(_crop(region_crop_box(pos[0], w, h, SIZE,
                                                                jitter=0.15, rng=rng))))
        p = clf.predict_proba(scaler.transform(feats(tiles)))[:, 1]
        docs.append(float(p.max()))
        labels.append(s.label)
        if s.label == 1 and pos and negs:
            within_neg.append(im.crop(_crop(region_crop_box(negs[0], w, h, SIZE,
                                                            jitter=0.15, rng=rng))))

    auc = roc_auc_score(labels, docs)
    # honest within-image check with field-matched negatives
    sp = clf.predict_proba(scaler.transform(feats(within_pos)))[:, 1]
    sn = clf.predict_proba(scaler.transform(feats(within_neg)))[:, 1]
    within = float(np.mean(sp > sn))
    res = {
        "probe": "field-matched crops, frozen ImageNet ResNet-18 + logistic regression, "
                 "document score = max tile probability (256/192 tiling)",
        "split": "val",
        "n_train_crops": len(X),
        "document_auc": round(float(auc), 4),
        "within_image_pair_accuracy": round(within, 4),
        "n_within_pairs": len(sp),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def _crop(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


if __name__ == "__main__":
    raise SystemExit(main())
