#!/usr/bin/env python3
"""Probe #3: how much does the whole-image input resolution matter?

Frozen ImageNet ResNet-18 features + logistic regression, evaluated on the
validation split, at several square input sizes. This isolates the *representation*
from the training recipe: if 384/512 px features cannot separate genuine from
tampered documents, no amount of fine-tuning at 256 px will.

Run:  PYTHONPATH=src .venv/bin/python experiments/tampering/resolution_probe.py
"""
from __future__ import annotations

import json
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

DATA = ROOT / "datasets" / "sidtd"
OUT = ROOT / "experiments" / "tampering" / "results" / "resolution_probe.json"
SIZES = (256, 384, 512)


def main() -> int:
    torch.manual_seed(42)
    samples = load_manifest(DATA / "manifest.csv")
    splits = load_splits(DATA / "splits.json")
    train = samples_for_split(samples, splits, "train")
    val = samples_for_split(samples, splits, "val")

    backbone = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
    backbone.fc = nn.Identity()
    backbone.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone.to(device)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    results = {}
    for size in SIZES:
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])

        def feats(rows):
            out = []
            with torch.no_grad():
                buf = []
                for s in rows:
                    with Image.open(DATA / s.image_path) as im:
                        buf.append(tf(im.convert("RGB")))
                    if len(buf) >= 16:
                        out.append(backbone(torch.stack(buf).to(device)).cpu().numpy())
                        buf = []
                if buf:
                    out.append(backbone(torch.stack(buf).to(device)).cpu().numpy())
            return np.concatenate(out)

        Xtr, ytr = feats(train), np.array([s.label for s in train])
        Xva, yva = feats(val), np.array([s.label for s in val])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.05)
        clf.fit(sc.transform(Xtr), ytr)
        auc = float(roc_auc_score(yva, clf.predict_proba(sc.transform(Xva))[:, 1]))
        results[str(size)] = round(auc, 4)
        print(f"[{size}px] frozen-feature val AUC = {auc:.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "probe": "frozen ImageNet ResNet-18 global-average-pooled features + "
                 "logistic regression (C=0.05, balanced), square resize to NxN",
        "split": "val (group-disjoint)",
        "val_auc_by_input_size": results,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
