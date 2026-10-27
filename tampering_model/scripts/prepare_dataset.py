#!/usr/bin/env python3
"""Prepare the SIDTD dataset for tampering-model training.

    .venv/bin/python scripts/prepare_dataset.py [--rebuild-splits]

  1. Loads the manifest built by scripts/download_sidtd.py.
  2. Validates every file (full pixel decode) and reports corrupt/missing ones.
  3. Creates deterministic, group-aware 70/15/15 train/val/test splits
     (genuine image + all its tampered variants stay together) — the official
     split CSVs leak base documents across splits, so they are not used for
     training (see docs/tampering_model_audit.md).
  4. Computes forensic quality metrics for every image and fixes the
     degraded-quality threshold as the 33rd percentile of *train-split genuine*
     quality scores (a pre-registered rule — never computed on the test split).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idguard.dataset import (build_group_splits, load_manifest,  # noqa: E402
                             load_splits, save_splits, split_digest,
                             validate_images)
from idguard.forensic import quality_metrics  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--rebuild-splits", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data"]["root"])
    manifest_path = Path(cfg["data"]["manifest"])
    splits_path = Path(cfg["data"]["splits_file"])

    samples = load_manifest(manifest_path)
    print(f"manifest: {len(samples)} images")

    # 1. integrity ---------------------------------------------------------
    rep = validate_images(root, samples)
    print(f"validation: ok={rep.ok} missing={len(rep.missing)} "
          f"unreadable={len(rep.unreadable)} too_small={len(rep.wrong_mode)}")
    if rep.invalid:
        bad = set(rep.invalid)
        samples = [s for s in samples if s.image_path not in bad]
        print(f"-> excluded {len(bad)} invalid files from all splits")
    (root / "validation_report.json").write_text(json.dumps(rep.to_dict(), indent=1))

    # 2. splits ------------------------------------------------------------
    if splits_path.exists() and not args.rebuild_splits:
        splits = load_splits(splits_path)
        print(f"using existing splits ({splits_path})")
    else:
        splits = build_group_splits(samples, ratios=tuple(cfg["data"]["split_ratios"]),
                                    seed=int(cfg["data"]["split_seed"]))
        save_splits(splits_path, splits, {
            "ratios": list(cfg["data"]["split_ratios"]),
            "seed": int(cfg["data"]["split_seed"]),
            "policy": "group-disjoint by base document",
            "digest": split_digest(splits),
        })
        print(f"wrote splits ({splits_path})")

    groups = splits.get("_groups")
    if groups:
        tr, va, te = set(groups["train"]), set(groups["val"]), set(groups["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), "GROUP LEAKAGE!"
        print(f"group-disjoint verified: {len(tr)}/{len(va)}/{len(te)} base documents")

    by_split = {}
    for name in ("train", "val", "test"):
        rows = [s for s in samples if s.image_path in set(splits[name])]
        n_fake = sum(s.label for s in rows)
        by_split[name] = (len(rows), n_fake)
        print(f"  {name:5s}: {len(rows):4d} images ({len(rows) - n_fake} genuine / {n_fake} tampered)")

    # 3. quality table + degraded threshold --------------------------------
    quality_rows = []
    for s in samples:
        try:
            with Image.open(root / s.image_path) as im:
                m = quality_metrics(im.convert("RGB"))
        except Exception as e:  # noqa: BLE001
            print(f"  quality failed for {s.image_path}: {e}")
            continue
        split = next(n for n in ("train", "val", "test") if s.image_path in set(splits[n]))
        quality_rows.append({"image_path": s.image_path, "split": split, "label": s.label, **m})

    qpath = root / "quality.csv"
    with qpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
        w.writeheader()
        w.writerows(quality_rows)
    print(f"wrote quality metrics for {len(quality_rows)} images -> {qpath}")

    train_genuine = [r["quality_score"] for r in quality_rows
                     if r["split"] == "train" and r["label"] == 0]
    import numpy as np
    thr = float(np.quantile(train_genuine, 0.33)) if train_genuine else float(
        cfg["quality"]["degraded_threshold"])
    meta = {
        "degraded_threshold": round(thr, 4),
        "rule": "33rd percentile of train-split genuine quality_scores (pre-registered)",
        "n_train_genuine": len(train_genuine),
        "quality_score_range": [round(min(r["quality_score"] for r in quality_rows), 4),
                                round(max(r["quality_score"] for r in quality_rows), 4)],
    }
    (root / "quality_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"degraded threshold = {thr:.4f} -> {root / 'quality_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
