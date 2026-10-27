#!/usr/bin/env python3
"""Patch-level (native-resolution) training for SIDTD tampering detection.

    PYTHONPATH=src .venv/bin/python -m idguard.train_patch --smoke
    PYTHONPATH=src .venv/bin/python -m idguard.train_patch

Recipe
------
* Backbone frozen (ImageNet ResNet-18) — the smallest update that fixes the
  real problem, which is resolution, not capacity.
* Positive patches: crops centred on the annotation-derived manipulated field.
* Negative patches: untouched crops of the *same* document (compression-matched
  hard negatives) plus random crops of genuine documents.
* Early stopping on patch-level validation ROC-AUC; then temperature scaling and
  decision thresholds are fitted on **document-level** validation scores
  (max tile logit) — never on the test split.

Outputs `checkpoints/improved_patch/best.pt` (with `patch` metadata so inference
and evaluation switch to the tiled path automatically) and
`experiments/tampering/results/improved_patch/{history,run_meta}.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import (build_transforms, load_manifest, load_splits,
                      samples_for_split, split_digest, validate_images)
from .model import TamperingModel, load_checkpoint, save_checkpoint
from .patch import (PatchDataset, build_patch_samples, score_document,
                    tile_boxes)
from .train import (collect_logits, fit_temperature, load_config, low_threshold,
                    resolve_device, select_threshold, set_seed, sigmoid,
                    train_one_epoch)

ROOT = Path(__file__).resolve().parents[2]


def _loader(ds: PatchDataset, batch: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      drop_last=False)


def _doc_scores(model, rows, root: Path, tf, device: str, patch: int, stride: int,
                batch: int = 16, want_heatmaps: bool = False):
    """Document-level (max tile logit) scores for a list of manifest rows."""
    logits, labels, times, heatmaps = [], [], [], []
    for s in rows:
        with Image.open(root / s.image_path) as im:
            pil = im.convert("RGB")
            t0 = time.perf_counter()
            r = score_document(model, pil, tf, device, patch=patch, stride=stride,
                               batch=batch, heatmap=want_heatmaps)
            times.append(time.perf_counter() - t0)
        logits.append(r["logit"])
        labels.append(s.label)
        if want_heatmaps:
            heatmaps.append(r["heatmap"])
    return np.asarray(logits), np.asarray(labels), times, heatmaps


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--smoke", action="store_true", help="one-batch sanity check only")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    run = dict(cfg["patch"])
    run_name = run["run_name"]
    seed = int(cfg["seed"])
    patch = int(run["patch_size"])
    stride = int(run["stride"])
    set_seed(seed)
    device = resolve_device(cfg["train"].get("device", "auto"))
    out_dir = ROOT / "experiments" / "tampering" / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ROOT / "checkpoints" / run_name / "best.pt"
    print(f"[patch] device={device} seed={seed} patch={patch} stride={stride}", flush=True)

    root = Path(cfg["data"]["root"])
    ann_dir = root / "templates" / "Annotations" / "reals"
    samples = load_manifest(Path(cfg["data"]["manifest"]))
    report = validate_images(root, samples)
    if report.invalid:
        bad = set(report.invalid)
        samples = [s for s in samples if s.image_path not in bad]
        print(f"[patch] excluded {len(bad)} invalid files", flush=True)
    splits = load_splits(Path(cfg["data"]["splits_file"]))

    tr_rows = samples_for_split(samples, splits, "train")
    va_rows = samples_for_split(samples, splits, "val")
    tr_idx, skip_tr = build_patch_samples(root, tr_rows, ann_dir, patch, seed,
                                          int(run.get("genuine_per_image", 2)))
    va_idx, skip_va = build_patch_samples(root, va_rows, ann_dir, patch, seed + 1,
                                          int(run.get("genuine_per_image", 2)))
    print(f"[patch] train patches={len(tr_idx)} (tampered {sum(s.label for s in tr_idx)}) "
          f"val patches={len(va_idx)} skipped={skip_tr} {skip_va}", flush=True)
    if not tr_idx or not va_idx:
        raise SystemExit("[patch] empty patch index — check the dataset/annotations")

    tf_train = build_transforms(cfg["data"], augment=bool(run.get("augment", True)))
    tf_eval = build_transforms(cfg["data"], augment=False)
    workers = int(cfg["data"].get("num_workers", 0))
    batch_size = int(run["batch_size"])
    tr_loader = _loader(PatchDataset(root, tr_idx, tf_train), batch_size, workers, True)
    va_loader = _loader(PatchDataset(root, va_idx, tf_eval), batch_size, workers, False)

    n_pos = sum(s.label for s in tr_idx)
    pos_weight = float(min(max((len(tr_idx) - n_pos) / max(n_pos, 1), 1.0), 20.0))
    pos_weight = None if cfg["train"].get("pos_weight") == "none" else pos_weight

    mc = cfg["model"]
    model = TamperingModel(arch=mc.get("architecture", "resnet18"),
                           pretrained=bool(mc.get("pretrained", True)),
                           dropout=float(mc.get("dropout", 0.3)),
                           freeze_backbone=bool(run.get("freeze_backbone", True)))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(
        lr=float(run["lr"]), backbone_lr=run.get("backbone_lr"),
        weight_decay=float(run.get("weight_decay", 1e-4))))

    if args.smoke:
        batch = next(iter(tr_loader))
        x, y = batch["image"].to(device), batch["label"].float().to(device)
        t0 = time.time()
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=(torch.tensor([pos_weight], device=device)
                                   if pos_weight else None))
        loss.backward()
        optimizer.step()
        print(f"[smoke] batch={tuple(x.shape)} logits={tuple(logits.shape)} "
              f"loss={loss.item():.4f} in {time.time() - t0:.1f}s", flush=True)
        if not torch.isfinite(loss):
            raise SystemExit("[smoke] non-finite loss")
        # one tiled document through the same code path
        with Image.open(root / va_rows[0].image_path) as im:
            pil = im.convert("RGB")
        t0 = time.time()
        sc = score_document(model, pil, tf_eval, device, patch=patch, stride=stride)
        print(f"[smoke] document {pil.size} tiles={len(sc['boxes'])} "
              f"prob={sc['probability']:.4f} heatmap={sc['heatmap'].shape} "
              f"in {time.time() - t0:.2f}s", flush=True)
        return {"smoke": True, "batch": list(x.shape), "loss": loss.item()}

    epochs = int(run["epochs"])
    patience = int(run.get("early_stop_patience", 3))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0) or 0.0)
    accum = int(run.get("accum_steps", 1))
    history: List[Dict] = []
    best = {"auc": -math.inf, "epoch": -1}
    since_best = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, tr_loader, optimizer, device, accum,
                                     pos_weight, grad_clip)
        v_logits, v_labels, v_loss = collect_logits(model, va_loader, device)
        try:
            v_auc = float(roc_auc_score(v_labels, sigmoid(v_logits)))
        except ValueError:
            v_auc = float("nan")
        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_loss": round(v_loss, 4),
               "val_patch_auc": None if math.isnan(v_auc) else round(v_auc, 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{run_name}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
              f"val_patch_auc={row['val_patch_auc']} ({row['seconds']}s)", flush=True)

        if not math.isnan(v_auc) and v_auc > best["auc"]:
            best = {"auc": v_auc, "epoch": epoch}
            save_checkpoint(ckpt_path, model, run_name=run_name, run_cfg=run,
                            task="patch", patch={"patch_size": patch, "stride": stride},
                            epoch=epoch, seed=seed,
                            image_size=int(cfg["data"]["image_size"]),
                            normalize_mean=cfg["data"]["normalize_mean"],
                            normalize_std=cfg["data"]["normalize_std"],
                            split_digest=split_digest(splits),
                            split_meta={"source": "existing",
                                        "file": cfg["data"]["splits_file"],
                                        "policy": "group-disjoint by base document"},
                            pos_weight=pos_weight,
                            model_version=f"sidtd-patch-resnet18-e{epoch}",
                            torch_version=torch.__version__,
                            created=datetime.now(timezone.utc).isoformat(),
                            val_patch_metrics={"val_loss": row["val_loss"],
                                               "val_patch_auc": row["val_patch_auc"]})
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                print(f"[{run_name}] early stop at epoch {epoch}", flush=True)
                break

    train_seconds = time.time() - t_start
    if best["epoch"] < 0:
        raise SystemExit("[patch] no checkpoint improved — check labels/metric")

    # ---- document-level calibration on validation -------------------------
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    d_logits, d_labels, d_times, _ = _doc_scores(model, va_rows, root, tf_eval,
                                                 device, patch, stride)
    temperature = (fit_temperature(d_logits, d_labels)
                   if cfg["calibration"].get("temperature_scaling", True) else 1.0)
    d_prob = sigmoid(d_logits / temperature)
    target_fpr = float(cfg["thresholds"].get("target_fpr", 0.02))
    high, rule = select_threshold(d_labels, d_prob, target_fpr)
    low = min(low_threshold(d_labels, d_prob), high)
    try:
        d_auc = float(roc_auc_score(d_labels, d_prob))
    except ValueError:
        d_auc = float("nan")

    ckpt.update({
        "temperature": float(temperature),
        "threshold": {"tamper_high": float(high), "tamper_low": float(low),
                      "target_fpr": target_fpr, "selection_rule": rule,
                      "low_rule": "5th percentile of calibrated validation tampered "
                                  "document scores",
                      "fitted_on": "validation (document-level max tile logit)"},
        "thresholds_decision": {
            "tamper_high": float(high), "tamper_low": float(low),
            "rescan_below": float(cfg["quality"]["rescan_below"]),
            "ocr_min": float(cfg["decision"].get("ocr_min", 0.55)),
        },
        "val_document_metrics": {
            "n": int(d_labels.size), "n_tampered": int(d_labels.sum()),
            "val_auc": None if math.isnan(d_auc) else round(d_auc, 4),
            "threshold_rule": rule,
            "mean_document_latency_ms": round(float(np.mean(d_times)) * 1e3, 1),
            "tiles_per_document": len(
                tile_boxes(*Image.open(root / va_rows[0].image_path).size, patch, stride)),
        },
        "train_seconds": round(train_seconds, 1),
        "history": history,
    })
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(ckpt_path)

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "run_name": run_name, "task": "patch", "seed": seed, "device": device,
        "patch_size": patch, "stride": stride,
        "batch_size": batch_size, "epochs_run": history[-1]["epoch"],
        "best_epoch": best["epoch"], "best_patch_auc": round(best["auc"], 4),
        "train_seconds": round(train_seconds, 1),
        "n_train_patches": len(tr_idx), "n_val_patches": len(va_idx),
        "skipped": {"train": skip_tr, "val": skip_va},
        "temperature": temperature, "tamper_high": high, "tamper_low": low,
        "val_document_auc": ckpt["val_document_metrics"]["val_auc"],
        "mean_document_latency_ms": ckpt["val_document_metrics"]["mean_document_latency_ms"],
        "split_digest": split_digest(splits), "pos_weight": pos_weight,
        "run_cfg": run, "model_cfg": mc,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "finished": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"[{run_name}] DONE best_epoch={best['epoch']} val_doc_auc={d_auc:.4f} "
          f"high={high:.3f} low={low:.3f} in {train_seconds:.0f}s -> {ckpt_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
