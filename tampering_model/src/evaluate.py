#!/usr/bin/env python3
"""Reproducible baseline-vs-improved evaluation on the held-out test set.

    .venv/bin/python src/evaluate.py \
        --models baseline=checkpoints/baseline/best.pt \
                  improved=checkpoints/improved/best.pt

Writes to experiments/tampering/results/:
  metrics_<name>.json   — all metrics, timings, group analysis
  confusion_<name>.csv/.png, roc_curves.png
  localization_<name>.json (only where annotation-derived masks exist)
  samples/              — Grad-CAM overlays + ELA on deterministic test examples
  report.md             — baseline-vs-improved report

Thresholds come from each checkpoint (fitted on validation only). The test
split is never used to tune anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idguard.dataset import (TamperDataset, build_transforms,  # noqa: E402
                             load_manifest, load_splits, samples_for_split)
from idguard.decision_engine import (MANUAL_REVIEW, NO_TAMPERING_DETECTED,  # noqa: E402
                                     POTENTIAL_TAMPERING, RESCAN_REQUIRED, decide)
from idguard.forensic import derived_region_for_fake, ela_map  # noqa: E402
from idguard.heatmap import GradCAM, overlay_cam  # noqa: E402
from idguard.model import model_from_checkpoint  # noqa: E402
from idguard.patch import score_document, tile_boxes  # noqa: E402
from idguard.train import resolve_device, sigmoid  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "n": int(y.size), "n_genuine": n_neg, "n_tampered": n_pos,
        "threshold": round(float(thr), 4),
        "accuracy": round(float((yhat == y).mean()), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "fpr": round(float(fp / n_neg), 4) if n_neg else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": (round(float(roc_auc_score(y, p)), 4) if 0 < n_pos < y.size else None),
    }


def group_analysis(y: np.ndarray, p: np.ndarray, q: np.ndarray,
                   thresholds: Dict, degraded_thr: float) -> Dict:
    """Four test groups x gate on/off -> decision-rate table.

    A: genuine good, B: genuine degraded, C: tampered good, D: tampered degraded.
    """
    out: Dict[str, Dict] = {}
    specs = [("A_genuine_good", 0, False), ("B_genuine_degraded", 0, True),
             ("C_tampered_good", 1, False), ("D_tampered_degraded", 1, True)]
    for gate in (True, False):
        key = "gate_on" if gate else "gate_off"
        for gname, label, degraded in specs:
            mask = (y == label) & ((q < degraded_thr) == degraded)
            n = int(mask.sum())
            if n == 0:
                out.setdefault(gname, {})[key] = {"n": 0}
                continue
            dec_counts = defaultdict(int)
            for pi, qi in zip(p[mask], q[mask]):
                d = decide(float(pi), float(qi), None, thresholds, gate_enabled=gate)
                dec_counts[d.decision] += 1
            row = {
                "n": n,
                "fraud_flag_rate": round(dec_counts[POTENTIAL_TAMPERING] / n, 4),
                "rescan_rate": round(dec_counts[RESCAN_REQUIRED] / n, 4),
                "manual_review_rate": round(dec_counts[MANUAL_REVIEW] / n, 4),
                "inconclusive_rate": round((dec_counts[RESCAN_REQUIRED] + dec_counts[MANUAL_REVIEW]) / n, 4),
                "cleared_rate": round(dec_counts[NO_TAMPERING_DETECTED] / n, 4),
                "not_cleared_rate": round(1 - dec_counts[NO_TAMPERING_DETECTED] / n, 4),
            }
            out.setdefault(gname, {})[key] = row
    return out


def timing_bench(model, tf, samples, root: Path, device: str, n_max: int = 50) -> Dict:
    """Per-image (batch=1) latency on up to n_max test images."""
    idx = list(range(0, len(samples), max(1, len(samples) // n_max)))[:n_max]
    times = []
    with torch.inference_mode():
        for i in idx:
            with Image.open(root / samples[i].image_path) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            model(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: KB -> MB
    return {"n_timed": len(times), "device": device,
            "mean_ms": round(float(arr.mean()), 1), "p50_ms": round(float(np.percentile(arr, 50)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "peak_rss_mb": round(peak_mb, 0)}


def predict_probs(model, tf, samples, root: Path, device: str, bs: int = 8) -> np.ndarray:
    ds = TamperDataset(root, samples, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    out = []
    with torch.inference_mode():
        for batch in dl:
            lg = model(batch["image"].to(device)).reshape(-1)
            out.append(lg.cpu().numpy())
    return np.concatenate(out)


def predict_documents_patch(model, tf, rows, root: Path, device: str, patch: int,
                            stride: int, temperature: float):
    """Document scores for a patch (tiled) checkpoint: max tile logit."""
    logits, latency = [], []
    with torch.inference_mode():
        for s in rows:
            with Image.open(root / s.image_path) as im:
                pil = im.convert("RGB")
            t0 = time.perf_counter()
            r = score_document(model, pil, tf, device, patch=patch, stride=stride,
                               temperature=temperature, heatmap=False)
            if device == "mps":
                torch.mps.synchronize()
            latency.append((time.perf_counter() - t0) * 1e3)
            logits.append(r["logit"])
    probs = sigmoid(np.asarray(logits) / temperature)
    return probs, np.asarray(latency)


def localization_patch(model, tf, root: Path, device: str, rows, ann_dir: Path,
                       patch: int, stride: int, temperature: float,
                       thr: float = 0.5, down: int = 8) -> Dict:
    """IoU/Dice of the thresholded tile map vs annotation-derived region rects."""
    ious, dices, missing = [], [], 0
    with torch.inference_mode():
        for s in rows:
            if s.label != 1:
                continue
            region = derived_region_for_fake(
                {"src_image": s.src_image, "tamper_field": s.tamper_field}, str(ann_dir))
            if region is None:
                missing += 1
                continue
            with Image.open(root / s.image_path) as im:
                pil = im.convert("RGB")
            r = score_document(model, pil, tf, device, patch=patch, stride=stride,
                               temperature=temperature, heatmap=True)
            heat = r["heatmap"][::down, ::down]
            h, w = heat.shape
            sx, sy = w / pil.size[0], h / pil.size[1]
            rx, ry, rw, rh = region
            mask = np.zeros((h, w), dtype=bool)
            x0, y0 = int(rx * sx), int(ry * sy)
            x1 = min(int((rx + rw) * sx) + 1, w)
            y1 = min(int((ry + rh) * sy) + 1, h)
            mask[max(y0, 0):y1, max(x0, 0):x1] = True
            pred = heat >= thr
            inter = float((pred & mask).sum())
            union = float((pred | mask).sum())
            ious.append(inter / union if union else 0.0)
            denom = float(pred.sum() + mask.sum())
            dices.append(2 * inter / denom if denom else 0.0)
    return {"n_tampered_test": int(sum(1 for s in rows if s.label == 1)),
            "n_with_derived_mask": len(ious), "n_without_annotation": missing,
            "tile_threshold": thr, "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "dice_mean": round(float(np.mean(dices)), 4) if dices else None,
            "note": "tile-resolution map (stride %d px, probability threshold %.2f) vs masks "
                    "DERIVED from official field annotations; SIDTD has no official "
                    "pixel-level masks" % (stride, thr)}


def roc_plot(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Test-set ROC — genuine vs tampered"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def confusion_plot(m: Dict, path: Path, title: str) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["genuine", "tampered"]); plt.yticks([0, 1], ["genuine", "tampered"])
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ---------------------------------------------------------------------------
def evaluate_model(name: str, ckpt_path: Path, cfg: Dict, samples, splits,
                   test_rows, qmap: Dict[str, float], degraded_thr: float,
                   ann_dir: Path, out_dir: Path) -> Dict:
    device = resolve_device(cfg["train"].get("device", "auto"))
    t0 = time.perf_counter()
    model, ckpt = model_from_checkpoint(ckpt_path, device=device)
    load_s = time.perf_counter() - t0

    # Each checkpoint declares the preprocessing it was trained with.
    data_cfg = dict(cfg["data"])
    if ckpt.get("image_size"):
        data_cfg["image_size"] = int(ckpt["image_size"])
    for key in ("normalize_mean", "normalize_std"):
        if ckpt.get(key):
            data_cfg[key] = ckpt[key]
    tf = build_transforms(data_cfg, augment=False)
    root = Path(cfg["data"]["root"])
    temperature = float(ckpt.get("temperature", 1.0))
    pcfg = ckpt.get("patch")
    if pcfg:
        patch_sz, stride = int(pcfg["patch_size"]), int(pcfg["stride"])
        probs, lat = predict_documents_patch(model, tf, test_rows, root, device,
                                             patch_sz, stride, temperature)
        with Image.open(root / test_rows[0].image_path) as im:
            n_tiles = len(tile_boxes(*im.size, patch_sz, stride))
        timing = {"n_timed": len(lat), "device": device,
                  "mean_ms": round(float(lat.mean()), 1),
                  "p50_ms": round(float(np.percentile(lat, 50)), 1),
                  "p95_ms": round(float(np.percentile(lat, 95)), 1),
                  "tiles_per_image": n_tiles,
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)}
    else:
        logits = predict_probs(model, tf, test_rows, root, device)
        probs = sigmoid(logits / temperature)
        timing = None
    thr = ckpt.get("threshold", {})
    high = float(thr.get("tamper_high", cfg["thresholds"]["tamper_high"]))

    y = np.array([s.label for s in test_rows], dtype=int)
    q = np.array([qmap.get(s.image_path, 0.5) for s in test_rows], dtype=float)

    result: Dict = {
        "model": name, "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "model_version": ckpt.get("model_version"), "device": device,
        "temperature": round(temperature, 3),
        "threshold": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in thr.items()},
        "classification": classification_metrics(y, probs, high),
        "timing": timing if pcfg else timing_bench(model, tf, test_rows, root, device),
        "model_load_s": round(load_s, 2),
        "task": "patch" if pcfg else "whole_image",
    }

    # per-manipulation-type recall (improved diagnostics; test fakes only)
    by_type = {}
    for i, s in enumerate(test_rows):
        if s.label == 1:
            by_type.setdefault(s.manipulation_type or "unknown", []).append(probs[i] >= high)
    result["per_manipulation_type"] = {
        k: {"n": len(v), "detected": int(np.sum(v)),
            "recall": round(float(np.mean(v)), 4)}
        for k, v in sorted(by_type.items())}

    # group x gate analysis
    result["groups"] = group_analysis(y, probs, q,
                                      {"tamper_high": high,
                                       "tamper_low": float(thr.get("tamper_low", high * 0.4)),
                                       "rescan_below": float(cfg["quality"]["rescan_below"]),
                                       "quality_degraded": degraded_thr,
                                       "ocr_min": float(cfg["decision"].get("ocr_min", 0.55))},
                                      degraded_thr)

    # localization against annotation-derived region masks
    hc = cfg.get("heatmap", {})
    cam_thr_loc = 0.5
    ious, dices = [], []
    n_missing_ann = 0
    if pcfg:
        result["localization"] = localization_patch(
            model, tf, root, device, test_rows, ann_dir,
            int(pcfg["patch_size"]), int(pcfg["stride"]), temperature)
        result["scores"] = probs.tolist()
        return {"result": result, "probs": probs, "y": y, "high": high,
                "model": model, "tf": tf, "device": device, "patch": pcfg,
                "temperature": temperature}

    cam = GradCAM(model, target_layer=hc.get("target_layer", "layer4"))
    for s in test_rows:
        if s.label != 1:
            continue
        region = derived_region_for_fake(
            {"src_image": s.src_image, "tamper_field": s.tamper_field}, str(ann_dir))
        if region is None:
            n_missing_ann += 1
            continue
        with Image.open(root / s.image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            x = tf(im).unsqueeze(0).to(device)
        with torch.enable_grad():
            c = cam(x)[0].cpu().numpy()
        # scale derived rect (source pixels) to CAM grid
        sx, sy = c.shape[1] / w, c.shape[0] / h
        rx, ry, rw, rh = region
        mask = np.zeros_like(c, dtype=bool)
        x0, y0 = int(rx * sx), int(ry * sy)
        x1, y1 = min(int((rx + rw) * sx), c.shape[1]), min(int((ry + rh) * sy), c.shape[0])
        mask[max(y0, 0):y1, max(x0, 0):x1] = True
        pred_mask = c >= cam_thr_loc
        inter = float((pred_mask & mask).sum())
        union = float((pred_mask | mask).sum())
        iou = inter / union if union else 0.0
        dice = 2 * inter / (pred_mask.sum() + mask.sum()) if (pred_mask.sum() + mask.sum()) else 0.0
        ious.append(iou); dices.append(dice)
    cam.close()
    result["localization"] = (
        {"n_tampered_test": int(sum(1 for s in test_rows if s.label == 1)),
         "n_with_derived_mask": len(ious), "n_without_annotation": n_missing_ann,
         "cam_threshold": cam_thr_loc,
         "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
         "dice_mean": round(float(np.mean(dices)), 4) if dices else None,
         "note": "masks are DERIVED from official field annotations (src + tampered field); "
                 "SIDTD provides no official pixel-level tampering masks"}
        if ious else
        {"n_with_derived_mask": 0,
         "note": "no derived masks available on the test split; localization metrics not measurable"})
    return {"result": result, "probs": probs, "y": y, "high": high,
            "model": model, "tf": tf, "device": device, "temperature": temperature}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=checkpoint pairs, e.g. baseline=checkpoints/baseline/best.pt")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "tampering" / "results"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    root = Path(cfg["data"]["root"])
    samples = load_manifest(Path(cfg["data"]["manifest"]))
    splits = load_splits(Path(cfg["data"]["splits_file"]))
    split_meta = json.loads(Path(cfg["data"]["splits_file"]).read_text()).get("_meta", {})
    splits["_meta"] = split_meta
    test_rows = samples_for_split(samples, splits, "test")

    qmap, degraded_thr = {}, float(cfg["quality"]["degraded_threshold"])
    qcsv = root / "quality.csv"
    if qcsv.exists():
        with open(qcsv) as f:
            for r in csv.DictReader(f):
                qmap[r["image_path"]] = float(r["quality_score"])
    qmeta = root / "quality_meta.json"
    if qmeta.exists():
        degraded_thr = float(json.loads(qmeta.read_text())["degraded_threshold"])
    print(f"test set: {len(test_rows)} images; degraded threshold {degraded_thr}")

    ann_dir = root / "templates" / "Annotations" / "reals"  # VIA field rectangles
    all_results, curves = {}, {}
    trained = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        res = evaluate_model(name, ROOT / path, cfg, samples, splits, test_rows,
                             qmap, degraded_thr, ann_dir, out_dir)
        r = res.pop("result")
        trained[name] = res
        all_results[name] = r
        curves[name] = (res["y"], res["probs"])
        (out_dir / f"metrics_{name}.json").write_text(json.dumps(r, indent=1))
        m = r["classification"]
        with (out_dir / f"confusion_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["actual\\predicted", "genuine", "tampered"])
            w.writerow(["genuine", m["tn"], m["fp"]])
            w.writerow(["tampered", m["fn"], m["tp"]])
        confusion_plot(m, out_dir / f"confusion_{name}.png", f"{name} (thr={m['threshold']})")
        (out_dir / f"localization_{name}.json").write_text(json.dumps(r["localization"], indent=1))
        print(f"[{name}] acc={m['accuracy']} f1={m['f1']} fpr={m['fpr']} "
              f"auc={m['roc_auc']} infer={r['timing']['mean_ms']}ms")

    roc_plot(curves, out_dir / "roc_curves.png")

    # deterministic sample heatmaps from the last model listed (usually improved)
    last = args.models[-1].partition("=")[0]
    tinfo = trained[last]
    model, tf, device = tinfo["model"], tinfo["tf"], tinfo["device"]
    picks = ([s for s in sorted(test_rows, key=lambda s: s.image_path) if s.label == 0][:2]
             + [s for s in sorted(test_rows, key=lambda s: s.image_path) if s.label == 1][:2])
    pcfg = tinfo.get("patch")
    cam = (None if pcfg else
           GradCAM(model, target_layer=cfg.get("heatmap", {}).get("target_layer", "layer4")))
    for s in picks:
        with Image.open(root / s.image_path) as im:
            im = im.convert("RGB")
            if pcfg:
                r = score_document(model, im, tf, device,
                                   patch=int(pcfg["patch_size"]), stride=int(pcfg["stride"]),
                                   temperature=float(tinfo.get("temperature", 1.0)),
                                   heatmap=True)
                overlay = overlay_cam(im, r["heatmap"])
            else:
                x = tf(im).unsqueeze(0).to(device)
                with torch.enable_grad():
                    c = cam(x)[0].cpu().numpy()
                overlay = overlay_cam(im, c)
            tag = "genuine" if s.label == 0 else "tampered"
            suffix = "patchmap" if pcfg else "cam"
            overlay.save(out_dir / "samples" / f"{last}_{tag}_{Path(s.image_path).stem}_{suffix}.png")
            ela = (ela_map(im) * 255).astype(np.uint8)
            Image.fromarray(ela, "L").save(out_dir / "samples" / f"ela_{Path(s.image_path).stem}.png")
    if cam is not None:
        cam.close()
    print(f"sample heatmaps -> {out_dir / 'samples'}")

    write_report(cfg, all_results, out_dir, degraded_thr, test_rows, splits)
    # free memory for subsequent runs
    return 0


# ---------------------------------------------------------------------------
def write_report(cfg: Dict, results: Dict, out_dir: Path, degraded_thr: float,
                 test_rows, splits) -> None:
    import platform

    names = list(results)
    L: List[str] = []
    L.append("# Baseline vs Improved — SIDTD tampering detection\n")
    L.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"- Hardware: {platform.platform()} / torch {torch.__version__}")
    L.append(f"- Device used: {results[names[0]]['device']}")
    L.append(f"- Test set: {len(test_rows)} images (group-disjoint, "
             f"digest {splits.get('_meta', {}).get('digest', 'n/a')})")
    L.append(f"- Degraded-quality threshold: {degraded_thr} "
             "(33rd pct of train-split genuine quality scores)")
    L.append("- Thresholds: fitted on validation only; test set untouched until this evaluation\n")

    L.append("## Headline metrics (same held-out test set)\n")
    L.append("| metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall (tamper TPR)"),
            ("f1", "F1"), ("fpr", "FPR on genuine"), ("roc_auc", "ROC-AUC")]
    for k, label in keys:
        vals = []
        for n in names:
            v = results[n]["classification"].get(k)
            vals.append("—" if v is None else f"{v:.4f}")
        L.append(f"| {label} | " + " | ".join(vals) + " |")
    for n in names:
        c = results[n]["classification"]
        L.append(f"- **{n}**: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']} "
                 f"(threshold {c['threshold']}, rule: {results[n].get('threshold', {}).get('selection_rule', 'n/a')})")
    L.append("")

    L.append("## Timing (batch=1, per document, MPS)\n")
    L.append("| model | task | tiles/doc | load (s) | mean (ms) | p50 (ms) | p95 (ms) | peak RSS (MB) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        t = results[n]["timing"]
        L.append(f"| {n} | {results[n].get('task', 'whole_image')} | "
                 f"{t.get('tiles_per_image', 1)} | {results[n]['model_load_s']} | "
                 f"{t['mean_ms']} | {t['p50_ms']} | {t['p95_ms']} | {t['peak_rss_mb']} |")
    L.append("")
    L.append("Patch models tile each document at native resolution; their latency includes "
             "every tile forward pass plus tile assembly, which is why it is far above the "
             "single-resize whole-image models.\n")

    L.append("## Per-manipulation-type recall (test fakes)\n")
    types = sorted({t for n in names for t in results[n]["per_manipulation_type"]})
    L.append("| type | " + " | ".join(f"{n} (n, recall)" for n in names) + " |")
    L.append("|---|" + "---|" * len(names))
    for t in types:
        row = []
        for n in names:
            d = results[n]["per_manipulation_type"].get(t)
            row.append("—" if not d else f"{d['n']}, {d['recall']:.3f}")
        L.append(f"| {t} | " + " | ".join(row) + " |")
    L.append("")

    L.append("## Confidence-gate analysis (four test groups)\n")
    L.append("Groups: A genuine/good, B genuine/degraded, C tampered/good, D tampered/degraded.\n"
             "Fraud flag = POTENTIAL_TAMPERING; inconclusive = RESCAN_REQUIRED + MANUAL_REVIEW.\n")
    for n in names:
        L.append(f"### {n}\n")
        L.append("| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | "
                 "manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        g = results[n]["groups"]
        for gname in ("A_genuine_good", "B_genuine_degraded", "C_tampered_good", "D_tampered_degraded"):
            on, off = g.get(gname, {}).get("gate_on", {"n": 0}), g.get(gname, {}).get("gate_off", {"n": 0})
            if on.get("n", 0) == 0:
                L.append(f"| {gname} | 0 | — | — | — | — | — | — | — |")
                continue
            L.append(f"| {gname} | {on['n']} | {on['fraud_flag_rate']} | {off['fraud_flag_rate']} | "
                     f"{on['rescan_rate']} | {on['manual_review_rate']} | {on['inconclusive_rate']} | "
                     f"{on['not_cleared_rate']} | {off['not_cleared_rate']} |")
        L.append("")

    L.append("## Localization\n")
    for n in names:
        loc = results[n]["localization"]
        L.append(f"- **{n}**: {json.dumps(loc)}")
    L.append("")

    L.append("## Limitations & unmeasured claims\n")
    L.append("- SIDTD templates are *digital synthetic* document images; real-world photographed/"
             "scanned captures (the degraded-deployment regime) live in the corpus's video parts, "
             "which were deliberately not downloaded (79 GB). Group B therefore covers the "
             "naturally lower-quality tail of the template set, not field-captured photos.")
    L.append("- No official pixel-level masks: IoU/Dice use masks *derived* from the official "
             "field annotations and are not official localization ground truth.")
    L.append("- No real government-issued documents were used; SIDTD documents are synthetic.")
    L.append("- OCR reliability is not part of this dataset; gate experiments use synthetic OCR "
             "signals in tests only, and `ocr_reliability=None` in all reported numbers.")
    L.append("- Single train/val/test split (group-disjoint); no cross-validation was run within "
             "this hardware budget.")
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L))
    print(f"report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    sys.exit(main())
