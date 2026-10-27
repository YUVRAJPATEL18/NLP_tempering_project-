"""Reproducible training entry point.

    .venv/bin/python -m idguard.train --run baseline --smoke   # one-batch smoke test
    .venv/bin/python -m idguard.train --run baseline
    .venv/bin/python -m idguard.train --run improved

Selects the best checkpoint on the validation ROC-AUC (early stopping), fits
temperature scaling and the decision thresholds on the *validation* split only,
and records everything needed for reproduction in the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from .dataset import (TamperDataset, build_group_splits, load_manifest, load_splits,
                      make_loaders, samples_for_split, save_splits, split_digest,
                      validate_images)
from .model import TamperingModel, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(spec: str = "auto") -> str:
    if spec == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return spec


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def ensure_splits(cfg: Dict, samples) -> Tuple[Dict[str, List[str]], Dict]:
    path = Path(cfg["data"]["splits_file"])
    if path.exists():
        splits = load_splits(path)
        meta = {"source": "existing", "file": str(path)}
        return splits, meta
    splits = build_group_splits(samples, ratios=tuple(cfg["data"]["split_ratios"]),
                                seed=int(cfg["data"]["split_seed"]))
    meta = {
        "source": "built",
        "ratios": list(cfg["data"]["split_ratios"]),
        "seed": int(cfg["data"]["split_seed"]),
        "policy": "group-disjoint by base document (genuine + all its fakes in one split)",
        "digest": split_digest(splits),
        "counts": {k: len(v) for k, v in splits.items() if not k.startswith("_")},
    }
    save_splits(path, splits, meta)
    return splits, meta


def select_threshold(y_true: np.ndarray, proba: np.ndarray,
                     target_fpr: float) -> Tuple[float, str]:
    """Validation-only threshold selection.

    Rule: largest threshold that still achieves the maximum achievable TPR
    while keeping FPR <= target_fpr; falls back to Youden's J when the
    constraint is infeasible. Never sees the test split.
    """
    fpr, tpr, ths = roc_curve(y_true, proba)
    ok = np.where(fpr <= target_fpr)[0]
    if ok.size:
        max_tpr = tpr[ok].max()
        candidates = ok[tpr[ok] == max_tpr]
        best = candidates[np.argmax(ths[candidates])]
        return float(ths[best]), f"max_tpr_at_fpr<= {target_fpr:.2f}"
    best = int(np.argmax(tpr - fpr))
    return float(ths[best]), "youden_j_fallback"


def low_threshold(y_true: np.ndarray, proba: np.ndarray, miss_rate: float = 0.05) -> float:
    """Below this, at most `miss_rate` of validation-tampered samples lie
    (i.e. clearing a document here misses <= 5% of known-tampered validation cases)."""
    tampered = proba[y_true == 1]
    if tampered.size == 0:
        return 0.0
    return float(np.quantile(tampered, miss_rate))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid-search temperature scaling on validation NLL (no scipy needed)."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(np.log(0.5), np.log(5.0), 60)):
        p = sigmoid(logits / t)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


# ---------------------------------------------------------------------------
# loops
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    logits, labels, losses = [], [], []
    bce = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].float().to(device)
        lg = model(x)
        losses.append(bce(lg, y).item())
        logits.append(lg.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(logits), np.concatenate(labels), float(np.mean(losses))


def train_one_epoch(model, loader, optimizer, device, accum: int,
                    pos_weight: Optional[float], grad_clip: float,
                    max_batches: Optional[int] = None) -> float:
    model.train()
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    total, n, step = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["label"].float().to(device)
        loss = bce(model(x), y) / accum
        loss.backward()
        total += loss.item() * accum
        n += 1
        if (bi + 1) % accum == 0 or (bi + 1) == len(loader):
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# main training routine
# ---------------------------------------------------------------------------
def train_run(cfg_path: Path, run_name: str, smoke: bool = False) -> Dict:
    cfg = load_config(cfg_path)
    if run_name not in cfg:
        raise SystemExit(f"run {run_name!r} not present in {cfg_path}")
    run = cfg[run_name]
    # A run may override the input resolution (e.g. 512 px); everything below
    # (transforms, loaders, checkpoint metadata) then uses that resolution.
    if "image_size" in run:
        cfg = {**cfg, "data": {**cfg["data"],
                               "image_size": int(run["image_size"])}}
    seed = int(cfg["seed"])
    set_seed(seed)
    device = resolve_device(cfg["train"].get("device", "auto"))
    out_dir = ROOT / "experiments" / "tampering" / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ROOT / "checkpoints" / run_name / "best.pt"
    print(f"[train:{run_name}] device={device} seed={seed}", flush=True)

    # data ---------------------------------------------------------------
    root = Path(cfg["data"]["root"])
    samples = load_manifest(Path(cfg["data"]["manifest"]))
    report = validate_images(root, samples)
    if report.invalid:
        print(f"[train] WARNING: excluding {len(report.invalid)} invalid files", flush=True)
        bad = set(report.invalid)
        samples = [s for s in samples if s.image_path not in bad]
        (out_dir / "excluded_files.json").write_text(json.dumps(report.to_dict(), indent=1))
    splits, split_meta = ensure_splits(cfg, samples)
    print(f"[train] data: {len(samples)} valid images, splits "
          f"{ {k: len(v) for k, v in splits.items() if not k.startswith('_')} }", flush=True)

    batch_size = int(run["batch_size"])
    loaders = make_loaders(root, samples, splits, cfg, augment=bool(run["augment"]),
                           batch_size=batch_size)
    n_train_fake = sum(s.label for s in samples_for_split(samples, splits, "train"))
    n_train = len(splits["train"])
    pos_weight = None
    if cfg["train"].get("pos_weight", "auto") == "auto" and n_train_fake:
        pos_weight = float(min(max((n_train - n_train_fake) / n_train_fake, 1.0), 20.0))
    print(f"[train] train={n_train} (tampered {n_train_fake}), pos_weight={pos_weight}", flush=True)

    # model --------------------------------------------------------------
    mc = cfg["model"]
    model = TamperingModel(arch=mc.get("architecture", "resnet18"),
                           pretrained=bool(mc.get("pretrained", True)),
                           dropout=float(mc.get("dropout", 0.3)),
                           freeze_backbone=bool(run["freeze_backbone"]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(lr=float(run["lr"]),
                               backbone_lr=run.get("backbone_lr"),
                               weight_decay=float(run["weight_decay"])))

    if smoke:
        batch = next(iter(loaders["train"]))
        x, y = batch["image"].to(device), batch["label"].float().to(device)
        t0 = time.time()
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None)
        loss.backward()
        optimizer.step()
        print(f"[smoke] batch={tuple(x.shape)} logits={tuple(logits.shape)} "
              f"loss={loss.item():.4f} step_ok in {time.time() - t0:.1f}s", flush=True)
        if not torch.isfinite(loss):
            raise SystemExit("[smoke] non-finite loss")
        return {"smoke": True, "batch_shape": list(x.shape), "loss": loss.item()}

    # training -----------------------------------------------------------
    monitor = cfg["train"].get("monitor", "val_auc")
    patience = int(run.get("early_stop_patience", 4))
    accum = int(run.get("accum_steps", 1))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0) or 0.0)
    epochs = int(run["epochs"])

    history = []
    best = {"score": -math.inf, "epoch": -1}
    since_best = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        try:
            train_loss = train_one_epoch(model, loaders["train"], optimizer, device,
                                         accum, pos_weight, grad_clip)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 2:
                batch_size = max(2, batch_size // 2)
                print(f"[train] OOM — reducing batch_size to {batch_size}, restarting loaders",
                      flush=True)
                loaders = make_loaders(root, samples, splits, cfg,
                                       augment=bool(run["augment"]), batch_size=batch_size)
                optimizer.zero_grad(set_to_none=True)
                continue
            raise
        v_logits, v_labels, v_loss = collect_logits(model, loaders["val"], device)
        v_prob = sigmoid(v_logits)
        try:
            v_auc = float(roc_auc_score(v_labels, v_prob))
        except ValueError:
            v_auc = float("nan")
        score = v_auc if (monitor == "val_auc" and not math.isnan(v_auc)) else -v_loss
        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_loss": round(v_loss, 4),
               "val_auc": (None if math.isnan(v_auc) else round(v_auc, 4)),
               "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{run_name}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
              f"val_loss={v_loss:.4f} val_auc={row['val_auc']} ({row['seconds']}s)", flush=True)

        if score > best["score"]:
            best = {"score": score, "epoch": epoch}
            save_checkpoint(ckpt_path, model,
                            run_name=run_name, run_cfg=run, epoch=epoch,
                            seed=seed, image_size=int(cfg["data"]["image_size"]),
                            normalize_mean=cfg["data"]["normalize_mean"],
                            normalize_std=cfg["data"]["normalize_std"],
                            split_digest=split_digest(splits), split_meta=split_meta,
                            pos_weight=pos_weight,
                            model_version=f"sidtd-resnet18-{run_name}-e{epoch}",
                            torch_version=torch.__version__,
                            created=datetime.now(timezone.utc).isoformat(),
                            val_metrics={"val_loss": row["val_loss"],
                                         "val_auc": row["val_auc"]})
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                print(f"[{run_name}] early stop at epoch {epoch} "
                      f"(best epoch {best['epoch']})", flush=True)
                break

    train_seconds = time.time() - t_start
    if best["epoch"] < 0:
        raise SystemExit("[train] no checkpoint improved — check the data/monitor")

    # calibration + thresholds on validation -----------------------------
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    v_logits, v_labels, v_loss = collect_logits(model, loaders["val"], device)
    temperature = fit_temperature(v_logits, v_labels) if cfg["calibration"].get(
        "temperature_scaling", True) else 1.0
    v_prob = sigmoid(v_logits / temperature)
    target_fpr = float(cfg["thresholds"].get("target_fpr", 0.02))
    high, rule = select_threshold(v_labels, v_prob, target_fpr)
    low = low_threshold(v_labels, v_prob)
    low = min(low, high)  # keep the band consistent
    try:
        v_auc = float(roc_auc_score(v_labels, v_prob))
    except ValueError:
        v_auc = float("nan")

    ckpt.update({
        "temperature": float(temperature),
        "threshold": {"tamper_high": float(high), "tamper_low": float(low),
                      "target_fpr": target_fpr, "selection_rule": rule,
                      "low_rule": "5th percentile of calibrated validation tampered probabilities",
                      "fitted_on": "validation"},
        "thresholds_decision": {
            "tamper_high": float(high), "tamper_low": float(low),
            "rescan_below": float(cfg["quality"]["rescan_below"]),
            "ocr_min": float(cfg["decision"].get("ocr_min", 0.55)),
        },
        "val_metrics_final": {"val_auc": (None if math.isnan(v_auc) else round(v_auc, 4)),
                              "val_loss": round(v_loss, 4)},
        "val_prob_stats": {"n": int(v_labels.size),
                           "n_tampered": int(v_labels.sum()),
                           "threshold_rule": rule},
        "train_seconds": round(train_seconds, 1),
        "history": history,
    })
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(ckpt_path)

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "run_name": run_name, "seed": seed, "device": device,
        "batch_size": batch_size, "epochs_run": history[-1]["epoch"],
        "best_epoch": best["epoch"], "train_seconds": round(train_seconds, 1),
        "temperature": temperature, "tamper_high": high, "tamper_low": low,
        "split": split_meta, "pos_weight": pos_weight,
        "run_cfg": run, "model_cfg": mc,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "finished": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"[{run_name}] DONE best_epoch={best['epoch']} val_auc={ckpt['val_metrics_final']['val_auc']} "
          f"temp={temperature:.2f} high={high:.3f} low={low:.3f} "
          f"train_time={train_seconds:.0f}s -> {ckpt_path}", flush=True)
    return {"best_epoch": best["epoch"], "val_auc": ckpt["val_metrics_final"]["val_auc"],
            "checkpoint": str(ckpt_path), "train_seconds": train_seconds}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run name from config.yaml (baseline|improved)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--smoke", action="store_true", help="one-batch smoke test only")
    args = ap.parse_args(argv)
    train_run(Path(args.config), args.run, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
