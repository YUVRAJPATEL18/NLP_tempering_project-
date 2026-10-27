# Tampering-detection experiments — reproducibility record

Everything below was produced on this machine with the configuration and seeds in `config.yaml`.
Re-running the same commands reproduces the same numbers (verified: the interrupted 512 px run
repeated epoch 1 bit-for-bit, val AUC 0.642 in both attempts).

## Environment

| Item | Value |
|---|---|
| Machine | MacBook Air, Apple M1, 8 GB RAM, macOS 15.6.1 |
| Python | 3.9.6 in `.venv` |
| PyTorch / torchvision | 2.5.1 / 0.20.1 (MPS backend) |
| Device | `mps`, FP32 (`amp: false`) |
| Deliberately not downloaded | SIDTD `videos.zip` + `clips.zip` (78 GB) |

## Dataset

| Item | Value |
|---|---|
| SIDTD v1 (figshare `27136242`, CC BY 4.0), DOI 10.6084/m9.figshare.27136242.v1 | 1.27 GB |
| Images | 2,222 (1,000 genuine / 1,222 tampered), all integrity-validated |
| Splits | group-disjoint 70/15/15, seed 42, digest `697c850e9401825e` |
| Test split | 333 images (149 genuine / 184 tampered), used once per model |
| Derived masks | 181/184 tampered test images (from official field annotations; **not** official pixel GT) |

## Models (all ResNet-18 + binary head, no architecture changes between runs)

| checkpoint | recipe | val AUC | test AUC | test F1 | test FPR | train time |
|---|---|---|---|---|---|---|
| `checkpoints/baseline/best.pt` | frozen backbone, linear head, 256 px, 6 ep | 0.5688 | 0.5813 | 0.032 | 0.007 | 292 s |
| `checkpoints/improved/best.pt` | full fine-tune, 256 px, aug, differential LR, 10 ep | 0.6966 | 0.6948 | 0.232 | 0.047 | 791 s |
| `checkpoints/improved_512/best.pt` | frozen backbone @512 px, aug, 6 ep | 0.7602 | 0.7750 | 0.185 | 0.020 | 425 s |
| `checkpoints/improved_512_raw/best.pt` | frozen @512 px, **no aug**, 6 ep | 0.7799 | 0.7800 | 0.211 | 0.020 | 430 s |
| `checkpoints/improved_512_long/best.pt` | frozen @512 px, no aug, 14 ep, lr 5e-4 → **production** | **0.8522** | **0.8437** | **0.455** | 0.040 | 1180 s |
| `checkpoints/improved_patch/best.pt` | native-resolution tiles (rejected — text-density confound) | 0.5529† | 0.5511 | 0.042 | 0.020 | 774 s |

† document-level val AUC for the patch run; its within-image patch AUC (0.94) was a confound.

## Commands

```bash
.venv/bin/python scripts/download_sidtd.py                 # acquisition (MD5-verified, resumable)
.venv/bin/python scripts/prepare_dataset.py                # validation + splits + quality.csv
PYTHONPATH=src .venv/bin/python -m idguard.train --run baseline
PYTHONPATH=src .venv/bin/python -m idguard.train --run improved
PYTHONPATH=src .venv/bin/python -m idguard.train --run improved_512
PYTHONPATH=src .venv/bin/python -m idguard.train --run improved_512_raw
PYTHONPATH=src .venv/bin/python -m idguard.train --run improved_512_long
PYTHONPATH=src .venv/bin/python experiments/tampering/resolution_probe.py   # evidence for 512 px
PYTHONPATH=src .venv/bin/python src/evaluate.py --models \
    baseline=checkpoints/baseline/best.pt improved=checkpoints/improved/best.pt \
    improved_512=checkpoints/improved_512/best.pt improved_512_raw=checkpoints/improved_512_raw/best.pt \
    improved_patch=checkpoints/improved_patch/best.pt improved_512_long=checkpoints/improved_512_long/best.pt
```

`--smoke` on any run does a one-batch pass instead of a full epoch.

## Artifacts in this directory

```
results/report.md                  baseline-vs-improved report (tables + gate analysis)
results/metrics_<run>.json         full per-model metrics
results/confusion_<run>.csv/.png   confusion matrices
results/roc_curves.png             all models on one ROC plot
results/localization_<run>.json    IoU/Dice vs derived masks
results/sample_predictions.json    sample inference output (lo\med\hi score per class, gate on/off)
results/samples/                   sample Grad-CAM overlays + ELA maps (test split)
results/*_probe.json               probe results (patch, field-matched patch, resolution)
results/<run>/history.json         per-epoch training curve
results/<run>/run_meta.json        resolved training configuration
```

## Threshold and calibration provenance

* Temperature scaling is fitted on the validation split and stored in the checkpoint
  (`temperature`), as are `tamper_high` / `tamper_low` (`thresholds_decision`).
* `tamper_high` = largest recall with validation FPR ≤ 0.02 (Youden fallback);
  `tamper_low` = 5th percentile of calibrated validation tampered scores.
* `quality.degraded_threshold` = 33rd percentile of train-split genuine quality scores (0.8251),
  computed once by `scripts/prepare_dataset.py`.
* The test split was touched exactly once per model, after all selection decisions.

## Hardware/runtime notes

* Latency (batch 1, warm, MPS): 14 ms (256 px) to 31 ms (512 px) per document; 425 ms for the
  rejected tiled model. Measured in `evaluate.py` with `torch.mps.synchronize()`; run-to-run
  variance across two full evaluations was ±8 ms at 256 px, so treat single-digit differences as
  noise.
* Peak RSS reported by the evaluation script is process-wide (`ru_maxrss`), not per-model — it
  reflects the whole evaluation process (~460 MB) and should not be read as a model footprint.

## Known limitations

* SIDTD documents are **synthetic** (AI-generated templates); no real government-issued document
  was used. The degraded-quality group is the lower tail of the template set, not photographed
  documents (those are in the 54 GB video corpus that was not downloaded).
* No official pixel-level masks → localization numbers are measured against **derived** field
  rectangles and are indicative only. Best IoU is 0.008, i.e. Grad-CAM is not a localization tool
  at this scale.
* OCR reliability is not part of SIDTD; all reported numbers use `ocr_reliability=None`. Gate
  behaviour with real OCR input is covered by unit tests only.
* One group-disjoint split, no cross-validation (hardware budget).
* Detection strength is modest (test TPR 30% at 4% FPR, AUC 0.844). Reported as-is.
