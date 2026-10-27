# Baseline vs Improved — SIDTD tampering detection

- Date: 2026-10-27 02:03
- Hardware: macOS-15.6.1-arm64-arm-64bit / torch 2.5.1
- Device used: mps
- Test set: 333 images (group-disjoint, digest 697c850e9401825e)
- Degraded-quality threshold: 0.8251 (33rd pct of train-split genuine quality scores)
- Thresholds: fitted on validation only; test set untouched until this evaluation

## Headline metrics (same held-out test set)

| metric | baseline | improved | improved_512 | improved_512_raw | improved_patch | improved_512_long |
|---|---|---|---|---|---|---|
| Accuracy | 0.4535 | 0.5015 | 0.4955 | 0.5045 | 0.4505 | 0.5976 |
| Precision | 0.7500 | 0.7812 | 0.8636 | 0.8800 | 0.5714 | 0.9032 |
| Recall (tamper TPR) | 0.0163 | 0.1359 | 0.1033 | 0.1196 | 0.0217 | 0.3043 |
| F1 | 0.0319 | 0.2315 | 0.1845 | 0.2105 | 0.0419 | 0.4553 |
| FPR on genuine | 0.0067 | 0.0470 | 0.0201 | 0.0201 | 0.0201 | 0.0403 |
| ROC-AUC | 0.5813 | 0.6948 | 0.7750 | 0.7800 | 0.5511 | 0.8437 |
- **baseline**: TN=148 FP=1 FN=181 TP=3 (threshold 0.6626, rule: max_tpr_at_fpr<= 0.02)
- **improved**: TN=142 FP=7 FN=159 TP=25 (threshold 0.7511, rule: max_tpr_at_fpr<= 0.02)
- **improved_512**: TN=146 FP=3 FN=165 TP=19 (threshold 0.8069, rule: max_tpr_at_fpr<= 0.02)
- **improved_512_raw**: TN=146 FP=3 FN=162 TP=22 (threshold 0.7368, rule: max_tpr_at_fpr<= 0.02)
- **improved_patch**: TN=146 FP=3 FN=180 TP=4 (threshold 0.6606, rule: max_tpr_at_fpr<= 0.02)
- **improved_512_long**: TN=143 FP=6 FN=128 TP=56 (threshold 0.653, rule: max_tpr_at_fpr<= 0.02)

## Timing (batch=1, per document, MPS)

| model | task | tiles/doc | load (s) | mean (ms) | p50 (ms) | p95 (ms) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| baseline | whole_image | 1 | 0.25 | 13.7 | 11.8 | 15.6 | 441600.0 |
| improved | whole_image | 1 | 0.83 | 17.0 | 16.2 | 22.9 | 441600.0 |
| improved_512 | whole_image | 1 | 0.31 | 27.2 | 24.8 | 31.0 | 462368.0 |
| improved_512_raw | whole_image | 1 | 0.39 | 25.1 | 23.9 | 33.3 | 462368.0 |
| improved_patch | patch | 77 | 0.4 | 425.0 | 389.0 | 729.1 | 462368.0 |
| improved_512_long | whole_image | 1 | 0.91 | 31.2 | 29.7 | 38.8 | 462368.0 |

Patch models tile each document at native resolution; their latency includes every tile forward pass plus tile assembly, which is why it is far above the single-resize whole-image models.

## Per-manipulation-type recall (test fakes)

| type | baseline (n, recall) | improved (n, recall) | improved_512 (n, recall) | improved_512_raw (n, recall) | improved_patch (n, recall) | improved_512_long (n, recall) |
|---|---|---|---|---|---|---|
| Crop_and_Replace | 22, 0.091 | 22, 0.045 | 22, 0.091 | 22, 0.091 | 22, 0.045 | 22, 0.227 |
| Inpaint_and_Rewrite | 162, 0.006 | 162, 0.148 | 162, 0.105 | 162, 0.123 | 162, 0.018 | 162, 0.315 |

## Confidence-gate analysis (four test groups)

Groups: A genuine/good, B genuine/degraded, C tampered/good, D tampered/degraded.
Fraud flag = POTENTIAL_TAMPERING; inconclusive = RESCAN_REQUIRED + MANUAL_REVIEW.

### baseline

| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |
|---|---|---|---|---|---|---|---|---|
| A_genuine_good | 93 | 0.0 | 0.0 | 0.0 | 0.914 | 0.914 | 0.914 | 0.0 |
| B_genuine_degraded | 56 | 0.0 | 0.0179 | 0.0 | 0.9286 | 0.9286 | 0.9286 | 0.0179 |
| C_tampered_good | 118 | 0.0169 | 0.0169 | 0.0 | 0.9746 | 0.9746 | 0.9915 | 0.0169 |
| D_tampered_degraded | 66 | 0.0 | 0.0152 | 0.0 | 0.8939 | 0.8939 | 0.8939 | 0.0152 |

### improved

| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |
|---|---|---|---|---|---|---|---|---|
| A_genuine_good | 93 | 0.0753 | 0.0753 | 0.0 | 0.6989 | 0.6989 | 0.7742 | 0.0753 |
| B_genuine_degraded | 56 | 0.0 | 0.0 | 0.0 | 0.8393 | 0.8393 | 0.8393 | 0.0 |
| C_tampered_good | 118 | 0.161 | 0.161 | 0.0 | 0.7288 | 0.7288 | 0.8898 | 0.161 |
| D_tampered_degraded | 66 | 0.0 | 0.0909 | 0.0 | 0.9394 | 0.9394 | 0.9394 | 0.0909 |

### improved_512

| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |
|---|---|---|---|---|---|---|---|---|
| A_genuine_good | 93 | 0.0 | 0.0 | 0.0 | 0.5806 | 0.5806 | 0.5806 | 0.0 |
| B_genuine_degraded | 56 | 0.0 | 0.0536 | 0.0 | 0.7679 | 0.7679 | 0.7679 | 0.0536 |
| C_tampered_good | 118 | 0.0508 | 0.0508 | 0.0 | 0.8983 | 0.8983 | 0.9492 | 0.0508 |
| D_tampered_degraded | 66 | 0.0 | 0.197 | 0.0 | 0.9848 | 0.9848 | 0.9848 | 0.197 |

### improved_512_raw

| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |
|---|---|---|---|---|---|---|---|---|
| A_genuine_good | 93 | 0.0 | 0.0 | 0.0 | 0.4839 | 0.4839 | 0.4839 | 0.0 |
| B_genuine_degraded | 56 | 0.0 | 0.0536 | 0.0 | 0.6786 | 0.6786 | 0.6786 | 0.0536 |
| C_tampered_good | 118 | 0.0678 | 0.0678 | 0.0 | 0.8305 | 0.8305 | 0.8983 | 0.0678 |
| D_tampered_degraded | 66 | 0.0 | 0.2121 | 0.0 | 0.9697 | 0.9697 | 0.9697 | 0.2121 |

### improved_patch

| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |
|---|---|---|---|---|---|---|---|---|
| A_genuine_good | 93 | 0.0108 | 0.0108 | 0.0 | 0.9032 | 0.9032 | 0.914 | 0.0108 |
| B_genuine_degraded | 56 | 0.0 | 0.0357 | 0.0 | 0.9107 | 0.9107 | 0.9107 | 0.0357 |
| C_tampered_good | 118 | 0.0169 | 0.0169 | 0.0 | 0.9746 | 0.9746 | 0.9915 | 0.0169 |
| D_tampered_degraded | 66 | 0.0 | 0.0303 | 0.0 | 0.9091 | 0.9091 | 0.9091 | 0.0303 |

### improved_512_long

| group | n | fraud flag (gate ON) | fraud flag (OFF) | rescan (ON) | manual (ON) | inconclusive (ON) | not cleared (ON) | not cleared (OFF) |
|---|---|---|---|---|---|---|---|---|
| A_genuine_good | 93 | 0.0215 | 0.0215 | 0.0 | 0.4516 | 0.4516 | 0.4731 | 0.0215 |
| B_genuine_degraded | 56 | 0.0 | 0.0714 | 0.0 | 0.625 | 0.625 | 0.625 | 0.0714 |
| C_tampered_good | 118 | 0.2712 | 0.2712 | 0.0 | 0.6695 | 0.6695 | 0.9407 | 0.2712 |
| D_tampered_degraded | 66 | 0.0 | 0.3636 | 0.0 | 0.9697 | 0.9697 | 0.9697 | 0.3636 |

## Localization

- **baseline**: {"n_tampered_test": 184, "n_with_derived_mask": 181, "n_without_annotation": 3, "cam_threshold": 0.5, "iou_mean": 0.0055, "dice_mean": 0.0108, "note": "masks are DERIVED from official field annotations (src + tampered field); SIDTD provides no official pixel-level tampering masks"}
- **improved**: {"n_tampered_test": 184, "n_with_derived_mask": 181, "n_without_annotation": 3, "cam_threshold": 0.5, "iou_mean": 0.0068, "dice_mean": 0.0127, "note": "masks are DERIVED from official field annotations (src + tampered field); SIDTD provides no official pixel-level tampering masks"}
- **improved_512**: {"n_tampered_test": 184, "n_with_derived_mask": 181, "n_without_annotation": 3, "cam_threshold": 0.5, "iou_mean": 0.0042, "dice_mean": 0.0081, "note": "masks are DERIVED from official field annotations (src + tampered field); SIDTD provides no official pixel-level tampering masks"}
- **improved_512_raw**: {"n_tampered_test": 184, "n_with_derived_mask": 181, "n_without_annotation": 3, "cam_threshold": 0.5, "iou_mean": 0.0064, "dice_mean": 0.0124, "note": "masks are DERIVED from official field annotations (src + tampered field); SIDTD provides no official pixel-level tampering masks"}
- **improved_patch**: {"n_tampered_test": 184, "n_with_derived_mask": 181, "n_without_annotation": 3, "tile_threshold": 0.5, "iou_mean": 0.0122, "dice_mean": 0.0238, "note": "tile-resolution map (stride 192 px, probability threshold 0.50) vs masks DERIVED from official field annotations; SIDTD has no official pixel-level masks"}
- **improved_512_long**: {"n_tampered_test": 184, "n_with_derived_mask": 181, "n_without_annotation": 3, "cam_threshold": 0.5, "iou_mean": 0.0077, "dice_mean": 0.0146, "note": "masks are DERIVED from official field annotations (src + tampered field); SIDTD provides no official pixel-level tampering masks"}

## Limitations & unmeasured claims

- SIDTD templates are *digital synthetic* document images; real-world photographed/scanned captures (the degraded-deployment regime) live in the corpus's video parts, which were deliberately not downloaded (79 GB). Group B therefore covers the naturally lower-quality tail of the template set, not field-captured photos.
- No official pixel-level masks: IoU/Dice use masks *derived* from the official field annotations and are not official localization ground truth.
- No real government-issued documents were used; SIDTD documents are synthetic.
- OCR reliability is not part of this dataset; gate experiments use synthetic OCR signals in tests only, and `ocr_reliability=None` in all reported numbers.
- Single train/val/test split (group-disjoint); no cross-validation was run within this hardware budget.
