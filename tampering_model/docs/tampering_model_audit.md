# IDGuard — Image Tampering Detection Model Audit

**Date:** 2026-10-26
**Scope:** Tampering-detection component only. OCR, face matching, database verification, and the frontend were out of scope and are not touched by this work.

---

## 1. Repository state at audit time

The workspace (`NLP_Tempering_model`) contained **no source code** — only `.freebuff/` workspace
metadata. There was no existing `TamperingModel`, `TamperingPrediction`, `ELASRMModel`,
`TruForModel`, `get_default_model`, no FastAPI backend, no pretrained weights and no evaluation
scripts to reuse. A repository-wide search found zero candidate files.

**Consequence:** there is no baseline model *inside the repo* to improve. The fastest reliable path
to the stated goal was therefore:

1. Scaffold the tampering-detection component of IDGuard with clean, reusable interfaces
   (`TamperingModel`, `TamperingPrediction`, `get_default_model`) so a future backend can consume it.
2. Establish a **measured baseline** — a standard frozen-backbone transfer-learning linear probe —
   on a real, licensed identity-document tampering dataset.
3. Fine-tune the same architecture properly and prove improvement against that baseline on the
   same held-out test set.

Nothing was overwritten because nothing existed. All artifacts are additive.

## 2. Hardware and runtime environment (measured)

| Item | Value |
|---|---|
| Machine | MacBook Air, Apple M1 (arm64), 8 GB RAM |
| OS | Darwin 24.6.0 (macOS) |
| Python | 3.9.6 (system) → project virtualenv `.venv` |
| PyTorch | 2.5.1, torchvision 0.20.1 (pinned) |
| Acceleration | MPS available (`torch.backends.mps.is_available() == True`) |
| Free disk at start | ~16 GB |
| Framework choice | PyTorch (no pre-existing stack to reuse) |

Training/inference budget: ResNet-18-scale models at 256×256, batch 4–8, are comfortably within
8 GB RAM on MPS. No GPU node, no multi-GPU, no large-model training.

## 3. Dataset audit (SIDTD)

| Item | Value |
|---|---|
| Dataset | SIDTD — Synthetic dataset of ID and Travel Documents (Boned et al., 2024) |
| Landing pages | https://tc11.cvc.uab.es/datasets/SIDTD_1 (slow/unreachable at audit time), mirrored on figshare |
| DOI | 10.6084/m9.figshare.27136242.v1 |
| Licence | **CC BY 4.0** (verified via figshare API) |
| Version | 1 (published 2024-12-19) |
| Full corpus | 79.4 GB (`videos.zip` 54.5 GB, `clips.zip` 23.5 GB, `templates.zip` 1.27 GB) |
| Subset acquired | `templates.zip` only — 1.27 GB, image-level task data |

### Structure of the acquired subset

- `templates/Images/reals/` — 1,000 genuine template images (10 nationalities × 100 documents)
- `templates/Images/fakes/` — 1,222 tampered template images (`<base>_fake_<type>_<n>.jpg`)
- `templates/Annotations/fakes/*.json` — one JSON per tamper, **image-level metadata**:
  `ctype` (manipulation type, e.g. `Inpaint_and_Rewrite`, crop-and-replace variants),
  `src` (source genuine image), `field` (which document field was altered)
- `templates/Annotations/reals/*.json` — 10 VIA-format files with **rectangle annotations for
  14 document fields** (photo, name, expiry_date, MRZ-related fields, …) on every genuine image

### Labels and annotations — what exists, what does not

- **Exists:** image-level genuine/tampered labels; manipulation type; manipulated field name;
  field rectangles on genuine templates (from which an *approximate* tamper-region mask can be
  derived for a fake via `src` + `field`, since fakes are pixel-aligned edits of their source).
- **Does not exist:** official pixel-level tampering ground-truth masks.

**Decision:** train and report an **image-level classifier** as the primary task. IoU/Dice will be
reported only where a derived region mask is used, and such numbers will be labelled explicitly as
*derived from field annotations*, never as official pixel ground truth.

### Official split problem (leakage)

The official `unbalanced_hold_out` split (952 train / 119 val / 120 test) **leaks base documents
across splits**: 55 groups shared train∩val, 64 train∩test, 11 val∩test, because a tampered image
and its genuine source can land in different splits. It also covers only 819 of the 2,222 images
and is heavily imbalanced (808 reals / 144 fakes in train).

**Decision:** build a **group-aware 70/15/15 split** where the group key is the base document ID
(filename stripped of `_fake_*`), so a genuine image and all its tampered variants always stay in
one split. Deterministic seed; recorded in the manifest. The official split CSVs are still
archived for reference.

## 4. Current architecture (implemented)

- **`TamperingModel`** — torchvision ResNet-18 (ImageNet-pretrained) with a binary head
  (dropout + linear → 1 logit). Single shared code path for baseline and improved runs;
  only the training schedule differs.
- **Baseline (measured reference):** frozen backbone + trained linear head (linear probe).
- **Improved:** full fine-tune with differential learning rates (backbone ≪ head), conservative
  augmentation, class-weighted loss, early stopping, best-checkpoint selection, threshold
  calibrated on the validation set only.
- **Complementary non-trainable evidence:** ELA map + image-quality metrics (sharpness, noise,
  blocking) computed in `forensic.py`; never used alone as proof of tampering.
- **Heatmaps:** Grad-CAM over `layer4`, upsampled and verified to align with the input image.

### Weight/trainability/licence check

| Weight | Source | Trainable | Licence |
|---|---|---|---|
| ResNet-18 ImageNet-1k | torchvision (`resnet18-*.pth`) | Yes (fine-tune) | BSD-style, pretrained on ImageNet-1k; permitted for this use |
| SIDTD data | figshare, CC BY 4.0 | n/a (training data) | CC BY 4.0 — attribution required, commercial/research use allowed |

No weight is overwritten; every checkpoint is written to a new file under `checkpoints/`.

## 5. Missing components (before this work) → now provided

| Missing | Provided |
|---|---|
| No model / weights | `src/idguard/model.py` (`TamperingModel`, `get_default_model`) |
| No dataset pipeline | `scripts/download_sidtd.py`, `src/idguard/dataset.py` |
| No training | `src/idguard/train.py` |
| No evaluation | `src/evaluate.py` |
| No inference / heatmap | `src/idguard/inference.py`, `src/idguard/heatmap.py` |
| No decision layer | `src/idguard/decision_engine.py` |
| No configuration | `config.yaml` |
| No API | `api/app.py` (FastAPI) |
| No tests | `tests/` |

## 6. Potential integration conflicts

1. **Empty repo:** there is no existing public interface to preserve, so the interfaces above are
   the contract. If a real IDGuard backend already exists elsewhere, map its calls onto
   `TamperingModel` / `TamperingPrediction` rather than re-implementing.
2. **Python 3.9.6:** pins `torch==2.5.1` (last LTS-ish line supporting 3.9 well with stable MPS).
3. **8 GB RAM / 16 GB disk:** dataset subset is bounded to `templates.zip` (1.27 GB) and only the
   best checkpoint is retained per run.
4. **No pixel masks:** any request for pixel-precision localization must be qualified — see §3.
5. **OCR coupling:** OCR confidence must remain a *separate* signal; it flows into the decision
   engine as an independent input and is never merged with tampering probability.

## 7. Recommended training strategy (chosen)

**Option A** (fine-tune existing model) applied to the scaffolded ResNet-18: freeze backbone first
(linear probe = baseline), then unfreeze with a low backbone LR. If validation gains plateau,
stop — report honestly. Pixel-level segmentation (Option B) is not pursued because the dataset has
no pixel masks; classical forensic detectors are preserved as complementary evidence (Option C
hybrid) through the decision layer.

## 8. Expected compute and storage

| Item | Estimate / measured |
|---|---|
| Dataset download | 1.27 GB (templates.zip only; 78 GB of videos/clips deliberately skipped) |
| Extracted data | ≈ 1.3 GB (images + JSON annotations) |
| Env (torch + deps) | ≈ 2 GB |
| Train time/epoch | seconds–tens of seconds (ResNet-18, ≤2.2k images, MPS) |
| Full training run | minutes per configuration on M1 |
| Checkpoints | best-only, one per run (≈ 45 MB each) |

## 9. Reproducibility commitments

Fixed seeds (python/numpy/torch), pinned dependency versions, manifest-recorded splits,
config-driven thresholds, evaluation that loads both checkpoints and scores the *same* test rows.
All results written under `experiments/tampering/results/`.

---

# 10. Post-training addendum — what the measurements changed

This section revises §4/§7 with evidence obtained after the initial audit. Two of the audit's
planning assumptions turned out to be wrong; both are recorded here rather than quietly dropped.

## 10.1 The 256 px whole-document input was the bottleneck (not model capacity)

A frozen-backbone probe (`experiments/tampering/resolution_probe.py`: ImageNet ResNet-18 GAP
features + logistic regression fitted on **train**, scored on **val**) shows:

| input size | frozen-feature val ROC-AUC |
|---|---|
| 256 | 0.664 |
| 384 | 0.851 |
| **512** | **0.932** |

Rewritten text fields are ~2 px tall on a 2167×1360 SIDTD card; a 256 px square resize destroys
them. §7's plan (fine-tune at 256 px) was therefore misdirected: the first full fine-tune at
256 px only reached val AUC 0.697, because it was learning from a representation that no longer
contained the forensic cue.

## 10.2 The patch/tiled approach was tried and rejected

Tiling at native resolution looked dramatically better — within-image patch AUC 0.977 on val —
ut *document-level* AUC stayed at 0.55. A controlled follow-up
(`experiments/tampering/patch_probe2.py`) made positives and negatives **field-matched crops**:
the signal collapsed to 0.588 document AUC / 0.671 within-image pair accuracy.

Interpretation: the patch model had learned the **text-density confound** (SIDTD fakes are
rewrites of text fields, so "this crop contains text" separated the probe's positives), not
tampering. The patch checkpoint (`checkpoints/improved_patch/best.pt`) is kept for the record and
excluded from production. Its 405–800 ms/document latency would also have made it a poor fit.

## 10.3 What actually improved the model — and what was measured

All rows are the same group-disjoint held-out **test** split (333 images, digest
`697c850e9401825e`); thresholds are fitted on validation only.

| run | change vs previous row | val AUC | test AUC | test F1 | test FPR |
|---|---|---|---|---|---|
| `baseline` | frozen backbone, linear head, 256 px | 0.5688 | 0.5813 | 0.032 | 0.007 |
| `improved` | full fine-tune, 256 px, aug | 0.6966 | 0.6948 | 0.232 | 0.047 |
| `improved_512` | frozen backbone @512 px, aug, 6 ep | 0.7602 | 0.7750 | 0.185 | 0.020 |
| `improved_512_raw` | same, augmentation **off**, 6 ep | 0.7799 | 0.7800 | 0.211 | 0.020 |
| **`improved_512_long`** | same, 14 ep, lr 5e-4 (best epoch 11) | **0.8522** | **0.8437** | **0.455** | 0.040 |
| `improved_patch` | native-resolution tiles (rejected, §10.2) | 0.5529† | 0.5511 | 0.042 | 0.020 |

† patch run reports document-level val AUC.

Findings:

1. **Input resolution dominates**: 256→512 px moved test AUC 0.695 → 0.844 at equal capacity.
2. **The augmentation recipe hurt this task**: identical recipe with augmentation off gained
   +0.02 val / +0.005 test AUC at 6 epochs. The mild affine/scale jitter perturbs exactly the
   sub-pixel resampling cues the model needs.
3. **The head was undertrained, not the features**: a well-converged linear model on the same
   frozen features reached val AUC 0.93, but 6 epochs at lr 1e-3 only reached 0.78. Extending to
   14 epochs at lr 5e-4 closed most of that gap (0.852). The residual gap to 0.93 is the
   difference between a 512-dim linear head trained by SGD and logistic regression on
   standardised features — worth one more experiment, not yet done.
4. **Absolute detection quality is still modest**: at the validation-fitted operating point the
   best model flags 30% of tampered documents with 4% false positives on genuine ones
   (precision 0.90). This is a real, honest limitation — the model is a useful triage signal,
   not an automatic verdict engine.

## 10.4 Decision-layer correction (degraded-quality handling)

The first gate implementation compared quality against `rescan_below` (0.30, the
"unusable capture" floor) when deciding whether a capture was *degraded*. Since SIDTD never falls
below 0.637, that branch could never fire and degraded genuine documents were still flagged
`POTENTIAL_TAMPERING`. The gate now takes the calibrated `quality_degraded` threshold
(0.8251) as a separate input, so a degraded-but-readable capture with strong forensic evidence
escalates to `MANUAL_REVIEW` instead of an automatic fraud flag. Measured on the best model
(`improved_512_long`, test split):

| group | n | fraud flag OFF | fraud flag ON | manual review ON | cleared ON |
|---|---|---|---|---|---|
| A genuine / good | 93 | 0.022 | 0.022 | 0.452 | 0.527 |
| B genuine / degraded | 56 | 0.071 | **0.000** | 0.625 | 0.375 |
| C tampered / good | 118 | 0.271 | 0.271 | 0.670 | 0.059 |
| D tampered / degraded | 66 | 0.364 | **0.000** | 0.970 | 0.030 |

The gate removes false fraud flags on degraded genuine documents (7.1% → 0%) **without clearing
tampering**: tampered degraded documents go to review (97% of them), so their not-cleared rate
stays at 0.97. The price is a high human-review load (45% of clean genuine documents fall in the
borderline band) — a direct consequence of the model's limited separability, and the first thing
to revisit if review capacity is constrained.

## 10.5 Production choice

`config.yaml` → `model.checkpoint: checkpoints/improved_512_long/best.pt` — selected on
**validation** AUC (0.8522), never on test. All other checkpoints remain unmodified for the
comparison. Inference cost at 512 px on M1 MPS: ~31 ms/document (batch 1, warm), 0.46 s load.
