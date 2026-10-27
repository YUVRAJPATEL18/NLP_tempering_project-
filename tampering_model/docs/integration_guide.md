# IDGuard tampering detection — integration guide

Everything in this repository is the **tampering-detection component only**. OCR, face
matching, database verification and the frontend were deliberately left untouched.

```
config.yaml                     model / threshold / quality configuration
src/idguard/
  model.py                      TamperingModel, save/load checkpoint, get_default_model
  dataset.py                    manifest loading, validation, group-aware splits, transforms
  train.py                      whole-image training (baseline + fine-tune)
  patch.py                      native-resolution tiling helpers (patch experiment)
  train_patch.py                patch-level training (see the report for why it was rejected)
  inference.py                  TamperingPredictor -> TamperingPrediction (production path)
  heatmap.py                    GradCAM, CAM -> regions, overlay
  forensic.py                   quality score, ELA, annotation-derived region masks
  decision_engine.py            four-state confidence-gated decision layer
src/evaluate.py                 reproducible baseline-vs-improved evaluation
api/app.py                      FastAPI service
scripts/download_sidtd.py       dataset acquisition (figshare API, MD5-checked)
scripts/prepare_dataset.py      integrity validation + splits + quality table
tests/                          pytest suite
```

## 1. Produce the tampering prediction

```python
from idguard.inference import TamperingPredictor

predictor = TamperingPredictor()            # loads config.yaml -> model.checkpoint once
pred = predictor.predict(image_bytes, ocr_reliability=0.87)
payload = pred.to_dict()                    # JSON-safe, no internal paths
```

`image_bytes` may also be a `PIL.Image` or a `uint8` HxWx3 numpy array. Anything else raises
`InvalidImageError` (the API maps it to HTTP 400).

Returned schema (stable):

| field | meaning |
|---|---|
| `tampering_probability` | calibrated forensic probability in [0,1] — *not* a verdict |
| `tampering_detected` | True only when the decision is `POTENTIAL_TAMPERING` |
| `quality_score` | sharpness/contrast capture-quality score in [0,1] |
| `ocr_reliability` | echoed OCR confidence, `None` when OCR was not run |
| `predicted_regions` | normalized `[x0,y0,x1,y1]` suspicious boxes (may be empty) |
| `decision` | `NO_TAMPERING_DETECTED` \| `POTENTIAL_TAMPERING` \| `RESCAN_REQUIRED` \| `MANUAL_REVIEW` |
| `decision_reasons` | human-readable justification lines |
| `model_version` | checkpoint identity, e.g. `sidtd-resnet18-improved_512-e3` |

Add `include_heatmap=True` to receive `heatmap_png_base64` (a square PNG at the model input
resolution) and `timings_ms`. For a displayable overlay that keeps the **source** image size:

```python
pred, overlay = predictor.predict_with_overlay(image_bytes)   # overlay.size == source size
```

## 2. Decision layer contract

The three signals stay separate on purpose:

* `quality_score` — capture quality (blur/contrast). **Never** evidence of fraud.
* `ocr_reliability` — caller-supplied OCR confidence. **Never** evidence of fraud on its own.
* `tampering_probability` — the only forensic signal.

Rules implemented in `decision_engine.decide`:

1. Quality below `quality.rescan_below` → `RESCAN_REQUIRED` (unjudgeable capture).
2. probability ≥ `tamper_high` and OCR confidence low → `MANUAL_REVIEW` (strong forensics must
   not be silently dropped by a weak OCR read).
3. probability ≥ `tamper_high` → `POTENTIAL_TAMPERING`.
4. `tamper_low` ≤ probability < `tamper_high`, or degraded quality, or OCR confidence below
   `decision.ocr_min` → `MANUAL_REVIEW` (conflicting/insufficient evidence).
5. otherwise → `NO_TAMPERING_DETECTED`.

`tamper_high` / `tamper_low` are **fitted on the validation split** during training and stored
inside the checkpoint (`thresholds_decision`); the values in `config.yaml` are fallbacks for the
untrained model. Do not retune them on the test split.

## 3. FastAPI service

```bash
.venv/bin/uvicorn api.app:app --port 8080
```

* `GET /health` → `{status, service, model_version, model_loaded}`
* `POST /v1/tampering/analyze` (multipart)
  * `image` (required) — JPEG/PNG/WEBP, ≤ 25 MiB
  * `ocr_reliability` (optional, 0–1)
  * `include_heatmap` (bool, default false)
  * `gate_enabled` (optional bool — force the confidence gate on/off for A/B analysis)

The model is loaded once in the FastAPI lifespan and reused; each request runs in eval mode
under `torch.inference_mode()`. Responses never contain filesystem paths; oversized uploads are
rejected with 413 and unreadable images with 400.

Multipart uploads need `python-multipart` (pinned in `requirements.txt`); without it FastAPI fails
at import time. A missing checkpoint raises `FileNotFoundError` at startup instead of silently
serving an untrained head.

To point the service at a different checkpoint:

```python
from api.app import create_app
app = create_app(checkpoint="checkpoints/improved_512/best.pt")
```

## 4. Configuration

`config.yaml` owns everything tunable: `model.checkpoint`, `data.image_size`, per-run training
recipes (`baseline`, `improved`, `improved_512`, `patch`), `thresholds`, `quality`,
`decision`, `heatmap` and `inference`. A checkpoint records the input size, normalization
statistics, temperature, thresholds and split digest it was produced with, so inference does not
depend on the config staying byte-identical — but keep `model.checkpoint` in sync with the model
you intend to serve.

## 5. Reproducing the pipeline

```bash
.venv/bin/python scripts/download_sidtd.py     # 1.27 GB, resumable, MD5-verified
.venv/bin/python scripts/prepare_dataset.py    # validate + split + quality table
PYTHONPATH=src .venv/bin/python -m idguard.train --run improved_512
PYTHONPATH=src .venv/bin/python src/evaluate.py \
    --models baseline=checkpoints/baseline/best.pt improved=checkpoints/improved/best.pt \
             improved_512=checkpoints/improved_512/best.pt
PYTHONPATH=src .venv/bin/python -m pytest -q      # 59 tests: dataset, decision, inference, patch, API
```

Results land in `experiments/tampering/results/` (`metrics_<name>.json`,
`confusion_<name>.csv/.png`, `roc_curves.png`, `report.md`, `samples/`).

## 6. Operational notes

* Thresholds, temperature and the input size travel inside the checkpoint — deploy the `.pt`
  file, not just the config.
* `predicted_regions` come from GradCAM and are coarse (a 7×7 grid at 256 px, 16×16 at 512).
  Treat them as "look here", not as a manipulation mask.
* Localization numbers in the report are measured against **derived** field rectangles, because
  SIDTD ships no pixel-level ground-truth masks.
* SIDTD documents are synthetic; results do not automatically transfer to photographs of real
  physical documents.
