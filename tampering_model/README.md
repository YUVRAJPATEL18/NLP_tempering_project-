# NLP_tempering_project- (IDGuard)

**IDGuard** is a production-grade forensic deep learning system designed to detect **image tampering, forgeries, and digital manipulation** in identity documents (such as National IDs, Passports, and Student ID Cards).

---

## 🚀 Key Features

* **Fine-Tuned ResNet-18 Backbone:** Trained at $512 \times 512$ resolution with differential learning rates to preserve fine sub-pixel text and inpainting boundaries.
* **Explainability via Grad-CAM:** Generates visual heatmaps highlighting the exact forged regions (text fields, signatures, photos).
* **Confidence-Gated Decision Engine:** Routes documents into 4 operational states:
  - `✅ GENUINE (Clean Document)` (Tampering probability $< 20\%$)
  - `🚨 FRAUD DETECTED (Tampered)` (Tampering probability $\ge 50\%$)
  - `⚠️ MANUAL REVIEW REQUIRED` (Borderline score or degraded scan quality)
  - `🔄 RESCAN REQUIRED` (Unusable capture blur/defocus)
* **REST API & CLI:** Ready-to-deploy FastAPI backend and command-line analyzer.

---

## 📁 Repository Structure

```
NLP_tempering_project-/
├── api/
│   └── app.py                  # FastAPI service (REST endpoints)
├── src/
│   ├── idguard/
│   │   ├── model.py            # TamperingModel (ResNet-18 + binary head)
│   │   ├── inference.py        # TamperingPredictor & TamperingPrediction pipeline
│   │   ├── decision_engine.py  # 4-state confidence-aware decision engine
│   │   ├── heatmap.py          # Grad-CAM heatmap & bounding box extraction
│   │   ├── forensic.py         # Capture quality (Laplacian blur, noise, ELA)
│   │   ├── dataset.py          # SIDTD dataset loader & group-aware splitting
│   │   └── train.py            # Whole-image model training pipeline
│   └── evaluate.py             # Reproducible evaluation & metric benchmarking
├── scripts/
│   ├── check_image.py          # One-line CLI tool to check any document
│   ├── download_sidtd.py       # SIDTD dataset acquisition & MD5 verification
│   └── prepare_dataset.py      # Split creation & quality calibration
├── checkpoints/
│   └── kaggle_finetuned/       # Fine-tuned model weights (best.pt)
├── config.yaml                 # Central configuration (hyperparams, thresholds)
├── requirements.txt            # Python dependencies
└── tests/                      # Automated pytest test suite
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YUVRAJPATEL18/NLP_tempering_project-.git
   cd NLP_tempering_project-
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🔍 How to Check Any Document Image

Run the CLI tool directly from your terminal:

```bash
python scripts/check_image.py --image path/to/document.jpg
```

**Example Output:**
```text
Analyzing: document.jpg ...
=======================================================
  VERDICT              : 🚨 FRAUD DETECTED (Tampered)
  Tampering Probability: 80.62%
  Image Quality Score  : 85.81%
  Model Version        : sidtd-resnet18-kaggle-finetuned-e5
=======================================================
Decision Reasons:
  • forensic tampering probability 0.806 >= calibrated threshold 0.500

Suspicious Regions Localized: 2 box(es)
  Box 1: [x0=0.596, y0=0.631, x1=1.000, y1=1.000]

Visual heatmap overlay saved to: tamper_heatmap.png
```

---

## 🌐 Running the REST API

Start the FastAPI service:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

### Endpoints:
* **`GET /health`**: Health check and model version status.
* **`POST /v1/tampering/analyze`**: Multipart upload endpoint for document analysis.
  ```bash
  curl -X POST "http://localhost:8080/v1/tampering/analyze" \
    -F "image=@path/to/document.jpg" \
    -F "include_heatmap=true"
  ```

---

## 🧪 Running Automated Tests

Run the full test suite with:

```bash
PYTHONPATH=. pytest
```
All 59 unit and integration tests validate the API, inference engine, Grad-CAM overlays, and decision logic.
