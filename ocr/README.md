# ID-Guard OCR pipeline

This directory contains the standalone PaddleOCR document pipeline and its early image-quality gate. The gate decodes and assesses the original upload before image resizing, enhancement, OCR, or any risk scoring. Accepted images run OCR; OCR confidence below the configured threshold returns `OCR_UNCERTAIN` and defers risk scoring. Rejected images return `RESCAN_REQUIRED` with OCR and downstream checks marked skipped.

The OCR pipeline is intentionally kept independent of `tampering_model/`. This branch contains the tampering model, but there is no integrated application call between the OCR API and that model yet. The API will not claim forensic results unless an explicit integration is added.

Thresholds are provisional, configurable heuristics and have not been calibrated against a labeled validation set. See `RESEARCH_EVALUATION.md`.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn ocr_module.api.routes:app --reload --host 127.0.0.1 --port 8000
```

Run tests from this directory:

```bash
pytest ocr_module/tests -q
```

The model-only confidence helper can be run as:

```bash
python run_model_ocr_confidence.py /path/to/document.jpg
```
