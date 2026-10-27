# ID-Guard research evaluation plan

## Current implementation boundary

The repository includes PaddleOCR parsing, field extraction, format/date validation, and OCR dataset metric helpers. It does not include a tampering model, forensic feature extractor, trusted reference-data source, face verifier, identity database, labeled calibration set, or ground-truth tampering masks. The API therefore returns empty field-forensics results and explicitly warns that authenticity was not assessed. The review-priority score is a configurable, uncalibrated heuristic, not a probability of authenticity.

## Proposed experiments

1. Compare current regex extraction against label-aware and layout-aware variants on a document-type-stratified held-out set. Report field exact-match accuracy, precision, recall, F1, CER, and WER with confidence intervals.
2. Annotate field polygons and tampering masks. Compare full-page forensic inference with OCR-region crops and report field-to-region association accuracy, IoU, and localization precision/recall. Preserve original-image coordinates through every preprocessing transform.
3. Compare OCR-only, validation-only, and evidence-fusion baselines using a train/validation/test split grouped by document identity and template. Select thresholds on validation data only. Report ROC-AUC, precision/recall/F1 and false-positive/false-negative rates on the untouched test set.
4. If a calibration set is available, fit calibration only on training/validation folds and report Brier score and ECE on held-out test data. Do not claim calibration from heuristic scores.
5. Report latency and peak memory by image size, preprocessing mode, and CPU/GPU. Test generalization by withholding document types/templates from training and threshold selection.

No experimental results are claimed by this implementation.

## Early image-quality gate calibration

The default upload-gate cutoffs in `OCRConfig` are provisional starting points: shorter edge 360 px; Laplacian variance reject below 100 and warn below 180; mean brightness reject below 25 and warn below 40; glare proxy reject at 0.98 only when global contrast is also below 8; global contrast reject below 8 and warn below 35. OCR uncertainty defaults to mean OCR confidence below 0.40. These values were not calibrated on a representative labeled document set. The glare measurement is a bright, low-saturation pixel fraction and can indicate white card stock; it is only a coarse warning except when the image is nearly uniform. Severe rotation and perspective are not currently auto-rejected because the repository has no validated detector for those conditions.

For calibration, collect human-reviewed clear, blurred, under/overexposed, glare-affected and low-resolution images across document types and capture devices. Choose quality cutoffs on a validation split by measuring OCR field exact-match and false rejection rate, then freeze them before evaluating on a separate test split. Keep the original upload available to forensic analysis in the downstream system; the OCR endpoint currently processes it in memory and this repository contains no downstream verifier.
