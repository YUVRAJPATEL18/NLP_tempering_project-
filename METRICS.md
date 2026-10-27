# Current verification metrics

Cosine threshold: **0.6**. Model: `yunet-2023mar+sface-2021dec`. Pipeline: `orientation-retry-v2`.

Evaluation: 54 endpoint calls, 6 artificial identities, 54 successful responses, 0 processing failures.

The six-identity MIDV-2020 subset was reused during development; this is a functional regression evaluation, not held-out production validation.

| Metric | Original upright cards | Annotated portrait control |
| --- | ---: | ---: |
| Comparison accuracy | 100.00% | 100.00% |
| Balanced accuracy | 100.00% | 100.00% |
| False match rate | 0.00% | 0.00% |
| False non-match rate | 0.00% | 0.00% |
| True match rate / recall | 100.00% | 100.00% |
| Match precision | 100.00% | 100.00% |
| F1 | 100.00% | 100.00% |
| ROC AUC | 1.000 | 1.000 |
| Latency p50 | 0.470 s | 0.062 s |
| Latency p95 | 0.722 s | 0.081 s |

Original upright confusion matrix: {'true_positive': 6, 'false_negative': 0, 'false_positive': 0, 'true_negative': 6}.
Portrait control confusion matrix: {'true_positive': 6, 'false_negative': 0, 'false_positive': 0, 'true_negative': 30}.

Original rotated-card results: 6/6 completed comparisons, 0/6 processing failures, 100.00% genuine acceptance across all attempts.

Full-card false match rate: 0.00%; illustrative pairwise Wilson 95% interval: 0.00% to 39.03%. Zero observed errors do not establish a zero population error rate.

ROC AUC measures ranking. It does not certify the operating threshold.

## Metric definitions

- Accuracy: (true positives + true negatives) / completed comparisons.
- False match rate: wrong-reference pairs accepted / completed wrong-reference pairs.
- False non-match rate: genuine pairs rejected / completed genuine pairs.
- True match rate / recall: genuine pairs accepted / completed genuine pairs.
- Precision: correct matches / all accepted matches; depends on the pair mix.
- Balanced accuracy: mean of true match rate and true non-match rate.
- F1: harmonic mean of match precision and recall.
- ROC AUC: ranking quality across thresholds; not accuracy at the configured threshold.
- Processing failures: endpoint attempts that did not produce a comparison.

## Protocol and interpretation

Reference portraits are cropped from template annotations. Queries are separate upright/rotated scan files. Full-card queries are unmodified; the portrait control uses annotation-assisted crops and does not establish full-card extraction performance. No failed samples are discarded. Rotated cards have only genuine comparisons in this run; no rotated impostor rate was measured.

Threshold 0.6 was selected after development results were inspected. Validate thresholds on separate development identities, lock them, and evaluate unseen identities and capture conditions. Report true match rate at a predetermined false match rate, extraction/rotation failures, latency, and identity-level uncertainty.

No valid production false-match target can be demonstrated from only six artificial identities. Pairwise scores share identities and are not independent. Ground-truth-assisted portrait results do not measure full-card extraction.

Latency covers in-process FastAPI TestClient requests on this workstation, not upload networks or production load. Rotation retries add work. See `metrics.json` for rotated-card latency and pairwise intervals.

## Reproduction and evidence

Evidence: `datasets/midv2020_subset/results/pairs.json`, `pairs.csv`, `summary.json`, `enrollment.json`, and `metrics.json`. Model SHA-256 hashes are in `summary.json`; source image hashes are in the dataset download manifest.

Re-run inference: `python scripts/test_dataset_subset.py`. Recalculate this report: `python scripts/report_dataset_metrics.py`. Setup and integration: [HANDOFF.md](HANDOFF.md).
