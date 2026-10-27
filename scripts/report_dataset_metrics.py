"""Calculate verification metrics from saved endpoint results; never tune the model."""
import json
from pathlib import Path
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'datasets' / 'midv2020_subset' / 'results'


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z*z / total
    center = (proportion + z*z/(2*total))/denominator
    half = z * math.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total))/denominator
    return [max(0, center-half), min(1, center+half)]


def group_metrics(rows):
    ok = [row for row in rows if row['http_status'] == 200]
    genuine = [row for row in ok if row['expected_match']]
    impostor = [row for row in ok if not row['expected_match']]
    tp = sum(row['matched'] is True for row in genuine)
    fn = len(genuine)-tp
    fp = sum(row['matched'] is True for row in impostor)
    tn = len(impostor)-fp
    recall, specificity = divide(tp, tp+fn), divide(tn, tn+fp)
    positive_scores = [row['similarity'] for row in genuine]
    negative_scores = [row['similarity'] for row in impostor]
    auc = divide(sum((a > b) + .5*(a == b) for a in positive_scores for b in negative_scores),
                 len(positive_scores)*len(negative_scores))
    return dict(attempted_requests=len(rows), completed_comparisons=len(ok),
                processing_errors=len(rows)-len(ok), processing_failure_rate=divide(len(rows)-len(ok), len(rows)),
                confusion_matrix=dict(true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn),
                accuracy_completed_only=divide(tp+tn, len(ok)),
                balanced_accuracy_completed_only=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                precision=divide(tp, tp+fp), true_match_rate=recall,
                false_match_rate=divide(fp, len(impostor)), false_non_match_rate=divide(fn, len(genuine)),
                f1=divide(2*tp, 2*tp+fp+fn), roc_auc_completed_only=auc,
                genuine_accept_rate_all_attempts=divide(tp, sum(row['expected_match'] for row in rows)),
                false_match_rate_pairwise_wilson95=wilson(fp, len(impostor)),
                false_non_match_rate_pairwise_wilson95=wilson(fn, len(genuine)),
                latency_seconds=dict(p50=float(np.percentile([row['seconds'] for row in rows], 50)),
                                     p95=float(np.percentile([row['seconds'] for row in rows], 95))))


def main():
    rows = json.loads((DIRECTORY / 'pairs.json').read_text(encoding='utf-8'))
    summary = json.loads((DIRECTORY / 'summary.json').read_text(encoding='utf-8'))
    groups = {f'{row["mode"]}_{row["capture"]}' for row in rows}
    report = dict(threshold=summary['threshold'], unique_identities=summary['identity_count'],
                  groups={name: group_metrics([row for row in rows if f'{row["mode"]}_{row["capture"]}' == name])
                          for name in sorted(groups)},
                  interpretation=[
                      'Processing errors are excluded from comparison accuracy and reported separately.',
                      'Do not pool full-card and ground-truth-assisted portrait-control results.',
                      'Pairs reuse six identities and are dependent; Wilson intervals are illustrative pairwise intervals, not population guarantees.',
                      'ROC AUC measures ranking, not decision safety at the configured threshold.',
                      'This synthetic six-identity sample cannot establish production accuracy or a safe deployment threshold.'])
    (DIRECTORY / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Current verification metrics', '',
             f'Cosine threshold: **{summary["threshold"]}**. Model: `{summary["model_version"]}`. '
             f'Pipeline: `{summary.get("pipeline_version", "unspecified")}`.', '',
             f'Evaluation: {summary["planned_pairs"]} endpoint calls, {summary["identity_count"]} artificial identities, '
             f'{summary["successful_responses"]} successful responses, {summary["failed_requests"]} processing failures.', '',
             'The six-identity MIDV-2020 subset was reused during development; this is a functional regression evaluation, not held-out production validation.', '',
             '| Metric | Original upright cards | Annotated portrait control |',
             '| --- | ---: | ---: |']
    first = report['groups']['original_card_scan_upright']
    second = report['groups']['annotated_portrait_control_scan_upright']
    for label, key in [('Comparison accuracy', 'accuracy_completed_only'), ('Balanced accuracy', 'balanced_accuracy_completed_only'),
                       ('False match rate', 'false_match_rate'), ('False non-match rate', 'false_non_match_rate'),
                       ('True match rate / recall', 'true_match_rate'), ('Match precision', 'precision'), ('F1', 'f1')]:
        lines.append(f'| {label} | {first[key]:.2%} | {second[key]:.2%} |')
    lines += [f'| ROC AUC | {first["roc_auc_completed_only"]:.3f} | {second["roc_auc_completed_only"]:.3f} |',
              f'| Latency p50 | {first["latency_seconds"]["p50"]:.3f} s | {second["latency_seconds"]["p50"]:.3f} s |',
              f'| Latency p95 | {first["latency_seconds"]["p95"]:.3f} s | {second["latency_seconds"]["p95"]:.3f} s |', '',
              f'Original upright confusion matrix: {first["confusion_matrix"]}.',
              f'Portrait control confusion matrix: {second["confusion_matrix"]}.', '',
              f'Original rotated-card results: {report["groups"]["original_card_scan_rotated"]["completed_comparisons"]}/6 completed comparisons, '
              f'{report["groups"]["original_card_scan_rotated"]["processing_errors"]}/6 processing failures, '
              f'{report["groups"]["original_card_scan_rotated"]["genuine_accept_rate_all_attempts"]:.2%} genuine acceptance across all attempts.', '',
              f'Full-card false match rate: {first["false_match_rate"]:.2%}; illustrative pairwise Wilson 95% interval: '
              f'{first["false_match_rate_pairwise_wilson95"][0]:.2%} to {first["false_match_rate_pairwise_wilson95"][1]:.2%}. '
              'Zero observed errors do not establish a zero population error rate.', '',
              'ROC AUC measures ranking. It does not certify the operating threshold.', '',
              '## Metric definitions', '',
              '- Accuracy: (true positives + true negatives) / completed comparisons.',
              '- False match rate: wrong-reference pairs accepted / completed wrong-reference pairs.',
              '- False non-match rate: genuine pairs rejected / completed genuine pairs.',
              '- True match rate / recall: genuine pairs accepted / completed genuine pairs.',
              '- Precision: correct matches / all accepted matches; depends on the pair mix.',
              '- Balanced accuracy: mean of true match rate and true non-match rate.',
              '- F1: harmonic mean of match precision and recall.',
              '- ROC AUC: ranking quality across thresholds; not accuracy at the configured threshold.',
              '- Processing failures: endpoint attempts that did not produce a comparison.', '',
              '## Protocol and interpretation', '',
              'Reference portraits are cropped from template annotations. Queries are separate upright/rotated scan files. Full-card queries are unmodified; the portrait control uses annotation-assisted crops and does not establish full-card extraction performance. No failed samples are discarded. Rotated cards have only genuine comparisons in this run; no rotated impostor rate was measured.', '',
              'Threshold 0.6 was selected after development results were inspected. Validate thresholds on separate development identities, lock them, and evaluate unseen identities and capture conditions. Report true match rate at a predetermined false match rate, extraction/rotation failures, latency, and identity-level uncertainty.', '',
              'No valid production false-match target can be demonstrated from only six artificial identities. Pairwise scores share identities and are not independent. Ground-truth-assisted portrait results do not measure full-card extraction.', '',
              'Latency covers in-process FastAPI TestClient requests on this workstation, not upload networks or production load. Rotation retries add work. See `metrics.json` for rotated-card latency and pairwise intervals.', '',
              '## Reproduction and evidence', '',
              'Evidence: `datasets/midv2020_subset/results/pairs.json`, `pairs.csv`, `summary.json`, `enrollment.json`, and `metrics.json`. Model SHA-256 hashes are in `summary.json`; source image hashes are in the dataset download manifest.', '',
              'Re-run inference: `python scripts/test_dataset_subset.py`. Recalculate this report: `python scripts/report_dataset_metrics.py`. Setup and integration: [HANDOFF.md](HANDOFF.md).']
    (ROOT / 'METRICS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
