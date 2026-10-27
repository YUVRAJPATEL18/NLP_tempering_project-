"""
OCR evaluation metrics: CER, WER, Field Precision, Recall, F1, and Exact Match Accuracy.
"""

from typing import Dict, Any, List
from rapidfuzz.distance import Levenshtein


class MetricsCalculator:
    """Calculates quantitative OCR performance metrics against ground truth labels."""

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculates Character Error Rate (CER): Levenshtein distance / len(reference).
        """
        ref = reference.strip()
        hyp = hypothesis.strip()
        if not ref:
            return 0.0 if not hyp else 1.0
        dist = Levenshtein.distance(ref, hyp)
        return float(dist) / float(len(ref))

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculates Word Error Rate (WER): Word Levenshtein distance / total reference words.
        """
        ref_words = ref = reference.strip().split()
        hyp_words = hyp = hypothesis.strip().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        dist = Levenshtein.distance(ref_words, hyp_words)
        return float(dist) / float(len(ref_words))

    @staticmethod
    def calculate_field_metrics(
        ground_truth_fields: Dict[str, str],
        predicted_fields: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculates Field Exact Accuracy, Precision, Recall, and F1.
        """
        if not ground_truth_fields:
            return {
                "exact_accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0
            }

        tp = 0
        fp = 0
        fn = 0
        exact_matches = 0

        for key, gt_val in ground_truth_fields.items():
            pred_val = predicted_fields.get(key, "").strip()
            gt_clean = str(gt_val).strip()

            if pred_val == gt_clean:
                exact_matches += 1
                tp += 1
            elif pred_val:
                fp += 1
            else:
                fn += 1

        total_gt = len(ground_truth_fields)
        exact_accuracy = exact_matches / float(total_gt) if total_gt > 0 else 0.0
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "exact_accuracy": round(exact_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
