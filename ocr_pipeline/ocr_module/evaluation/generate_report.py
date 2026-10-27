"""
Report generation utility saving JSON and CSV evaluation summaries.
"""

import os
import json
import csv
from typing import Dict, Any
from ..schemas import EvaluationReport


class ReportGenerator:
    """Exports evaluation reports to JSON and CSV formats."""

    @staticmethod
    def save_report(report: EvaluationReport, output_prefix: str = "ocr_eval_report"):
        """Saves evaluation report to .json and .csv files."""
        out_dir = os.path.dirname(output_prefix) or "."
        os.makedirs(out_dir, exist_ok=True)

        # Save JSON
        json_path = f"{output_prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, indent=4)

        # Save CSV
        csv_path = f"{output_prefix}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(list(report.model_dump().keys()))
            writer.writerow(list(report.model_dump().values()))

        return json_path, csv_path
