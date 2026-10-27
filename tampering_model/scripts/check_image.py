#!/usr/bin/env python3
"""CLI utility to check an identity document image for tampering.

Usage:
    .venv/bin/python scripts/check_image.py --image path/to/document.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idguard.inference import TamperingPredictor


def main():
    parser = argparse.ArgumentParser(description="Check an identity document for tampering.")
    parser.add_argument(
        "--image",
        type=str,
        default="datasets/sidtd/templates/Images/fakes/est_id_03_fake_9_4.jpg",
        help="Path to the document image file (JPG, PNG, WEBP)",
    )
    parser.add_argument(
        "--save-heatmap",
        type=str,
        default="tamper_heatmap.png",
        help="Output path to save the visual Grad-CAM heatmap overlay",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        return 1

    print(f"\nAnalyzing: {image_path.name} ...")
    predictor = TamperingPredictor()

    from PIL import Image

    with Image.open(image_path) as pil_img:
        pred, overlay = predictor.predict_with_overlay(pil_img)

    # Status icon based on decision
    icons = {
        "NO_TAMPERING_DETECTED": "✅ GENUINE (Clean Document)",
        "POTENTIAL_TAMPERING": "🚨 FRAUD DETECTED (Tampered)",
        "MANUAL_REVIEW": "⚠️  MANUAL REVIEW REQUIRED (Borderline)",
        "RESCAN_REQUIRED": "🔄 RESCAN REQUIRED (Unusable Image)",
    }
    status = icons.get(pred.decision, pred.decision)

    print("\n" + "=" * 55)
    print(f"  VERDICT              : {status}")
    print(f"  Tampering Probability: {pred.tampering_probability * 100:.2f}%")
    print(f"  Image Quality Score  : {pred.quality_score * 100:.2f}%")
    print(f"  Model Version        : {pred.model_version}")
    print("=" * 55)

    print("\nDecision Reasons:")
    for r in pred.decision_reasons:
        print(f"  • {r}")

    if pred.predicted_regions:
        print(f"\nSuspicious Regions Localized: {len(pred.predicted_regions)} box(es)")
        for i, box in enumerate(pred.predicted_regions, 1):
            print(f"  Box {i}: [x0={box[0]:.3f}, y0={box[1]:.3f}, x1={box[2]:.3f}, y1={box[3]:.3f}]")

    if args.save_heatmap:
        overlay.save(args.save_heatmap)
        print(f"\nVisual heatmap overlay saved to: {args.save_heatmap}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
