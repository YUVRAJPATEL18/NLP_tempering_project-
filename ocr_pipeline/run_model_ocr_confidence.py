"""Run PaddleOCR only and report mean and low-confidence text regions."""

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Tuple


def predictions_to_rows(output: Any) -> List[Tuple[str, float]]:
    """Read modern PaddleOCR dict output or the legacy [box, (text, score)] form."""
    rows: List[Tuple[str, float]] = []

    def walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            value = value.get("res", value)
            texts = value.get("rec_texts", value.get("rec_text", value.get("texts")))
            scores = value.get("rec_scores", value.get("rec_score", value.get("scores")))
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                if scores is None:
                    scores = []
                elif isinstance(scores, (int, float)):
                    scores = [scores]
                for index, text in enumerate(texts):
                    if index < len(scores) and text:
                        rows.append((str(text), float(scores[index])))
                return
            return
        if isinstance(value, (list, tuple)):
            # Legacy PaddleOCR line: [polygon, (recognized text, confidence)]
            if len(value) >= 2 and isinstance(value[1], (list, tuple)) and len(value[1]) >= 2:
                try:
                    rows.append((str(value[1][0]), float(value[1][1])))
                    return
                except (TypeError, ValueError):
                    pass
            for child in value:
                walk(child)

    walk(output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument("--low-threshold", type=float, default=0.80,
                        help="Highlight regions below this confidence (default: 0.80)")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    from paddleocr import PaddleOCR

    # PaddleOCR 3.x removed use_gpu/use_angle_cls constructor arguments;
    # this simple initialization also remains accepted by the installed version.
    engine = PaddleOCR(lang="en")
    try:
        output = engine.ocr(str(args.image), cls=True)
    except TypeError:
        output = engine.ocr(str(args.image))

    rows = predictions_to_rows(output)
    if not rows:
        print("PaddleOCR returned no recognized text regions.")
        return

    mean = sum(score for _, score in rows) / len(rows)
    print(f"Mean OCR confidence: {mean:.3f} ({mean:.1%}) across {len(rows)} regions")
    print(f"Low-confidence threshold: {args.low_threshold:.3f} ({args.low_threshold:.1%})")
    low_rows = [(text, score) for text, score in rows if score < args.low_threshold]
    if low_rows:
        print("Regions to review:")
        for text, score in low_rows:
            print(f"  {score:.3f} ({score:.1%})  {text}")
    else:
        print("No regions are below the threshold.")


if __name__ == "__main__":
    main()
