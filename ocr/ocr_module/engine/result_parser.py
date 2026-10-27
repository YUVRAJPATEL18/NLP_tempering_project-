"""
Parser for converting raw OCR engine outputs into structured OCRResult schemas.
Supports PaddleOCR v2, PaddleOCR v3.x, PaddleX, and custom dict/object outputs.
"""

import uuid
import logging
from typing import List, Any, Optional
import numpy as np
from ..schemas import OCRResult, OCRLine

logger = logging.getLogger(__name__)


class ResultParser:
    """Utility class to convert engine outputs into validated Pydantic schemas."""

    @staticmethod
    def parse_paddle_output(
        raw_output: Any,
        image_shape: tuple,
        processing_time_ms: float,
        document_id: Optional[str] = None,
        variant_name: str = "original"
    ) -> OCRResult:
        doc_id = document_id or str(uuid.uuid4())
        h, w = image_shape[:2]

        lines: List[OCRLine] = []
        confidences: List[float] = []
        raw_text_parts: List[str] = []

        if raw_output is None:
            return OCRResult(
                document_id=doc_id,
                raw_text="",
                lines=[],
                average_confidence=0.0,
                image_width=w,
                image_height=h,
                processing_time_ms=round(processing_time_ms, 2),
                preprocessing_variant=variant_name
            )

        # Handle generator output
        if not isinstance(raw_output, (list, dict, tuple)):
            try:
                raw_output = list(raw_output)
            except Exception:
                raw_output = [raw_output]

        line_idx = 1

        # Flatten list of predictions if wrapped
        items_to_process = []
        if isinstance(raw_output, list):
            for res in raw_output:
                if isinstance(res, list):
                    items_to_process.extend(res)
                else:
                    items_to_process.append(res)
        elif isinstance(raw_output, dict):
            items_to_process = [raw_output]

        for item in items_to_process:
            if not item:
                continue

            # Convert object with dict attributes or .json property
            item_dict = None
            if isinstance(item, dict):
                item_dict = item
            elif hasattr(item, "json"):
                try:
                    item_dict = item.json
                except Exception:
                    pass
            elif hasattr(item, "__dict__"):
                item_dict = item.__dict__

            if item_dict is not None:
                # Extract rec_text / texts
                res_dict = item_dict.get("res", item_dict)
                rec_texts = res_dict.get("rec_text", res_dict.get("texts", res_dict.get("rec_texts", [])))
                rec_scores = res_dict.get("rec_score", res_dict.get("scores", res_dict.get("rec_scores", [])))
                dt_polys = res_dict.get("dt_polys", res_dict.get("polygons", res_dict.get("boxes", res_dict.get("dt_boxes", []))))

                if isinstance(rec_texts, list) and len(rec_texts) > 0:
                    for i in range(len(rec_texts)):
                        text = str(rec_texts[i]).strip()
                        if not text:
                            continue
                        conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                        bbox_raw = dt_polys[i] if i < len(dt_polys) else [[0, 0], [w, 0], [w, h], [0, h]]

                        bbox: List[List[int]] = []
                        if isinstance(bbox_raw, (list, np.ndarray)):
                            for pt in bbox_raw:
                                bbox.append([int(round(float(pt[0]))), int(round(float(pt[1])))])

                        lines.append(
                            OCRLine(
                                text=text,
                                confidence=round(conf, 4),
                                bbox=bbox or [[0, 0], [w, 0], [w, h], [0, h]],
                                line_number=line_idx
                            )
                        )
                        confidences.append(conf)
                        raw_text_parts.append(text)
                        line_idx += 1
                    continue

                # Check if item_dict is a single prediction dict
                text = str(item_dict.get("rec_text", item_dict.get("text", ""))).strip()
                if text:
                    conf = float(item_dict.get("rec_score", item_dict.get("score", 1.0)))
                    bbox_raw = item_dict.get("dt_polys", item_dict.get("points", item_dict.get("bbox", [])))
                    bbox: List[List[int]] = []
                    for pt in bbox_raw:
                        bbox.append([int(round(float(pt[0]))), int(round(float(pt[1])))])

                    lines.append(
                        OCRLine(
                            text=text,
                            confidence=round(conf, 4),
                            bbox=bbox or [[0, 0], [w, 0], [w, h], [0, h]],
                            line_number=line_idx
                        )
                    )
                    confidences.append(conf)
                    raw_text_parts.append(text)
                    line_idx += 1
                    continue

            # Legacy PaddleOCR tuple format: [bbox, (text, conf)]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                bbox_raw, text_conf = item[0], item[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    text = str(text_conf[0]).strip()
                    conf = float(text_conf[1])
                else:
                    text = str(text_conf).strip()
                    conf = 1.0

                if text:
                    bbox: List[List[int]] = []
                    for pt in bbox_raw:
                        bbox.append([int(round(float(pt[0]))), int(round(float(pt[1])))])

                    lines.append(
                        OCRLine(
                            text=text,
                            confidence=round(conf, 4),
                            bbox=bbox,
                            line_number=line_idx
                        )
                    )
                    confidences.append(conf)
                    raw_text_parts.append(text)
                    line_idx += 1

        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
        full_raw_text = "\n".join(raw_text_parts)

        return OCRResult(
            document_id=doc_id,
            raw_text=full_raw_text,
            lines=lines,
            average_confidence=avg_conf,
            image_width=w,
            image_height=h,
            processing_time_ms=round(processing_time_ms, 2),
            preprocessing_variant=variant_name
        )
