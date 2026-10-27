"""
Spatial line merger for combining broken adjacent OCR bounding boxes.
"""

from typing import List
from ..schemas import OCRLine


class LineMerger:
    """Merges fragmented OCR text lines based on spatial proximity."""

    @staticmethod
    def sort_lines_spatially(lines: List[OCRLine]) -> List[OCRLine]:
        """
        Sorts lines primarily by top Y position (vertical), then left X position (horizontal).
        """
        def get_sort_key(line: OCRLine):
            bbox = line.bbox
            if len(bbox) >= 4:
                min_y = min(pt[1] for pt in bbox)
                min_x = min(pt[0] for pt in bbox)
                return (min_y // 15, min_x)  # Group within vertical tolerance band
            return (0, 0)

        return sorted(lines, key=get_sort_key)

    @staticmethod
    def merge_adjacent_lines(lines: List[OCRLine], vertical_thresh: int = 12) -> List[OCRLine]:
        """
        Merges adjacent lines on nearly identical vertical alignment.
        """
        if not lines:
            return []

        sorted_lines = LineMerger.sort_lines_spatially(lines)
        merged: List[OCRLine] = []

        curr = sorted_lines[0]
        for next_line in sorted_lines[1:]:
            curr_y = min(pt[1] for pt in curr.bbox) if len(curr.bbox) >= 4 else 0
            next_y = min(pt[1] for pt in next_line.bbox) if len(next_line.bbox) >= 4 else 0

            if abs(curr_y - next_y) <= vertical_thresh:
                # Merge into a single line
                new_text = f"{curr.text} {next_line.text}".strip()
                new_conf = round((curr.confidence + next_line.confidence) / 2.0, 4)
                # Combine bounding box points
                combined_bbox = curr.bbox + next_line.bbox
                min_x = min(p[0] for p in combined_bbox)
                min_y = min(p[1] for p in combined_bbox)
                max_x = max(p[0] for p in combined_bbox)
                max_y = max(p[1] for p in combined_bbox)
                new_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

                curr = OCRLine(
                    text=new_text,
                    confidence=new_conf,
                    bbox=new_bbox,
                    line_number=curr.line_number
                )
            else:
                merged.append(curr)
                curr = next_line

        merged.append(curr)
        # Renumber lines
        for idx, line in enumerate(merged, start=1):
            line.line_number = idx

        return merged
