"""
Tokenization utilities for words, lines, ID characters, dates, and MRZ lines.
"""

import re
from typing import List, Dict, Any


class TextTokenizer:
    """Lightweight text tokenizer tailored for OCR documents."""

    @staticmethod
    def word_tokenize(text: str) -> List[str]:
        """Splits text into words by whitespace and punctuation boundaries."""
        return [w for w in re.split(r"\s+", text) if w]

    @staticmethod
    def line_tokenize(text: str) -> List[str]:
        """Splits text into non-empty lines."""
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def character_tokenize_id(id_text: str) -> List[str]:
        """Splits an ID string into individual alphanumeric characters."""
        clean_id = re.sub(r"[^\w]", "", id_text)
        return list(clean_id)

    @staticmethod
    def tokenize_date(date_text: str) -> Dict[str, str]:
        """Extracts date components (day, month, year) from a date string."""
        match = re.search(r"(\d{1,4})[./\s-]([a-zA-Z0-9]{1,3}|\d{1,2})[./\s-](\d{1,4})", date_text)
        if match:
            p1, p2, p3 = match.group(1), match.group(2), match.group(3)
            if len(p1) == 4:
                return {"year": p1, "month": p2, "day": p3}
            elif len(p3) == 4:
                return {"day": p1, "month": p2, "year": p3}
        return {"raw": date_text}

    @staticmethod
    def tokenize_mrz(mrz_text: str) -> List[str]:
        """Extracts 2 or 3 MRZ lines (44 or 30 characters long)."""
        lines = [line.strip() for line in mrz_text.splitlines() if line.strip()]
        mrz_lines = [l for l in lines if ("<<" in l or len(l) >= 28)]
        return mrz_lines
