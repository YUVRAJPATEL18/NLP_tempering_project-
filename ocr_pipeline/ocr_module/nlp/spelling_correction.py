"""
Context-aware OCR confusion map corrector.
"""

from typing import Tuple, List, Dict, Any, Optional
import re


class ContextualOCRCorrector:
    """
    Applies field-aware OCR error corrections based on character confusion maps.
    Enforces strict context rules (numeric fields vs alphabetic names vs dates).
    """

    NUMERIC_CONFUSION = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', 'i': '1',
        'S': '5', 's': '5',
        'B': '8',
        'G': '6',
        'Z': '2'
    }

    ALPHA_CONFUSION = {
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '8': 'B',
        '6': 'G',
        '2': 'Z'
    }

    @staticmethod
    def correct_numeric_field(text: str) -> Tuple[str, List[str]]:
        """Applies character substitutions for strictly numeric fields (e.g., pure ID digits)."""
        reasons = []
        chars = []
        for ch in text:
            if ch in ContextualOCRCorrector.NUMERIC_CONFUSION:
                replaced = ContextualOCRCorrector.NUMERIC_CONFUSION[ch]
                reasons.append(f"Substituted '{ch}' -> '{replaced}' (numeric context)")
                chars.append(replaced)
            else:
                chars.append(ch)
        return "".join(chars), reasons

    @staticmethod
    def correct_name_field(text: str) -> Tuple[str, List[str]]:
        """
        Applies character substitutions for personal name fields.
        Preserves 'O', 'I', 'S', 'B', 'G'. Replaces digit noise (e.g., 'PAT3L' or 'YUVRA0').
        """
        reasons = []
        chars = []
        for ch in text:
            if ch in ContextualOCRCorrector.ALPHA_CONFUSION:
                replaced = ContextualOCRCorrector.ALPHA_CONFUSION[ch]
                reasons.append(f"Substituted digit noise '{ch}' -> '{replaced}' (name context)")
                chars.append(replaced)
            else:
                chars.append(ch)
        return "".join(chars), reasons

    @staticmethod
    def correct_passport_number(text: str) -> Tuple[str, List[str]]:
        """
        Passport numbers generally start with 1-2 alpha characters followed by 6-7 digits.
        Example: A1234567 or J123456.
        """
        text_clean = text.replace(" ", "").upper()
        if not text_clean:
            return text, []

        reasons = []
        corrected_chars = []

        for idx, ch in enumerate(text_clean):
            if idx < 1:  # First character is usually letter
                if ch in ContextualOCRCorrector.ALPHA_CONFUSION:
                    rep = ContextualOCRCorrector.ALPHA_CONFUSION[ch]
                    reasons.append(f"Passport prefix pos {idx}: '{ch}' -> '{rep}'")
                    corrected_chars.append(rep)
                else:
                    corrected_chars.append(ch)
            else:  # Trailing positions are usually numeric
                if ch in ContextualOCRCorrector.NUMERIC_CONFUSION:
                    rep = ContextualOCRCorrector.NUMERIC_CONFUSION[ch]
                    reasons.append(f"Passport digit pos {idx}: '{ch}' -> '{rep}'")
                    corrected_chars.append(rep)
                else:
                    corrected_chars.append(ch)

        return "".join(corrected_chars), reasons

    @staticmethod
    def correct_date_separators(text: str) -> str:
        """Standardizes date separators like comma, slash, dash, dot."""
        # Replace misread separators like colons or spaces between digits
        t = re.sub(r"(\d{2})[ :]+(\d{2})[ :]+(\d{4})", r"\1/\2/\3", text)
        return t
