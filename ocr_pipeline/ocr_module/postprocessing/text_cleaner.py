"""
Text cleaning and normalization utility for raw OCR outputs.
"""

import re
import unicodedata


class TextCleaner:
    """Cleans OCR noise while preserving essential identity document characters."""

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Removes ASCII/Unicode control characters except newlines and tabs."""
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapses multiple spaces/tabs into a single space while keeping line structures clean."""
        lines = text.splitlines()
        cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        return "\n".join([line for line in cleaned_lines if line])

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Executes non-destructive cleaning on OCR output.
        Preserves hyphens, slashes, dots, commas, colons, letters, and numbers.
        """
        if not raw_text:
            return ""

        text = TextCleaner.remove_control_characters(raw_text)
        text = TextCleaner.normalize_whitespace(text)
        return text
