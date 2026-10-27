"""
Unicode, case, whitespace, and punctuation text normalization utility.
"""

import re
import unicodedata
from typing import Dict, Any


class TextNormalizer:
    """Provides comprehensive text normalization layers for string comparison and matching."""

    @staticmethod
    def unicode_normalize(text: str, form: str = "NFC") -> str:
        """Applies Unicode normalization (NFC or NFKD)."""
        return unicodedata.normalize(form, text)

    @staticmethod
    def case_normalize(text: str) -> str:
        """Converts string to lower case for case-insensitive matching."""
        return text.lower()

    @staticmethod
    def whitespace_normalize(text: str) -> str:
        """Removes extra spaces, leading/trailing whitespace, and normalizes tabs/newlines."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def punctuation_normalize(text: str) -> str:
        """Normalizes quotes, hyphens, and em-dashes."""
        # Standardize quotes and dashes
        text = re.sub(r"[‘’´`]", "'", text)
        text = re.sub(r"[“”«»]", '"', text)
        text = re.sub(r"[—–−]", "-", text)
        return text

    @staticmethod
    def generate_comparison_versions(text: str) -> Dict[str, str]:
        """
        Returns dictionary of text variants for exact, case-insensitive,
        whitespace-insensitive, and punctuation-free comparisons.
        """
        u_norm = TextNormalizer.unicode_normalize(text)
        p_norm = TextNormalizer.punctuation_normalize(u_norm)
        w_norm = TextNormalizer.whitespace_normalize(p_norm)
        c_norm = TextNormalizer.case_normalize(w_norm)

        # Remove all punctuation for fuzzy/relaxed matching
        no_punct = re.sub(r"[^\w\s]", "", c_norm)
        no_punct_compact = re.sub(r"\s+", "", no_punct)

        return {
            "original_text": text,
            "unicode_text": u_norm,
            "cleaned_text": w_norm,
            "normalized_text": c_norm,
            "comparison_text": no_punct_compact
        }
