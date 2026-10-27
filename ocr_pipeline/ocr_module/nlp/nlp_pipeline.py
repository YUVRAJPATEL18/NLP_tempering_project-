"""
Complete NLP pipeline orchestrator.
"""

from typing import Dict, Any, List
from ..schemas import OCRResult, CleanedTextResult, FieldExtractionValue
from .normalization import TextNormalizer
from .tokenization import TextTokenizer
from .field_extraction import FieldExtractor
from .fuzzy_matching import FuzzyMatcher


class NLPPipeline:
    """Orchestrates NLP normalization, tokenization, field extraction, and fuzzy matching."""

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer()
        self.field_extractor = FieldExtractor()

    def process(
        self,
        raw_text: str,
        cleaned_text: str,
        ocr_result: OCRResult
    ) -> Dict[str, Any]:
        """
        Executes complete NLP pipeline on OCR text outputs.
        """
        norm_dict = self.normalizer.generate_comparison_versions(cleaned_text or raw_text)
        tokens = self.tokenizer.word_tokenize(cleaned_text or raw_text)
        char_tokens = self.tokenizer.character_tokenize_id(cleaned_text or raw_text)

        # Extract structured fields
        fields = self.field_extractor.extract_fields(ocr_result)

        warnings: List[str] = []
        if not fields:
            warnings.append("No document fields could be automatically extracted from text.")

        return {
            "original_text": raw_text,
            "cleaned_text": cleaned_text,
            "normalized_text": norm_dict["normalized_text"],
            "tokens": tokens,
            "character_tokens": char_tokens[:50],  # Truncate character tokens list for neatness
            "comparison_text": norm_dict["comparison_text"],
            "fields": fields,
            "warnings": warnings
        }
