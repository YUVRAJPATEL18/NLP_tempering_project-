"""
RapidFuzz fuzzy string matching for non-sensitive identity document fields
and abstract EmbeddingMatcher stub interface for future semantic vector models.
"""

from typing import Dict, Any, Optional
from rapidfuzz import fuzz, process


class FuzzyMatcher:
    """Fuzzy matching utility for non-sensitive strings (colleges, courses, addresses)."""

    @staticmethod
    def match_fields(
        field_name: str,
        query: str,
        choices: list,
        threshold: float = 80.0
    ) -> Dict[str, Any]:
        """
        Matches extracted text query against reference choices list using RapidFuzz.
        Returns detailed match metrics dictionary.
        """
        if not query or not choices:
            return {
                "field_name": field_name,
                "query": query,
                "exact_match": False,
                "normalized_match": False,
                "fuzzy_score": 0.0,
                "best_match": None,
                "match_status": "no_match",
                "reason": "Empty query or empty choice database."
            }

        clean_query = query.strip()
        query_lower = clean_query.lower()

        # Check exact & case-insensitive match
        for choice in choices:
            clean_choice = str(choice).strip()
            if clean_query == clean_choice:
                return {
                    "field_name": field_name,
                    "query": query,
                    "exact_match": True,
                    "normalized_match": True,
                    "fuzzy_score": 100.0,
                    "best_match": clean_choice,
                    "match_status": "exact_match",
                    "reason": "Exact character string match."
                }
            if query_lower == clean_choice.lower():
                return {
                    "field_name": field_name,
                    "query": query,
                    "exact_match": False,
                    "normalized_match": True,
                    "fuzzy_score": 100.0,
                    "best_match": clean_choice,
                    "match_status": "normalized_match",
                    "reason": "Case-insensitive normalized match."
                }

        # RapidFuzz ratio
        result = process.extractOne(clean_query, choices, scorer=fuzz.WRatio)
        if result:
            best_match, score, index = result
            if score >= threshold:
                return {
                    "field_name": field_name,
                    "query": query,
                    "exact_match": False,
                    "normalized_match": False,
                    "fuzzy_score": round(float(score), 2),
                    "best_match": best_match,
                    "match_status": "fuzzy_match",
                    "reason": f"Fuzzy match score {score:.1f}% exceeds threshold {threshold}%."
                }
            else:
                return {
                    "field_name": field_name,
                    "query": query,
                    "exact_match": False,
                    "normalized_match": False,
                    "fuzzy_score": round(float(score), 2),
                    "best_match": best_match,
                    "match_status": "below_threshold",
                    "reason": f"Highest fuzzy match score {score:.1f}% below threshold {threshold}%."
                }

        return {
            "field_name": field_name,
            "query": query,
            "exact_match": False,
            "normalized_match": False,
            "fuzzy_score": 0.0,
            "best_match": None,
            "match_status": "no_match",
            "reason": "No match found."
        }


class EmbeddingMatcher:
    """
    Abstract interface stub for future semantic vector embedding matchers (BERT, SBERT).
    Does not require heavy transformer dependencies during baseline execution.
    """

    def embed(self, text: str) -> Optional[list]:
        """Placeholder for text embedding vector calculation."""
        return None

    def similarity(self, text_a: str, text_b: str) -> float:
        """Placeholder for cosine similarity calculation."""
        if text_a.strip().lower() == text_b.strip().lower():
            return 1.0
        return 0.0
