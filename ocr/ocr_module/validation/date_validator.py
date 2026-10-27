"""
Date parsing, ISO 8601 normalization, and ambiguity detector.
"""

from typing import Tuple, Optional
import datetime
import re


class DateValidator:
    """Parses and validates document dates into ISO format (YYYY-MM-DD) with ambiguity tracking."""

    MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    @staticmethod
    def validate_and_normalize(date_str: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Parses date string into ISO YYYY-MM-DD format.

        Returns:
            (is_valid: bool, normalized_iso_date: str or None, warning_message: str or None)
        """
        if not date_str:
            return False, None, "Empty date string."

        clean_str = date_str.strip()

        # Pattern 1: ISO YYYY-MM-DD
        m1 = re.match(r"^(\d{4})[\-/\.](\d{1,2})[\-/\.](\d{1,2})$", clean_str)
        if m1:
            y, m, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
            try:
                dt = datetime.date(y, m, d)
                return True, dt.isoformat(), None
            except ValueError:
                return False, None, f"Invalid calendar date values: Y={y}, M={m}, D={d}"

        # Pattern 2: DD/MM/YYYY or MM/DD/YYYY
        m2 = re.match(r"^(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{4})$", clean_str)
        if m2:
            n1, n2, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            warning = None
            if n1 <= 12 and n2 <= 12 and n1 != n2:
                warning = f"Ambiguous date format '{clean_str}': could be day={n1}/month={n2} or month={n1}/day={n2}."

            # Default interpretation for ID cards in international standard: DD/MM/YYYY
            d, m = n1, n2
            try:
                dt = datetime.date(y, m, d)
                return True, dt.isoformat(), warning
            except ValueError:
                # Try MM/DD/YYYY if DD/MM/YYYY failed
                try:
                    dt = datetime.date(y, n1, n2)
                    return True, dt.isoformat(), warning
                except ValueError:
                    return False, None, f"Invalid date components in '{clean_str}'"

        # Pattern 3: DD MMM YYYY or MMM DD, YYYY (e.g., 15 JAN 2024, JAN 15, 2024)
        m3 = re.match(r"^(\d{1,2})\s+([a-zA-Z]{3,9})\s+(\d{4})$", clean_str)
        if m3:
            d, month_str, y = int(m3.group(1)), m3.group(2).lower()[:3], int(m3.group(3))
            if month_str in DateValidator.MONTH_MAP:
                m = DateValidator.MONTH_MAP[month_str]
                try:
                    dt = datetime.date(y, m, d)
                    return True, dt.isoformat(), None
                except ValueError:
                    pass

        m4 = re.match(r"^([a-zA-Z]{3,9})\s+(\d{1,2})[,\s]+(\d{4})$", clean_str)
        if m4:
            month_str, d, y = m4.group(1).lower()[:3], int(m4.group(2)), int(m4.group(3))
            if month_str in DateValidator.MONTH_MAP:
                m = DateValidator.MONTH_MAP[month_str]
                try:
                    dt = datetime.date(y, m, d)
                    return True, dt.isoformat(), None
                except ValueError:
                    pass

        return False, None, f"Could not parse date format '{clean_str}'"
