"""
Base validator interface and generic string/email/phone field validators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import re


class BaseValidator(ABC):
    """Abstract base class for field validators."""

    @abstractmethod
    def validate(self, value: str) -> Tuple[str, Optional[str]]:
        """Returns (validation_status: 'valid'|'invalid'|'warning', message: str or None)."""
        pass


class EmailValidator(BaseValidator):
    EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    def validate(self, value: str) -> Tuple[str, Optional[str]]:
        if not value:
            return "unknown", None
        if re.match(self.EMAIL_REGEX, value.strip()):
            return "valid", None
        return "invalid", "Invalid email format."


class PhoneValidator(BaseValidator):
    PHONE_REGEX = r"^\+?[0-9\s\-]{7,15}$"

    def validate(self, value: str) -> Tuple[str, Optional[str]]:
        if not value:
            return "unknown", None
        if re.match(self.PHONE_REGEX, value.strip()):
            return "valid", None
        return "invalid", "Invalid phone number format."
