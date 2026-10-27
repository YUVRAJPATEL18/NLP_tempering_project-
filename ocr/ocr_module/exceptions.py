"""
Custom exceptions for IDGuard OCR module.
"""


class IDGuardError(Exception):
    """Base exception for all IDGuard errors."""

    pass


class InvalidImageError(IDGuardError):
    """Raised when an uploaded/provided image file is invalid, corrupt, or unsupported."""

    pass


class OCRError(IDGuardError):
    """Raised when an OCR engine operation fails."""

    pass


class EngineInitializationError(OCRError):
    """Raised when an OCR engine fails to initialize."""

    pass


class PostprocessingError(IDGuardError):
    """Raised during text postprocessing failures."""

    pass


class ValidationError(IDGuardError):
    """Raised when field validation fails critically."""

    pass


class EvaluationError(IDGuardError):
    """Raised during evaluation failures."""

    pass
