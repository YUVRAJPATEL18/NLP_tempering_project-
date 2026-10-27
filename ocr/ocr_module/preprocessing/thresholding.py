"""
Image thresholding and contrast enhancement operations.
"""

import cv2
import numpy as np
from .grayscale import to_grayscale


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to balance lighting.
    Returns grayscale or 3-channel image matching input.
    """
    is_color = len(img.shape) == 3
    gray = to_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(gray)
    if is_color:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """Applies Otsu's binarization."""
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(
    img: np.ndarray,
    block_size: int = 11,
    c: int = 2,
    method: str = "gaussian"
) -> np.ndarray:
    """Applies Adaptive Thresholding (Gaussian or Mean)."""
    gray = to_grayscale(img)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    return cv2.adaptiveThreshold(
        gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c
    )
