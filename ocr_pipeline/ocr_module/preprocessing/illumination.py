"""
Shadow / uneven illumination correction and border removal utilities.
"""

import cv2
import numpy as np
from .grayscale import to_grayscale


def remove_shadows(img: np.ndarray) -> np.ndarray:
    """
    Normalizes uneven background illumination using morphological background estimation.
    """
    is_color = len(img.shape) == 3
    gray = to_grayscale(img)

    # Estimate background illumination using large morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Divide image by background estimate to normalize lighting
    normalized = cv2.divide(gray, background, scale=255)

    if is_color:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized


def remove_borders(img: np.ndarray, margin_percent: float = 0.02) -> np.ndarray:
    """
    Crops dark scanner/camera borders from edges.
    """
    h, w = img.shape[:2]
    crop_h = int(h * margin_percent)
    crop_w = int(w * margin_percent)

    if crop_h > 0 and crop_w > 0 and (h - 2 * crop_h > 50) and (w - 2 * crop_w > 50):
        return img[crop_h:h - crop_h, crop_w:w - crop_w].copy()
    return img.copy()
