"""
Grayscale conversion utility.
"""

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts a BGR or RGB image to single-channel grayscale."""
    if len(img.shape) == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensures image is 3-channel BGR format."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()
