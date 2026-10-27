"""
Image resizing utility for OCR optimization.
"""

import cv2
import numpy as np
from ..config import settings


def resize_image(
    img: np.ndarray,
    min_dim: int = settings.min_image_dimension,
    max_dim: int = settings.max_image_dimension
) -> np.ndarray:
    """
    Resizes image to ensure dimensions stay within target bounds [min_dim, max_dim]
    while strictly maintaining aspect ratio.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    short_side = min(h, w)

    # Upscale low-res images
    if short_side < min_dim:
        scale = min_dim / float(short_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Downscale overly large images
    if long_side > max_dim:
        scale = max_dim / float(long_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img.copy()
