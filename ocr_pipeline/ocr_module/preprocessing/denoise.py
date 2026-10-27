"""
Image denoising and sharpening utilities.
"""

import cv2
import numpy as np


def gaussian_blur(img: np.ndarray, ksize: int = 3, sigma_x: float = 0.0) -> np.ndarray:
    """Applies Gaussian Blur for smoothing."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma_x)


def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies Median Blur to suppress salt-and-pepper noise."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, ksize)


def fast_denoise(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Applies Non-Local Means Denoising to preserve edge crispness."""
    if len(img.shape) == 2:
        return cv2.fastNlMeansDenoising(img, h=h)
    return cv2.fastNlMeansDenoisingColored(img, h=h, hColor=h)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """Applies unsharp masking kernel to crisp character edges."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)
