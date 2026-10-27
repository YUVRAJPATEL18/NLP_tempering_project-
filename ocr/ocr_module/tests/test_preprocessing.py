"""
Unit tests for image preprocessing, loading, CLAHE, resizing, and deskewing.
"""

import pytest
import numpy as np
import cv2

from ocr_module.preprocessing.image_loader import ImageLoader
from ocr_module.preprocessing.resize import resize_image
from ocr_module.preprocessing.grayscale import to_grayscale, to_bgr
from ocr_module.preprocessing.thresholding import apply_clahe, adaptive_threshold
from ocr_module.preprocessing.deskew import deskew_image
from ocr_module.exceptions import InvalidImageError


def test_image_loader_valid_numpy():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    loaded_img, metrics = ImageLoader.load_image(img)
    assert loaded_img.shape == (400, 600, 3)
    assert metrics["width"] == 600
    assert metrics["height"] == 400


def test_image_loader_empty_numpy():
    empty_img = np.array([])
    with pytest.raises(InvalidImageError):
        ImageLoader.load_image(empty_img)


def test_resize_upscale():
    small_img = np.zeros((200, 200, 3), dtype=np.uint8)
    resized = resize_image(small_img, min_dim=600, max_dim=2048)
    assert min(resized.shape[:2]) >= 600


def test_grayscale_and_bgr_conversion():
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(color_img)
    assert len(gray.shape) == 2
    bgr = to_bgr(gray)
    assert len(bgr.shape) == 3 and bgr.shape[2] == 3


def test_clahe_enhancement():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    enhanced = apply_clahe(img)
    assert enhanced.shape == img.shape


def test_deskew_horizontal():
    horizontal_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(horizontal_img, "TEST DOCUMENT PASSPORT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    deskewed = deskew_image(horizontal_img)
    assert deskewed.shape == horizontal_img.shape
