"""
Document deskewing utility based on contour bounding box orientation and Hough lines.
"""

import cv2
import numpy as np
from .grayscale import to_grayscale


def calculate_skew_angle(img: np.ndarray) -> float:
    """
    Estimates document rotation/skew angle in degrees [-45, 45].
    """
    gray = to_grayscale(img)
    # Invert image: text becomes white, background black
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilate text lines to form consolidated horizontal blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours of horizontal text lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore tiny noise contours
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]

        # OpenCV minAreaRect angle normalization
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    # Median angle is robust against outliers
    return float(np.median(angles))


def deskew_image(img: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
    """
    Rotates the image to align text horizontally if skew angle exceeds 0.5 degrees.
    """
    angle = calculate_skew_angle(img)
    if abs(angle) <= 0.5 or abs(angle) > max_angle:
        return img.copy()

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Border value white background for document images
    border_val = (255, 255, 255) if len(img.shape) == 3 else 255
    rotated = cv2.warpAffine(
        img, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val
    )
    return rotated
