"""
Image loader, corrupt image detector, and quality metrics utility.
"""

import os
from typing import Tuple, Union, Dict, Any
import cv2
import numpy as np
from PIL import Image

from ..exceptions import InvalidImageError
from ..config import settings


class ImageLoader:
    """Handles secure loading, integrity verification, and quality analysis of document images."""

    @staticmethod
    def load_image(source: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads an image from path, raw bytes, or numpy array.
        Returns (bgr_image_array, metadata_dict).
        """
        if isinstance(source, np.ndarray):
            if source.size == 0:
                raise InvalidImageError("Provided numpy array is empty.")
            img = source.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            metrics = ImageLoader.evaluate_quality(img)
            return img, metrics

        if isinstance(source, bytes):
            if not source:
                raise InvalidImageError("Provided image bytes are empty.")
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError("Failed to decode image from bytes. File may be corrupt.")
            metrics = ImageLoader.evaluate_quality(img)
            return img, metrics

        if isinstance(source, str):
            if not os.path.exists(source):
                raise InvalidImageError(f"Image path does not exist: {source}")
            ext = os.path.splitext(source)[1].lower()
            if ext not in settings.allowed_extensions:
                raise InvalidImageError(
                    f"Unsupported image extension '{ext}'. Allowed: {settings.allowed_extensions}"
                )

            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                # Try loading via Pillow as fallback for obscure WEBP/PNG variants
                try:
                    pil_img = Image.open(source)
                    pil_img.verify()  # Check corruption
                    pil_img = Image.open(source).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise InvalidImageError(f"Corrupt or unreadable image file '{source}': {str(e)}")

            metrics = ImageLoader.evaluate_quality(img)
            metrics["file_path"] = source
            return img, metrics

        raise InvalidImageError(f"Unsupported image source type: {type(source)}")

    @staticmethod
    def evaluate_quality(img: np.ndarray) -> Dict[str, Any]:
        """
        Calculates image quality metrics: blur score (Laplacian variance),
        brightness, and contrast.
        """
        if img is None or img.size == 0 or len(img.shape) not in (2, 3):
            raise InvalidImageError("Image is empty or has an unsupported shape.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        dark_fraction = float(np.mean(gray <= 20))
        bright_fraction = float(np.mean(gray >= 250))
        # Bright, low-saturation pixels are a cheap glare proxy. It is not a
        # definitive glare detector and can also fire on white document stock.
        if len(img.shape) == 3 and img.shape[2] >= 3:
            hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
            glare_fraction = float(np.mean((gray >= 245) & (hsv[:, :, 1] <= 35)))
        else:
            glare_fraction = bright_fraction

        is_blurry = blur_score < settings.quality_warning_blur_score
        is_dark = brightness < settings.quality_warning_brightness
        is_low_contrast = contrast < 35.0

        return {
            "height": img.shape[0],
            "width": img.shape[1],
            "channels": img.shape[2] if len(img.shape) == 3 else 1,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "dark_fraction": round(dark_fraction, 4),
            "bright_fraction": round(bright_fraction, 4),
            "glare_fraction": round(glare_fraction, 4),
            "is_blurry": is_blurry,
            "is_dark": is_dark,
            "is_low_contrast": is_low_contrast
        }
