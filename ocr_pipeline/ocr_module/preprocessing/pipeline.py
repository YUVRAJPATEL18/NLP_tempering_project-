"""
Preprocessing pipeline generating multiple image variants and transparent candidate scoring.
"""

from enum import Enum
from typing import Dict, Any, Union, Optional, Tuple
import os
import cv2
import numpy as np

from .image_loader import ImageLoader
from .resize import resize_image
from .grayscale import to_grayscale, to_bgr
from .denoise import fast_denoise, sharpen_image, gaussian_blur
from .thresholding import apply_clahe, otsu_threshold, adaptive_threshold
from .deskew import deskew_image
from .perspective import correct_perspective
from .illumination import remove_shadows, remove_borders
from ..config import settings


class PreprocessingVariant(str, Enum):
    ORIGINAL = "original"
    GRAYSCALE = "grayscale"
    DENOISED = "denoised"
    ENHANCED = "enhanced"
    THRESHOLDED = "thresholded"
    DESKEWED = "deskewed"
    PERSPECTIVE = "perspective-corrected"


class PreprocessingPipeline:
    """
    Modular preprocessing pipeline for document images.
    """

    def __init__(self, save_intermediate: bool = settings.save_intermediate_images):
        self.save_intermediate = save_intermediate

    def process_single(
        self,
        image_source: Union[str, bytes, np.ndarray],
        variant: Union[str, PreprocessingVariant] = PreprocessingVariant.ORIGINAL
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes image source for a specific variant.
        Returns (processed_bgr_image, quality_metrics).
        """
        img, metrics = ImageLoader.load_image(image_source)
        img = resize_image(img)
        img = remove_borders(img)

        variant_str = variant.value if isinstance(variant, PreprocessingVariant) else str(variant)

        if variant_str == PreprocessingVariant.ORIGINAL.value:
            processed = img
        elif variant_str == PreprocessingVariant.GRAYSCALE.value:
            processed = to_bgr(to_grayscale(img))
        elif variant_str == PreprocessingVariant.DENOISED.value:
            denoised = fast_denoise(img)
            processed = sharpen_image(denoised)
        elif variant_str == PreprocessingVariant.ENHANCED.value:
            clahe_img = apply_clahe(img, clip_limit=2.5)
            shadow_removed = remove_shadows(clahe_img)
            processed = sharpen_image(shadow_removed)
        elif variant_str == PreprocessingVariant.THRESHOLDED.value:
            gray = to_grayscale(img)
            thresh = adaptive_threshold(gray)
            processed = to_bgr(thresh)
        elif variant_str == PreprocessingVariant.DESKEWED.value:
            processed = deskew_image(img)
        elif variant_str == PreprocessingVariant.PERSPECTIVE.value:
            processed = correct_perspective(img)
        else:
            processed = img

        processed = to_bgr(processed)

        if self.save_intermediate and isinstance(image_source, str) and os.path.exists(image_source):
            out_dir = os.path.join(os.path.dirname(image_source), "preprocessed_out")
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.basename(image_source)
            out_path = os.path.join(out_dir, f"{variant_str}_{base_name}")
            cv2.imwrite(out_path, processed)

        metrics["applied_variant"] = variant_str
        return processed, metrics

    def generate_all_variants(
        self,
        image_source: Union[str, bytes, np.ndarray]
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Generates dictionary mapping variant names to (bgr_image, metrics).
        """
        results = {}
        for var in PreprocessingVariant:
            results[var.value] = self.process_single(image_source, variant=var)
        return results
