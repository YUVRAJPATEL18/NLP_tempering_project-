import io
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.errors import PipelineError


@dataclass
class Face:
    bbox: list[int]
    detection_confidence: float
    aligned: np.ndarray
    embedding: np.ndarray
    rotation_degrees: int = 0


class FaceEngine(Protocol):
    def extract(self, image: np.ndarray) -> Face: ...


def decode_image(payload: bytes, settings: Settings) -> np.ndarray:
    if not payload:
        raise PipelineError('EMPTY_IMAGE', 'Image is empty', 400)
    if len(payload) > settings.max_upload_bytes:
        raise PipelineError('IMAGE_TOO_LARGE', 'Image exceeds the upload limit', 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in ('JPEG', 'PNG'):
                    raise PipelineError('UNSUPPORTED_IMAGE', 'Only JPEG and PNG are supported', 415)
                if source.width * source.height > settings.max_pixels:
                    raise PipelineError('IMAGE_TOO_LARGE', 'Image exceeds the pixel limit', 413)
                rgb = np.array(ImageOps.exif_transpose(source).convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PipelineError('IMAGE_TOO_LARGE', 'Image exceeds the pixel limit', 413)
    except (UnidentifiedImageError, OSError, ValueError):
        raise PipelineError('INVALID_IMAGE', 'Image cannot be decoded', 400)


class OpenCVFaceEngine:
    model_version = 'yunet-2023mar+sface-2021dec'
    pipeline_version = 'orientation-retry-v2'

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = Lock()
        self.detector = self.recognizer = None

    def extract(self, image: np.ndarray) -> Face:
        # OpenCV model instances have mutable input/output buffers.
        with self.lock:
            if self.detector is None:
                if not self.settings.detector.is_file() or not self.settings.recognizer.is_file():
                    raise PipelineError('MODELS_UNAVAILABLE', 'Run scripts/download_models.py first', 503)
                try:
                    self.detector = cv2.FaceDetectorYN.create(str(self.settings.detector), '', (320, 320), 0.9)
                    self.recognizer = cv2.FaceRecognizerSF.create(str(self.settings.recognizer), '')
                except cv2.error:
                    self.detector = self.recognizer = None
                    raise PipelineError('MODELS_UNAVAILABLE', 'Face models could not be loaded', 503)
            # Retry orientation only after a detection miss. Never choose an
            # orientation based on its match score against a database identity.
            working = image
            inverse = None
            angle = 0
            faces = None
            for angle in (0, 90, 180, 270, 30, 330, 60, 300, 120, 240, 150, 210):
                if angle:
                    working, inverse = rotate_with_bounds(image, angle)
                self.detector.setInputSize((working.shape[1], working.shape[0]))
                _, faces = self.detector.detect(working)
                if faces is not None and len(faces):
                    break
            if faces is None or len(faces) == 0:
                raise PipelineError('NO_FACE', 'No face detected after orientation retries; provide a clearer image')
            if len(faces) != 1:
                raise PipelineError('MULTIPLE_FACES', 'Expected exactly one face; crop the card portrait region')
            face = faces[0]
            x, y, w, h = face[:4]
            if min(w, h) < 32:
                raise PipelineError('FACE_TOO_SMALL', 'Face must be at least 32 pixels on each side')
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
            if inverse is not None:
                corners = cv2.transform(corners[None], inverse)[0]
            height, width = image.shape[:2]
            bbox = [max(0, int(np.floor(corners[:, 0].min()))),
                    max(0, int(np.floor(corners[:, 1].min()))),
                    min(width, int(np.ceil(corners[:, 0].max()))),
                    min(height, int(np.ceil(corners[:, 1].max())))]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise PipelineError('INVALID_FACE', 'Detected face lies outside the input image')
            aligned = self.recognizer.alignCrop(working, face)
            embedding = self.recognizer.feature(aligned).copy().reshape(-1)
            return Face(bbox, float(face[-1]), aligned.copy(), embedding, angle)


def rotate_with_bounds(image: np.ndarray, degrees: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping, returning the map back to input pixel coordinates."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    corners = np.array([[[0, 0], [width, 0], [width, height], [0, height]]], dtype=np.float64)
    transformed = cv2.transform(corners, matrix)[0]
    minimum = transformed.min(axis=0)
    extent = np.ceil(transformed.max(axis=0)-minimum-1e-8).astype(int)
    matrix[:, 2] -= minimum
    rotated = cv2.warpAffine(image, matrix, tuple(int(v) for v in extent),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated, cv2.invertAffineTransform(matrix)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 1 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise PipelineError('INVALID_EMBEDDING', 'Face feature extraction returned invalid features', 503)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0 or not np.isfinite(denominator):
        raise PipelineError('INVALID_EMBEDDING', 'Face feature extraction failed', 503)
    score = float(np.dot(first, second) / denominator)
    if not np.isfinite(score):
        raise PipelineError('INVALID_EMBEDDING', 'Face feature extraction failed', 503)
    return float(np.clip(score, -1, 1))
