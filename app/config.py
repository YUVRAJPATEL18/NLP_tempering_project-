import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database: Path = Path(os.getenv('FACE_DATABASE', 'data/faces.sqlite3'))
    detector: Path = Path(os.getenv('FACE_DETECTOR', 'models/face_detection_yunet_2023mar.onnx'))
    recognizer: Path = Path(os.getenv('FACE_RECOGNIZER', 'models/face_recognition_sface_2021dec.onnx'))
    threshold: float = float(os.getenv('FACE_MATCH_THRESHOLD', '0.6'))
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 20_000_000

    def __post_init__(self):
        if not -1 <= self.threshold <= 1:
            raise ValueError('FACE_MATCH_THRESHOLD must be between -1 and 1')
