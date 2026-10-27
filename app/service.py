import base64
import cv2

from app.config import Settings
from app.errors import PipelineError
from app.face import FaceEngine, cosine_similarity, decode_image
from app.repository import ReferenceRepository


class VerificationService:
    def __init__(self, repository: ReferenceRepository, engine: FaceEngine, settings: Settings):
        self.repository, self.engine, self.settings = repository, engine, settings

    def verify(self, reference_id: str, card: bytes, include_images: bool = False) -> dict:
        image = decode_image(card, self.settings)
        reference = self.repository.get_photo(reference_id)
        if reference is None:
            raise PipelineError('REFERENCE_NOT_FOUND', 'Reference record does not exist', 404)
        face = self.engine.extract(image)
        try:
            stored_face = self.engine.extract(decode_image(reference, self.settings))
        except PipelineError as exc:
            if exc.status == 503:
                raise
            raise PipelineError('INVALID_REFERENCE', 'Stored reference photo needs replacement', 409) from exc
        score = cosine_similarity(face.embedding, stored_face.embedding)
        result = dict(reference_id=reference_id, matched=score >= self.settings.threshold,
                      similarity=score, threshold=self.settings.threshold,
                      model_version=getattr(self.engine, 'model_version', 'custom'),
                      pipeline_version=getattr(self.engine, 'pipeline_version', 'custom'),
                      face=dict(bbox=face.bbox, detection_confidence=face.detection_confidence,
                                rotation_degrees=face.rotation_degrees),
                      coordinate_system='exif-oriented image; [left, top, right, bottom), pixels')
        if include_images:
            annotated = image.copy()
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(annotated, (x1, y1), (x2-1, y2-1), (0, 255, 0), 2)
            result['images'] = dict(aligned_face=encode_png(face.aligned), annotated_card=encode_png(annotated))
        return result


def encode_png(image) -> str:
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise PipelineError('IMAGE_ENCODING_FAILED', 'Could not encode result image', 500)
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')
