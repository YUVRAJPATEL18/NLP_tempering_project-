import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import PipelineError
from app.face import Face, OpenCVFaceEngine, decode_image, rotate_with_bounds, cosine_similarity
from app.main import create_app
from app.repository import SQLiteReferenceRepository
from app.service import VerificationService


def picture():
    output = io.BytesIO()
    Image.new('RGB', (80, 80), 'white').save(output, format='PNG')
    return output.getvalue()


class Engine:
    def __init__(self, mismatch=False, error=None):
        self.calls = 0
        self.mismatch, self.error = mismatch, error

    def extract(self, image):
        if self.error:
            raise PipelineError(self.error, 'test rejection')
        self.calls += 1
        embedding = np.array([0., 1.]) if self.mismatch and self.calls % 2 == 0 else np.array([1., 0.])
        return Face([10, 10, 60, 60], .99, image[10:60, 10:60], embedding)


def client(tmp_path, engine=None, limit=1024*1024):
    settings = replace(Settings(), database=tmp_path / 'test.sqlite', max_upload_bytes=limit)
    repo = SQLiteReferenceRepository(settings.database)
    repo.save_photo('person-1', picture())
    return TestClient(create_app(VerificationService(repo, engine or Engine(), settings), settings))


def post(api, payload=None, **fields):
    return api.post('/v1/verify-id-card', data={'reference_id': 'person-1', **fields},
                    files={'id_card': ('card.png', picture() if payload is None else payload, 'image/png')})


def test_match_and_images(tmp_path):
    response = post(client(tmp_path), include_images='true')
    assert response.status_code == 200
    body = response.json()
    assert body['matched'] and body['similarity'] == 1
    assert body['face']['bbox'] == [10, 10, 60, 60]
    assert body['images']['aligned_face'].startswith('data:image/png;base64,')
    assert response.headers['cache-control'] == 'no-store'


def test_mismatch(tmp_path):
    body = post(client(tmp_path, Engine(mismatch=True))).json()
    assert not body['matched'] and body['similarity'] == 0
    assert 'images' not in body


def test_unknown_record(tmp_path):
    assert post(client(tmp_path), reference_id='missing').status_code == 404


def test_invalid_and_oversized_images(tmp_path):
    api = client(tmp_path)
    assert post(api, b'not an image').status_code == 400
    assert post(api, b'').status_code == 400
    assert post(client(tmp_path, limit=10)).status_code == 413


def test_no_face_and_multiple_faces(tmp_path):
    for error in ['NO_FACE', 'MULTIPLE_FACES']:
        response = post(client(tmp_path, Engine(error=error)))
        assert response.status_code == 422
        assert response.json()['error']['code'] == error


def test_repository_parameterized_lookup(tmp_path):
    repo = SQLiteReferenceRepository(tmp_path / 'db.sqlite')
    repo.save_photo('valid', b'photo')
    assert repo.get_photo("' OR 1=1 --") is None
    assert repo.get_photo('valid') == b'photo'


def test_real_engine_missing_models(tmp_path):
    settings = replace(Settings(), detector=tmp_path/'absent', recognizer=tmp_path/'absent2')
    try:
        OpenCVFaceEngine(settings).extract(decode_image(picture(), settings))
    except PipelineError as error:
        assert error.code == 'MODELS_UNAVAILABLE' and error.status == 503
    else:
        raise AssertionError('Expected unavailable models')


def test_engine_detect_align_embed_contract():
    class Detector:
        def setInputSize(self, size):
            assert size == (80, 80)

        def detect(self, image):
            return None, np.array([[10, 10, 50, 50, 25, 25, 45, 25, 35, 35, 25, 45, 45, 45, .99]], dtype=np.float32)

    class Recognizer:
        def alignCrop(self, image, landmarks):
            assert len(landmarks) == 15
            return np.ones((112, 112, 3), dtype=np.uint8)

        def feature(self, aligned):
            assert aligned.shape == (112, 112, 3)
            return np.array([[1., 0.]], dtype=np.float32)

    engine = OpenCVFaceEngine(Settings())
    engine.detector, engine.recognizer = Detector(), Recognizer()
    face = engine.extract(np.zeros((80, 80, 3), dtype=np.uint8))
    assert face.bbox == [10, 10, 60, 60]
    assert face.aligned.shape == (112, 112, 3)
    assert face.embedding.tolist() == [1., 0.]


def test_invalid_reference_has_distinct_error(tmp_path):
    settings = replace(Settings(), database=tmp_path / 'broken.sqlite')
    repo = SQLiteReferenceRepository(settings.database)
    repo.save_photo('person-1', b'corrupted reference')
    api = TestClient(create_app(VerificationService(repo, Engine(), settings), settings))
    response = post(api)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'INVALID_REFERENCE'


@pytest.mark.parametrize('score, expected', [(0.59, False), (0.6, True), (0.61, True)])
def test_threshold_boundary(tmp_path, score, expected):
    class ScoredEngine(Engine):
        def extract(self, image):
            face = super().extract(image)
            if self.calls % 2 == 0:
                face.embedding = np.array([score, np.sqrt(1-score*score)])
            return face

    api = client(tmp_path, ScoredEngine())
    body = post(api).json()
    assert body['threshold'] == .6
    assert body['matched'] is expected


@pytest.mark.parametrize('angle', [30, 90, 180, 270])
def test_rotation_roundtrip(angle):
    import cv2
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    rotated, inverse = rotate_with_bounds(image, angle)
    forward = cv2.invertAffineTransform(inverse)
    corners = np.array([[[20., 15.], [70., 15.], [70., 60.], [20., 60.]]])
    transformed = cv2.transform(corners, forward)
    assert (transformed >= -1e-6).all()
    assert transformed[..., 0].max() <= rotated.shape[1]
    assert transformed[..., 1].max() <= rotated.shape[0]
    assert np.allclose(cv2.transform(transformed, inverse), corners)


def test_rotation_retry_restores_original_box():
    class Detector:
        calls = 0

        def setInputSize(self, size):
            self.size = size

        def detect(self, image):
            self.calls += 1
            if self.calls == 1:
                return None, None
            assert self.size == (80, 120)
            # Inverse of +90 rotation maps this box to [20, 10, 60, 50].
            return None, np.array([[10, 60, 40, 40, 20, 70, 40, 70, 30, 80, 20, 90, 40, 90, .99]], dtype=np.float32)

    class Recognizer:
        def alignCrop(self, image, face):
            assert image.shape[:2] == (120, 80)
            return np.ones((112, 112, 3), dtype=np.uint8)

        def feature(self, image):
            return np.array([[1., 0.]])

    engine = OpenCVFaceEngine(Settings())
    engine.detector, engine.recognizer = Detector(), Recognizer()
    face = engine.extract(np.zeros((80, 120, 3), dtype=np.uint8))
    assert face.rotation_degrees == 90
    assert np.allclose(face.bbox, [20, 10, 60, 50], atol=1)


def test_invalid_feature_dimensions_fail_closed():
    with pytest.raises(PipelineError, match='invalid features'):
        cosine_similarity(np.ones(128), np.ones(64))
