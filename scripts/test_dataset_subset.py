"""Exercise the real HTTP endpoint using downloaded MIDV captures and fixed labels."""
import base64
import argparse
import csv
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import sys
import cv2
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.config import Settings
from app.face import OpenCVFaceEngine, decode_image
from app.main import create_app
from app.repository import SQLiteReferenceRepository
from app.service import VerificationService
from app.errors import PipelineError

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'datasets' / 'midv2020_subset'


def image_path(capture, identity):
    kind, number = identity.split(':')
    return DIRECTORY / f'dataset__{capture}__images__{kind}__{number}.jpg'


def annotated_portrait(capture, identity, settings):
    """Ground-truth-assisted control, deliberately separate from full-card testing."""
    kind, number = identity.split(':')
    annotations = json.loads((DIRECTORY / f'dataset__{capture}__annotations__{kind}.json').read_text(encoding='utf-8'))
    metadata = next(row for row in annotations['_via_img_metadata'].values() if row['filename'] == number+'.jpg')
    field = 'photo' if capture == 'templates' else 'face'
    region = next(row['shape_attributes'] for row in metadata['regions'] if row['region_attributes'].get('field_name') == field)
    image = decode_image(image_path(capture, identity).read_bytes(), settings)
    x, y, w, h = (region[key] for key in ('x', 'y', 'width', 'height'))
    margin = .1 if capture == 'templates' else .25
    left, top = max(0, int(x-w*margin)), max(0, int(y-h*margin))
    right, bottom = min(image.shape[1], int(x+w*(1+margin))), min(image.shape[0], int(y+h*(1+margin)))
    success, encoded = cv2.imencode('.png', image[top:bottom, left:right])
    if not success:
        raise RuntimeError('Could not prepare annotated reference portrait')
    return encoded.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-images', action='store_true', help='Save reference crops and annotated response images')
    args = parser.parse_args()
    report_dir = DIRECTORY / 'results'
    report_dir.mkdir(exist_ok=True)
    settings = replace(Settings(), database=DIRECTORY / 'test_references.sqlite3',
                       detector=ROOT / 'models' / 'face_detection_yunet_2023mar.onnx',
                       recognizer=ROOT / 'models' / 'face_recognition_sface_2021dec.onnx')
    repository = SQLiteReferenceRepository(settings.database)
    engine = OpenCVFaceEngine(settings)
    identities = [f'{kind}:{number:02d}' for kind in ('alb_id', 'fin_id') for number in range(3)]
    enrollment = []
    for identity in identities:
        payload = annotated_portrait('templates', identity, settings)
        if args.save_images:
            (report_dir / f'reference_{identity.replace(":", "_")}.png').write_bytes(payload)
        try:
            face = engine.extract(decode_image(payload, settings))
            repository.save_photo(identity, payload)
            enrollment.append(dict(identity=identity, status='ok', bbox=face.bbox))
        except PipelineError as exc:
            enrollment.append(dict(identity=identity, status='failed', error=exc.code))
        print(f'Enrollment {identity}: {enrollment[-1]["status"]}', flush=True)
    # Failed enrollments are explicit; never silently drop inconvenient identities.
    (report_dir / 'enrollment.json').write_text(json.dumps(enrollment, indent=2), encoding='utf-8')
    results = []
    with TestClient(create_app(VerificationService(repository, engine, settings), settings)) as client:
        for mode, capture in [('original_card', 'scan_upright'), ('original_card', 'scan_rotated'),
                              ('annotated_portrait_control', 'scan_upright')]:
            for identity in identities:
                targets = identities if mode == 'annotated_portrait_control' else (
                    [identity, identities[(identities.index(identity)+1) % len(identities)]]
                    if capture == 'scan_upright' else [identity])
                for target in targets:
                    path = image_path(capture, identity)
                    payload = path.read_bytes() if mode == 'original_card' else annotated_portrait(capture, identity, settings)
                    started = perf_counter()
                    response = client.post('/v1/verify-id-card',
                        data={'reference_id': target, 'include_images': str(identity == target).lower()},
                        files={'id_card': (path.name, payload, 'application/octet-stream')})
                    elapsed = perf_counter() - started
                    body = response.json()
                    row = dict(mode=mode, capture=capture, query_identity=identity, reference_id=target,
                               expected_match=identity == target, http_status=response.status_code,
                               matched=body.get('matched'), similarity=body.get('similarity'),
                               error=body.get('error', {}).get('code'), seconds=round(elapsed, 4))
                    results.append(row)
                    if args.save_images and 'images' in body:
                        for name, encoded in body['images'].items():
                            (report_dir / f'{mode}_{capture}_{identity.replace(":", "_")}_{name}.png').write_bytes(
                                base64.b64decode(encoded.split(',', 1)[1]))
                    print(json.dumps(row), flush=True)
    (report_dir / 'pairs.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (report_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = dict(threshold=settings.threshold, model_version=engine.model_version,
                   pipeline_version=engine.pipeline_version,
                   identity_count=len(identities), planned_pairs=len(results),
                   successful_responses=sum(r['http_status'] == 200 for r in results),
                   failed_requests=sum(r['http_status'] != 200 for r in results),
                   enrollment_failures=sum(r['status'] != 'ok' for r in enrollment))
    summary['model_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (settings.detector, settings.recognizer)}
    for mode, capture in [('original_card', 'scan_upright'), ('original_card', 'scan_rotated'),
                          ('annotated_portrait_control', 'scan_upright')]:
        rows = [r for r in results if r['capture'] == capture and r['mode'] == mode]
        genuine = [r for r in rows if r['expected_match']]
        impostor = [r for r in rows if not r['expected_match']]
        summary[f'{mode}_{capture}'] = dict(
            genuine_pairs=len(genuine), genuine_accepted=sum(r['matched'] is True for r in genuine),
            genuine_rejected=sum(r['matched'] is False for r in genuine),
            genuine_processing_errors=sum(r['http_status'] != 200 for r in genuine),
            impostor_pairs=len(impostor), false_matches=sum(r['matched'] is True for r in impostor),
            impostor_rejected=sum(r['matched'] is False for r in impostor),
            impostor_processing_errors=sum(r['http_status'] != 200 for r in impostor))
    summary['limitations'] = [
        'Six artificial identities from a community mirror; not a biometric accuracy certification.',
        'Matching labels derive from document type plus filename, not the endpoint predictions.',
        'References are ground-truth-annotated portrait crops from templates; scans are different source files.',
        'Annotated portrait control uses ground-truth crop coordinates and does not establish full-card extraction success.',
        'Processing errors are reported separately and must not be counted as correct mismatches.',
        'HTTP requests use FastAPI TestClient with actual models and SQLite; no external network server.',
        'Threshold fixed during this run; these identities were reused during development, so this is not held-out validation.',
        'No removal of failed samples; original-card mode uses unmodified downloads.']
    (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
