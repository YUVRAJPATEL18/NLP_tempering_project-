"""Run Swagger against the downloaded dataset's dedicated reference database."""
import argparse
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    database = ROOT / 'datasets' / 'midv2020_subset' / 'test_references.sqlite3'
    if not database.is_file():
        parser.error('Run python scripts/test_dataset_subset.py to prepare the reference database first.')
    with sqlite3.connect(database) as connection:
        identities = [row[0] for row in connection.execute('SELECT reference_id FROM reference_faces ORDER BY reference_id')]
    if not identities:
        parser.error('Dataset reference database is empty; run scripts/test_dataset_subset.py first.')
    os.environ['FACE_DATABASE'] = str(database)
    os.environ['FACE_DETECTOR'] = str(ROOT / 'models' / 'face_detection_yunet_2023mar.onnx')
    os.environ['FACE_RECOGNIZER'] = str(ROOT / 'models' / 'face_recognition_sface_2021dec.onnx')
    print(f'Database: {database}', flush=True)
    print(f'Reference IDs: {", ".join(identities)}', flush=True)
    from app.config import Settings
    from app.face import OpenCVFaceEngine
    print(f'Match threshold: {Settings().threshold}; pipeline: {OpenCVFaceEngine.pipeline_version}', flush=True)
    print(f'Docs: http://127.0.0.1:{args.port}/docs', flush=True)
    import uvicorn
    uvicorn.run('app.main:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
