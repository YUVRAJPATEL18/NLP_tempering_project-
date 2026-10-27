"""Administrative CLI; no public enrollment endpoint is exposed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.face import OpenCVFaceEngine, decode_image
from app.repository import SQLiteReferenceRepository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-id', required=True)
    parser.add_argument('--photo', type=Path, required=True)
    args = parser.parse_args()
    if not args.reference_id.strip() or len(args.reference_id) > 128:
        parser.error('reference-id must contain 1 to 128 characters')
    if not args.photo.is_file():
        parser.error(f'Photo does not exist: {args.photo}. Supply the full path to an existing JPEG or PNG photo.')
    settings = Settings()
    payload = args.photo.read_bytes()
    OpenCVFaceEngine(settings).extract(decode_image(payload, settings))
    SQLiteReferenceRepository(settings.database).save_photo(args.reference_id, payload)
    print('Reference photo validated and saved.')


if __name__ == '__main__':
    main()
