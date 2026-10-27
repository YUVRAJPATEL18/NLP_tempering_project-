"""Download the official OpenCV Zoo models explicitly, never during a request."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'face_detection_yunet_2023mar.onnx': 'face_detection_yunet',
    'face_recognition_sface_2021dec.onnx': 'face_recognition_sface',
}


def main():
    directory = ROOT / 'models'
    directory.mkdir(exist_ok=True)
    for filename, folder in MODELS.items():
        destination = directory / filename
        if destination.exists():
            print(f'Already exists: {filename}')
            continue
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{folder}/{filename}'
        temporary = destination.with_suffix('.download')
        try:
            with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size < 100_000:
                raise RuntimeError(f'Unexpected model download: {filename}')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Downloaded: {filename}')


if __name__ == '__main__':
    main()
