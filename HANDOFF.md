# Complete user guide: ID-card face verification

This service verifies an ID-card face against a claimed database record using
one endpoint. The current cosine threshold is **0.6** and the pipeline includes
rotation retries. [METRICS.md](METRICS.md) is the current evaluation report.

## 1. Before starting

You need the complete project folder, Python 3.13, PowerShell, and internet
access for the initial dependency/model downloads. Face processing runs locally
on CPU; an OpenAI API key or GPU is not required. Keep the folder structure
intact. Run every command below from the project root, which contains this guide,
`requirements.txt`, `app/`, and `scripts/`.

If the folder is in a different location on your computer:

```powershell
$projectFolder = Read-Host 'Enter the full path to the project folder'
Set-Location -LiteralPath $projectFolder
python --version
```

Choose between two workflows: **dataset demo** uses six artificial identities
already provided for testing; **your own photos** requires enrollment into a
separate database. Reference IDs identify existing records. The endpoint does
not infer an ID number from the uploaded card.

## 2. First-time setup and dataset demo

Use Python 3.13 (the validated version). Run from the repository root in
PowerShell. A clean environment avoids conflicting OpenCV installations.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_models.py
python scripts/test_dataset_subset.py
python scripts/report_dataset_metrics.py
python scripts/run_dataset_api.py
```

The small dataset is included. If missing, run
`python scripts/download_test_subset.py` before evaluation. Model binaries and
SQLite databases are Git ignored: the commands above recreate them after a Git
clone. Existing model files in a directly shared folder are reused.

If PowerShell blocks activation, run the virtual environment's Python directly:

If `.venv/Scripts/python.exe` does not exist, create it with `python -m venv
.venv` first. A shared folder or Git clone does not include a virtual environment.
If the system `python` already has the required packages installed, the demo
launcher also works without activation: `python scripts/run_dataset_api.py`.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/download_models.py
.venv\Scripts\python.exe scripts/test_dataset_subset.py
.venv\Scripts\python.exe scripts/report_dataset_metrics.py
.venv\Scripts\python.exe scripts/run_dataset_api.py
```

Keep the server terminal open while testing. `Application startup complete`
indicates startup succeeded. Stop it with **Ctrl+C**. To use a different port,
run `python scripts/run_dataset_api.py --port 8001`, then open
`http://127.0.0.1:8001/docs`.

## 3. Test with FastAPI docs

Open [FastAPI docs](http://127.0.0.1:8000/docs), expand the POST endpoint, select
**Try it out**, and enter:

| Field | Value |
| --- | --- |
| `id_card` | `datasets/midv2020_subset/dataset__scan_upright__images__alb_id__00.jpg` |
| `reference_id` | `alb_id:00` |
| `include_images` | `true` |

Expect HTTP 200 and `matched=true`. With the same image and reference `alb_id:01`,
expect `matched=false`. For rotation, use
`dataset__scan_rotated__images__alb_id__00.jpg` and `alb_id:00`.
Available IDs: `alb_id:00`, `alb_id:01`, `alb_id:02`, `fin_id:00`, `fin_id:01`,
and `fin_id:02`.

**Database selection:** the demo launcher selects
`datasets/midv2020_subset/test_references.sqlite3` and prints its IDs. Ordinary
Uvicorn defaults to `data/faces.sqlite3`, initially empty. A 404 with
`REFERENCE_NOT_FOUND` means the reference is absent from the selected database,
not that the HTTP endpoint is missing. Stop a running server with Ctrl+C before
starting another on the same port.

Click **Execute**, then scroll to **Server response**. Check HTTP status,
`matched`, `similarity`, and `threshold`. An accepted demo comparison should
show `threshold: 0.6`. With `include_images=true`, the returned image strings
are expected; they are not paths to missing files.

For an already prepared installation, daily startup only requires:

```powershell
.venv\Scripts\Activate.ps1
python scripts/run_dataset_api.py
```

Do not re-download models or re-run the dataset evaluation on every startup.
Do rerun evaluation when changing the matching pipeline or preparing a missing
demo database.

## 4. Enroll and test your own photos

Stop the demo server first. In an activated terminal at the project root, choose
a dedicated database and an existing, clear reference photo with exactly one face:

```powershell
$env:FACE_DATABASE = Join-Path (Get-Location).Path 'data/personal_test.sqlite3'
$env:FACE_MATCH_THRESHOLD = '0.6'
$referencePhoto = Read-Host 'Enter the full path to an existing reference JPEG or PNG'
Test-Path -LiteralPath $referencePhoto
python scripts/enroll_reference.py --reference-id person-123 --photo "$referencePhoto"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`Test-Path` must print `True`. Enrollment must print
`Reference photo validated and saved.` If it fails, do not assume the record
exists. Reusing `person-123` replaces that record's photo. Store the reference
ID used by the rest of your system instead when integrating.

In Swagger, upload a **different ID-card image of the same person** and enter
`reference_id=person-123`. For a negative comparison, upload a card showing a
different person. Comparing an image with itself only checks the software path;
it does not measure recognition accuracy.

Use a visible, sufficiently large portrait. Avoid glare, heavy blur, occlusion,
and clipped faces. For `MULTIPLE_FACES`, submit the primary portrait region or
add document-specific portrait selection in the larger system.

Use ordinary Uvicorn for this workflow. The dataset launcher deliberately
selects the dataset database, so it will not use `personal_test.sqlite3`.
Environment settings apply to the current terminal and its child processes;
after opening a new terminal, set `FACE_DATABASE` again before enrollment or
server startup. Developer auto-reload is available with `--reload`.

## 5. Call the API from another program

With the demo server running, a PowerShell request is:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/verify-id-card" -F "reference_id=alb_id:00" -F "id_card=@datasets/midv2020_subset/dataset__scan_upright__images__alb_id__00.jpg" -F "include_images=false"
```

Let the HTTP client generate the multipart boundary; do not manually set a
bare `Content-Type: multipart/form-data` header. A Python client can also save
the returned images:

```python
import base64
from pathlib import Path
import requests

card = Path('datasets/midv2020_subset/dataset__scan_upright__images__alb_id__00.jpg')
with card.open('rb') as stream:
    response = requests.post(
        'http://127.0.0.1:8000/v1/verify-id-card',
        data={'reference_id': 'alb_id:00', 'include_images': 'true'},
        files={'id_card': (card.name, stream, 'image/jpeg')},
        timeout=60,
    )
if response.status_code != 200:
    raise RuntimeError(f'{response.status_code}: {response.text}')
result = response.json()
print(result['matched'], result['similarity'], result['threshold'])
output = Path('output')
output.mkdir(exist_ok=True)
for name, data_url in result.get('images', {}).items():
    (output / f'{name}.png').write_bytes(base64.b64decode(data_url.split(',', 1)[1]))
```

Use a timeout that allows rotation retries. Requests run synchronously through
verification; no second result endpoint or polling is needed.

## 6. Understand the endpoint response

`POST /v1/verify-id-card` accepts multipart form data: JPEG/PNG `id_card`,
`reference_id` (1-128 characters), and optional `include_images` (default false).

```json
{
  "reference_id": "alb_id:00",
  "matched": true,
  "similarity": 0.6986696720123291,
  "threshold": 0.6,
  "model_version": "yunet-2023mar+sface-2021dec",
  "pipeline_version": "orientation-retry-v2",
  "face": {
    "bbox": [120, 80, 220, 200],
    "detection_confidence": 0.99,
    "rotation_degrees": 0
  },
  "coordinate_system": "exif-oriented image; [left, top, right, bottom), pixels",
  "request_id": "example-request-id"
}
```

The bounding box above is illustrative. Coordinates refer to the image after
EXIF orientation, with right/bottom exclusive. `rotation_degrees` reports the
counterclockwise correction. With `include_images=true`, PNG data URLs appear
in `images.aligned_face` and `images.annotated_card`; decode them to view images.
Swagger displays the URLs as text. Similarity is not a probability or confidence
percentage. A match means `similarity >= threshold`; a non-match is HTTP 200.

Pipeline errors use `{"error":{"code":"...","message":"..."}}`.
Missing/invalid form fields can instead use FastAPI's `detail` validation response.

| Status | Meaning |
| --- | --- |
| 400 | Empty or undecodable image |
| 404 | Missing reference record |
| 409 | Unusable reference photo |
| 413 | Upload exceeds 10 MiB or image exceeds 20 million pixels |
| 415 | Unsupported image format |
| 422 | No face, multiple faces, small face, input or processing failure |
| 503 | Models, database or embeddings unavailable |

## 7. How the pipeline works

1. Validate/decode the image and normalize EXIF orientation.
2. Fetch the claimed reference photo through the repository adapter.
3. Detect faces with YuNet. After a detection miss, retry right-angle rotations,
   then 30-degree increments. Stop at the first detection, independently of the
   reference identity and match score.
4. Require one face at least 32 pixels on each side. Multiple portraits, including
   some cards' secondary portraits, require a portrait-region upload.
5. Align with SFace landmarks into a 112 x 112 crop and extract a 128-value
   embedding. Process the reference with the same pipeline.
6. Compare embeddings with cosine similarity and return the decision. Map the
   bounding box back to the original EXIF-oriented image.

Pretrained model weights are unchanged. OpenCV access is locked to protect its
mutable buffers; inference runs outside the async event loop. Orientation
retries increase latency, including for images without a detectable face.

## 8. Modules, configuration and integration

| File | Responsibility |
| --- | --- |
| `app/main.py` | App factory, endpoint and HTTP errors |
| `app/service.py` | Reusable verification orchestration |
| `app/face.py` | Decode, detect, rotate, align and embed |
| `app/repository.py` | Repository protocol and SQLite adapter |
| `app/config.py` | Environment-backed settings and limits |
| `app/errors.py` | Structured pipeline errors |
| `tests/test_verification.py` | API contracts, threshold boundaries, rotation geometry |
| `scripts/` | Downloads, evaluation, reporting, enrollment and demo launcher |
| `datasets/midv2020_subset/` | Test inputs, provenance manifest, current evidence |

For your database, implement `ReferenceRepository.get_photo(reference_id)
-> bytes | None`. Inject it into `VerificationService(repository, engine,
settings)`. Call the service directly or expose it with
`create_app(service, settings)`. A replacement face backend implements
`FaceEngine`. SQLite uses
`reference_faces(reference_id TEXT PRIMARY KEY, photo BLOB NOT NULL)`.

Enrollment and the server must select the same database. There is no public
enrollment endpoint. The host application should authorize access to the claimed
reference record before calling verification. If calling this synchronous
service from an async route, run inference in a worker thread as `app/main.py`
does, rather than blocking the event loop.

| Environment variable | Default |
| --- | --- |
| `FACE_DATABASE` | `data/faces.sqlite3` |
| `FACE_DETECTOR` | `models/face_detection_yunet_2023mar.onnx` |
| `FACE_RECOGNIZER` | `models/face_recognition_sface_2021dec.onnx` |
| `FACE_MATCH_THRESHOLD` | `0.6` |

Paths are relative to the working directory unless absolute. Set environment
variables before launch and restart after changes. For example:
`$env:FACE_MATCH_THRESHOLD = '0.6'`.

## 9. Reproduce tests and interpret metrics

```powershell
python -m pytest tests -q
python scripts/test_dataset_subset.py
python scripts/report_dataset_metrics.py
```

The unit suite contains 18 tests. Dataset evaluation makes 54 endpoint calls
through FastAPI TestClient using real models and SQLite; it does not measure a
live network server. Current evidence under `datasets/midv2020_subset/results/`
includes `pairs.json`, `pairs.csv`, `summary.json`, `enrollment.json`, and
`metrics.json`. The report command writes the single root `METRICS.md`.
Visual outputs are optional:
`python scripts/test_dataset_subset.py --save-images`.

Read [METRICS.md](METRICS.md) after generating it. The current subset has six
genuine and six wrong-reference upright-card comparisons, six genuine rotated
comparisons, and a portrait-control set of six genuine plus 30 wrong-reference
comparisons. These are 54 calls, not 54 independent people. No rotated impostor
rate was measured. Current observed decisions are correct on this subset, but
the identities were reused while developing the pipeline and choosing 0.6.
Do not report that as universal 100% accuracy.

For wider validation, separate identities into development and final-test sets,
fix the threshold using development data, then measure genuine acceptance,
false matches, extraction failures, and latency on unseen final-test identities.
Keep poor-quality and rotated failures in the report. Changing the threshold
trades off false matches against false non-matches; it does not retrain SFace.

## 10. Troubleshooting

| Symptom | What to check or do |
| --- | --- |
| `REFERENCE_NOT_FOUND` | Use an existing exact ID and the same database as enrollment. For the demo, use `run_dataset_api.py`. |
| `FileNotFoundError` during enrollment | Supply a real existing file; check with `Test-Path`. |
| `MODELS_UNAVAILABLE` | Run the model download script; check model paths and working directory. |
| `NO_FACE` | Use a clearer, larger portrait; retries cannot recover every angle or quality issue. |
| `MULTIPLE_FACES` | Crop to the intended portrait; the endpoint refuses ambiguous detections. |
| `FACE_TOO_SMALL` | Upload a higher-resolution image or a tighter portrait region. |
| `INVALID_REFERENCE` | Replace the stored reference with a valid single-face photo. |
| 413 response | Reduce image byte size or dimensions within the documented limits. |
| FastAPI 422 with `detail` | Check the required `id_card` and `reference_id` fields. |
| Port 8000 already in use | Stop the old server or launch the demo with `--port 8001`. |
| Connection refused | Start the server and use the port printed in its terminal. |
| Threshold appears unchanged | Check inherited `FACE_MATCH_THRESHOLD`, set it to `0.6`, and restart. |
| Import/package errors | Use the activated virtual environment or its Python executable directly. |
| Demo database missing | Run `python scripts/test_dataset_subset.py`, then restart the demo launcher. |

`FACE_MATCH_THRESHOLD` is read when settings are loaded. Creating a `.env` file
alone does not change configuration: this project does not automatically load
`.env` files. Set environment variables explicitly in the launching terminal.

## 11. Dataset attribution and deployment limits

The included subset has 18 images and six annotation files (~19.8 MB), covering
six artificial identities and two document types. The
[source manifest](datasets/midv2020_subset/download_manifest.json) records source
paths and SHA-256 hashes. Downloads use HTTP ranges and ZIP CRC checks.

Source: [community mirror at the pinned revision](https://huggingface.co/datasets/AmAFakePerson123/TrialforGeneratedIDs/tree/c1da4db).
The mirror declares CC BY-SA 3.0; provenance was not independently authenticated.
Check original terms before redistribution and preserve attribution:
K. Bulatov et al., "MIDV-2020: A Comprehensive Benchmark Dataset for Identity
Document Analysis", Computer Optics 46(2), 252-270, 2022.
[Original paper](https://arxiv.org/abs/2107.00396).
Model sources/licences: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models).

The module does not read card fields, authenticate cards, prove liveness, search
for an unknown person across the database, or establish trustworthiness. These
synthetic identities were reused during development; the metrics are regression
evidence, not production accuracy. Validate with unseen identities and realistic
reference photos. The host must supply authentication, record authorization,
rate limits, TLS, protected biometric storage and retention controls. SQLite
photos are unencrypted. Enforce upload limits at the proxy too, since multipart
parsing precedes the endpoint size check.
