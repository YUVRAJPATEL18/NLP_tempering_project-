from functools import partial
import sqlite3
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.errors import PipelineError
from app.face import OpenCVFaceEngine
from app.repository import SQLiteReferenceRepository
from app.service import VerificationService


def create_app(service: VerificationService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    service = service or VerificationService(SQLiteReferenceRepository(settings.database), OpenCVFaceEngine(settings), settings)
    api = FastAPI(title='ID Card Face Verification', version='1.0.0')

    @api.exception_handler(PipelineError)
    async def pipeline_error(request: Request, exc: PipelineError):
        return JSONResponse(status_code=exc.status, content={'error': {'code': exc.code, 'message': exc.message}}, headers={'Cache-Control': 'no-store'})

    @api.post('/v1/verify-id-card')
    async def verify_id_card(id_card: UploadFile = File(...),
                             reference_id: str = Form(..., min_length=1, max_length=128),
                             include_images: bool = Form(False)):
        try:
            payload = await id_card.read(settings.max_upload_bytes + 1)
        finally:
            await id_card.close()
        try:
            result = await run_in_threadpool(partial(service.verify, reference_id, payload, include_images))
        except sqlite3.Error:
            raise PipelineError('DATABASE_UNAVAILABLE', 'Reference database is unavailable', 503)
        except cv2.error:
            raise PipelineError('FACE_PROCESSING_FAILED', 'Face processing failed; retry with a clearer image', 422)
        result['request_id'] = str(uuid4())
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})

    return api


app = create_app()
