"""
감정인식 FastAPI 서비스 (ONNX Runtime 전용)

엔드포인트:
  GET  /api/health
  GET  /api/models
  POST /api/analyze
  POST /api/analyze/compare
  POST /api/analyze/base64
"""
import base64
import logging
import os
import sys
import traceback as tb

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.predictor import ModelManager, detect_and_crop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Face Emotion API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

manager = ModelManager()  # lazy loading — 첫 요청 시 모델 자동 로드


def _decode_image(contents: bytes) -> np.ndarray:
    np_arr = np.frombuffer(contents, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='이미지 디코딩 실패')
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'models': list(manager.available_models()),
    }


@app.get('/api/models')
def get_models():
    return {'models': manager.available_models()}


@app.post('/api/analyze')
async def analyze(
    file:     UploadFile = File(...),
    model_id: str        = Form(default='densenet121'),
):
    try:
        if model_id not in manager.sessions:
            loaded = list(manager.sessions.keys())
            if not loaded:
                raise HTTPException(status_code=503, detail='로드된 모델 없음')
            model_id = loaded[0]

        img_bgr = _decode_image(await file.read())
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)
        result = manager.predict(model_id, face_rgb)
        if result is None:
            raise HTTPException(status_code=503, detail=f'추론 실패: {model_id}')

        return {**result, 'face_b64': face_b64, 'face_detected': bbox is not None,
                'bbox': bbox, 'model_id': model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'analyze error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/analyze/compare')
async def analyze_compare(file: UploadFile = File(...)):
    try:
        img_bgr = _decode_image(await file.read())
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)
        results = manager.predict_all(face_rgb)
        if not results:
            raise HTTPException(status_code=503, detail='로드된 모델 없음')
        return {'results': results, 'face_b64': face_b64,
                'face_detected': bbox is not None, 'bbox': bbox}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'compare error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/analyze/base64')
async def analyze_base64(payload: dict):
    try:
        image_b64 = payload.get('image_b64', '')
        model_id  = payload.get('model_id', 'densenet121')
        compare   = payload.get('compare', False)
        if not image_b64:
            raise HTTPException(status_code=400, detail='image_b64 없음')
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        img_bgr = _decode_image(base64.b64decode(image_b64))
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)
        if compare:
            results = manager.predict_all(face_rgb)
            return {'results': results, 'face_b64': face_b64,
                    'face_detected': bbox is not None}
        if model_id not in manager.sessions:
            model_id = next(iter(manager.sessions), None)
            if not model_id:
                raise HTTPException(status_code=503, detail='로드된 모델 없음')
        result = manager.predict(model_id, face_rgb)
        return {**result, 'face_b64': face_b64, 'face_detected': bbox is not None,
                'model_id': model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'base64 error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))
