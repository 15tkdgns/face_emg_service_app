"""
ONNX Runtime 기반 감정인식 추론기.
Vercel 서버리스 환경을 위해 lazy loading 사용.
"""
import base64
import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

EMOTIONS      = ['기쁨', '당황', '분노', '상처']
EMOTION_EMOJI = {'기쁨': '😄', '당황': '😳', '분노': '😡', '상처': '😢'}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

MODEL_REGISTRY = {
    'densenet121': {
        'label':       'DenseNet121',
        'description': '기본 전처리 · Best 모델 (87.6%)',
        'onnx':        'densenet121.onnx',
        'color':       '#4F86C6',
        'val_acc':     0.8762,
        'f1_per':      {'기쁨': 0.968, '당황': 0.902, '분노': 0.860, '상처': 0.828},
        'use_clahe':   False,
        'use_edge':    False,
    },
}


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _extract_edge(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def detect_and_crop(img_bgr: np.ndarray):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    if len(faces) == 0:
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        x1, y1 = (w - s) // 2, (h - s) // 2
        face_bgr = img_bgr[y1:y1 + s, x1:x1 + s]
        bbox = None
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        px, py = int(fw * 0.1), int(fh * 0.1)
        x1 = max(0, x - px);  y1 = max(0, y - py)
        x2 = min(img_bgr.shape[1], x + fw + px)
        y2 = min(img_bgr.shape[0], y + fh + py)
        face_bgr = img_bgr[y1:y2, x1:x2]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    _, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    face_b64 = base64.b64encode(buf).decode('utf-8')
    return bbox, cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), face_b64


def _preprocess(face_rgb: np.ndarray, use_clahe: bool, use_edge: bool) -> np.ndarray:
    face = cv2.resize(face_rgb, (224, 224))
    if use_clahe:
        face = _apply_clahe(face)
    norm = (face.astype(np.float32) / 255.0 - MEAN) / STD
    rgb  = norm.transpose(2, 0, 1)
    if use_edge:
        edge = _extract_edge(face).astype(np.float32) / 255.0
        tensor = np.concatenate([rgb, edge[np.newaxis]], axis=0)
    else:
        tensor = rgb
    return tensor[np.newaxis].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ModelManager:
    """Lazy-loading model manager (Vercel 서버리스 대응)."""

    def __init__(self):
        self._sessions: dict = {}
        self._opts = ort.SessionOptions()
        self._opts.inter_op_num_threads = 1
        self._opts.intra_op_num_threads = 1

    def _get_session(self, model_id: str) -> ort.InferenceSession | None:
        if model_id in self._sessions:
            return self._sessions[model_id]
        info = MODEL_REGISTRY.get(model_id)
        if not info:
            return None
        path = os.path.join(MODELS_DIR, info['onnx'])
        if not os.path.isfile(path):
            logger.warning(f'[{model_id}] ONNX 없음: {path}')
            return None
        sess = ort.InferenceSession(path, sess_options=self._opts,
                                    providers=['CPUExecutionProvider'])
        self._sessions[model_id] = sess
        logger.info(f'[{model_id}] 로드 완료')
        return sess

    def available_models(self) -> list:
        return [
            {
                'id':          mid,
                'label':       info['label'],
                'description': info['description'],
                'color':       info['color'],
                'loaded':      os.path.isfile(os.path.join(MODELS_DIR, info['onnx'])),
                'val_acc':     info['val_acc'],
                'f1_per':      info['f1_per'],
            }
            for mid, info in MODEL_REGISTRY.items()
        ]

    def predict(self, model_id: str, face_rgb: np.ndarray) -> dict | None:
        sess = self._get_session(model_id)
        if sess is None:
            return None
        info   = MODEL_REGISTRY[model_id]
        tensor = _preprocess(face_rgb, info['use_clahe'], info['use_edge'])
        t0     = time.time()
        logits = sess.run(None, {'input': tensor})[0][0]
        elapsed = (time.time() - t0) * 1000
        probs  = _softmax(logits)
        pred   = int(probs.argmax())
        return {
            'emotion':    EMOTIONS[pred],
            'emoji':      EMOTION_EMOJI[EMOTIONS[pred]],
            'confidence': float(probs[pred]),
            'scores':     {e: float(probs[i]) for i, e in enumerate(EMOTIONS)},
            'infer_ms':   round(elapsed, 1),
        }

    def predict_all(self, face_rgb: np.ndarray) -> list:
        results = []
        for mid, info in MODEL_REGISTRY.items():
            res = self.predict(mid, face_rgb)
            if res:
                res.update({'model_id': mid, 'model_label': info['label'],
                            'color': info['color']})
                results.append(res)
        return results
