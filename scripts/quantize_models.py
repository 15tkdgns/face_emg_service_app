"""
ONNX FP32 to INT8 dynamic quantization.
Vercel 250MB function size limit workaround.

Usage:
    cd face_emg_service
    python scripts/quantize_models.py
"""
import os
import shutil
import tempfile

from onnxruntime.quantization import QuantType, quantize_dynamic

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODELS = [
    "densenet121.onnx",
    "densenet121_new.onnx",
]


def quantize(src_name: str):
    src = os.path.join(MODELS_DIR, src_name)
    dst = os.path.join(MODELS_DIR, src_name.replace(".onnx", "_q.onnx"))

    if not os.path.isfile(src):
        print(f"[SKIP] {src_name} not found")
        return

    src_mb = os.path.getsize(src) / 1024 / 1024
    print(f"[INFO] Quantizing {src_name} ({src_mb:.1f}MB)...")

    # Workaround for Korean path: use ASCII temp directory
    with tempfile.TemporaryDirectory(prefix="onnx_q_") as tmp:
        tmp_src = os.path.join(tmp, "model.onnx")
        tmp_dst = os.path.join(tmp, "model_q.onnx")
        shutil.copy2(src, tmp_src)

        quantize_dynamic(tmp_src, tmp_dst, weight_type=QuantType.QInt8)

        shutil.copy2(tmp_dst, dst)

    dst_mb = os.path.getsize(dst) / 1024 / 1024
    ratio = src_mb / dst_mb
    print(f"[DONE] {src_name.replace('.onnx', '_q.onnx')} ({dst_mb:.1f}MB) - {ratio:.1f}x smaller")


if __name__ == "__main__":
    for model in MODELS:
        quantize(model)
    print("\nDone. Add models/*_q.onnx to git.")
