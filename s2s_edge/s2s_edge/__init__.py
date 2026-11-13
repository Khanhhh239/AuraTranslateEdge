"""s2s_edge: offline on-device speech-to-speech translation (vi <-> en / zh / ko) for Qualcomm Dragonwing IQ-9075."""
__version__ = "0.1.0"

# qai_appbuilder (imported when the first NPU graph is created) REPLACES sys.modules["onnxruntime"] with its own QNN wrapper module, which breaks plain CPU ONNX
# models (punctuation model, GTCRN). Keep the real module here, imported before anything can touch it, and use `from . import _ort_real as ort` for CPU sessions.
try:
    import onnxruntime as _ort_real
except Exception:                                   # CPU onnxruntime missing: only the NPU graphs work
    _ort_real = None
