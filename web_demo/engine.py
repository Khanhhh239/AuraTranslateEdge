"""S2S engine adapter for the web demo.

The web server only talks to `get_engine()` and the three methods below.
To plug in real models, implement a class with the same interface (for example
by wrapping ZipformerVi / SenseVoiceFp32 / NllbMt / PiperVi / SupertonicTTS /
MeloTtsZh from src/step5_pipeline/pipeline_s2s.py) and return it from
`get_engine()`.

Interface
---------
    name: str                                  shown in the UI header
    asr(wav: bytes, lang: str) -> str          wav = 16 kHz mono PCM16 WAV file bytes
    mt(text: str, src: str, tgt: str) -> str
    tts(text: str, lang: str) -> bytes         returns a complete WAV file (any sample rate)

Language codes: "vi", "en", "zh", "ko". Only the directions in PAIRS are served.
"""
import io
import math
import os
import random
import time
import wave

LANGS = ("vi", "en", "zh", "ko")
# Supported directions: Vietnamese <-> each foreign language.
PAIRS = (("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"), ("vi", "ko"), ("ko", "vi"))


class MockEngine:
    """Placeholder engine: canned sentences + a synthetic voice-like tone.

    Sleeps a little at each stage so the UI's progress states are visible.
    """

    name = "Mock engine"

    _SAMPLES = [
        {
            "vi": "Xin chào, hôm nay dây chuyền số ba hoạt động bình thường.",
            "en": "Hello, production line three is running normally today.",
            "zh": "你好，今天三号生产线运行正常。",
            "ko": "안녕하세요, 오늘 3번 생산 라인은 정상적으로 가동되고 있습니다.",
        },
        {
            "vi": "Vui lòng đeo mũ bảo hộ trước khi vào khu vực này.",
            "en": "Please wear a safety helmet before entering this area.",
            "zh": "进入该区域前请佩戴安全帽。",
            "ko": "이 구역에 들어가기 전에 안전모를 착용해 주세요.",
        },
        {
            "vi": "Lô hàng này cần được giao trước ba giờ chiều.",
            "en": "This shipment needs to be delivered before three in the afternoon.",
            "zh": "这批货需要在下午三点前送达。",
            "ko": "이 화물은 오후 3시 전에 배송되어야 합니다.",
        },
        {
            "vi": "Máy này đang quá nhiệt, hãy tắt nó ngay.",
            "en": "This machine is overheating, turn it off right away.",
            "zh": "这台机器过热了，请立即关掉。",
            "ko": "이 기계가 과열되고 있으니 즉시 꺼 주세요.",
        },
    ]

    def __init__(self):
        self._i = -1
        self._current = self._SAMPLES[0]

    def asr(self, wav, lang):
        time.sleep(random.uniform(0.35, 0.6))
        self._i = (self._i + 1) % len(self._SAMPLES)
        self._current = self._SAMPLES[self._i]
        return self._current[lang]

    def mt(self, text, src, tgt):
        time.sleep(random.uniform(0.25, 0.45))
        for s in self._SAMPLES:
            if s.get(src) == text:
                return s[tgt]
        return f"[{tgt}] {text}"

    def tts(self, text, lang):
        time.sleep(random.uniform(0.3, 0.5))
        return _babble_wav(text)


def _babble_wav(text, sr=16000):
    """Soft syllable-like tone bursts, length proportional to text."""
    n_syll = max(4, min(28, len(text) // 3))
    rng = random.Random(len(text))
    frames = bytearray()
    for _ in range(n_syll):
        dur = rng.uniform(0.11, 0.19)
        f0 = rng.uniform(170, 240)
        n = int(dur * sr)
        for i in range(n):
            t = i / sr
            env = math.sin(math.pi * i / n) ** 2
            f = f0 * (1 + 0.06 * math.sin(2 * math.pi * 5 * t))
            s = (math.sin(2 * math.pi * f * t) + 0.35 * math.sin(4 * math.pi * f * t)
                 + 0.15 * math.sin(6 * math.pi * f * t))
            v = int(max(-1, min(1, 0.22 * env * s)) * 32767)
            frames += v.to_bytes(2, "little", signed=True)
        frames += b"\x00\x00" * int(rng.uniform(0.03, 0.08) * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _find_s2s_root():
    """Folder that contains the s2s_edge Python package: $S2S_PKG, the installed package folder (web_demo/ copied next to s2s_edge/),
    or the repo layout (repo/s2s_edge/s2s_edge)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("S2S_PKG"), os.path.dirname(here), os.path.join(os.path.dirname(here), "s2s_edge")):
        if c and os.path.isfile(os.path.join(c, "s2s_edge", "__init__.py")):
            return os.path.abspath(c)
    return None


def _wav_to_f32(data):
    """WAV bytes (any rate, mono/stereo, PCM16/PCM32/float) -> float32 mono at 16 kHz."""
    import numpy as np
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes(); raw = w.readframes(n)
    if sw == 2: x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 4: x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1: x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else: raise ValueError(f"unsupported WAV sample width {sw}")
    if ch > 1: x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def _f32_to_wav(x, sr):
    import numpy as np
    pcm = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr)); w.writeframes(pcm)
    return buf.getvalue()


class S2SEngine:
    """The real pipeline (s2s_edge): NPU on the AIBOX 8550 (backend npu), ONNX on a laptop (backend cpu, needs the ONNX exports).
    Same code path as the CLI / box web UI: VAD segments -> ASR -> junk filter (clean_asr) -> Vietnamese punctuation -> MT (3.3B / 600M route,
    sentence by sentence, output guard, names, Korean politeness) -> TTS."""

    def __init__(self, models=None, backend=None):
        import sys
        root = _find_s2s_root()
        if root is None: raise RuntimeError("s2s_edge package not found: set S2S_PKG=<folder that contains s2s_edge/>")
        if root not in sys.path: sys.path.insert(0, root)
        from s2s_edge.registry import Registry
        from s2s_edge.conversation import Engine
        self.backend = backend or os.environ.get("S2S_BACKEND", "npu")
        models = models or os.environ.get("S2S_MODELS") or _default_models(root)
        self.reg = Registry(models, self.backend); self.e = Engine(self.reg); self.root, self.models = root, models
        self.name = f"s2s_edge · {self.backend.upper()}"
        if os.environ.get("S2S_WARMUP", "1") != "0":                     # load every graph once so the first request is not 1-2 min
            self.reg.warmup([p for p in PAIRS], log=lambda m: print("[warmup]", m, flush=True))
        st = self.e.mt.status() if hasattr(self.e.mt, "status") else {}
        if st: self.name += " · MT " + ("3.3B+600M" if st.get("big_in_use") else "600M") + (" (DEGRADED)" if st.get("degraded") else "")
        self.info = dict(root=root, models=os.path.abspath(models), backend=self.backend, mt=st)

    def asr(self, wav, lang):
        from s2s_edge.audio import segment_with_breaks
        from s2s_edge.asr_clean import clean_asr
        x = _wav_to_f32(wav); a = self.e.asr(lang); out = []
        for (s, t), brk in segment_with_breaks(x, max_s=getattr(a, "max_s", None), vad_profile=self.reg.vad_profile):
            seg = self.e.prep(x[s:t]); text, raw, we = self.e._asr_stage(seg, lang)
            text, _ = clean_asr(text, lang)
            if text: text, _ = self.e._punct_stage(text, lang, raw, we, brk, min_chars=1)
            if text: out.append(text)
        return ("" if lang == "zh" else " ").join(out)

    def mt(self, text, src, tgt):
        return self.e.translate_text(text, src, tgt) if text.strip() else ""

    def tts(self, text, lang):
        if not text.strip(): return _f32_to_wav([], 16000)
        wav, sr = self.e.tts(lang).synthesize(text)
        return _f32_to_wav(wav, sr)


def _default_models(root):
    """The model set tools/session0.sh validated (logs/models_ok.txt), else models_qcs8550/models.json (3.3B route + 600M)."""
    ok = os.path.join(root, "logs", "models_ok.txt")
    if os.path.isfile(ok):
        p = open(ok, encoding="utf-8").read().strip()
        if p: return p if os.path.isabs(p) else os.path.join(root, p)
    return os.path.join(root, "models_qcs8550", "models.json")


def get_engine():
    """S2S_ENGINE=real (default when the s2s_edge package is found) -> S2SEngine; S2S_ENGINE=mock -> MockEngine (UI work without models)."""
    want = os.environ.get("S2S_ENGINE", "real" if _find_s2s_root() else "mock").lower()
    if want == "mock":
        return MockEngine()
    return S2SEngine()
