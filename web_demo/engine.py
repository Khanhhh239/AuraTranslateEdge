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


def get_engine():
    """Return the engine the server uses. Swap MockEngine for the real one here."""
    if os.environ.get("S2S_ENGINE", "mock") == "mock":
        return MockEngine()
    raise RuntimeError("Real engine not wired yet: implement it in web_demo/engine.py")
