# -*- coding: utf-8 -*-
"""Model registry: models.json maps component names to files. Paths are relative to the json file (or absolute for development).
  {"asr_vi": {"onnx": ..., "ctx": ..., "name": ...}, "asr_xx": ..., "mt": ..., "tts_vi": ..., "tts_zh": ..., "tts_enko": ...}"""
import json, os
from .backends import make_graph
from .asr import AsrVi, AsrXx
from .mt import Nllb
from .tts import TtsVi, TtsZh, TtsEnKo

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DIRECTIONS = [("vi", "en"), ("vi", "zh"), ("vi", "ko"), ("en", "vi"), ("zh", "vi"), ("ko", "vi")]


class Registry:
    def __init__(self, config_path, backend="npu"):
        self.cfg = json.load(open(config_path, encoding="utf-8")); self.dir = os.path.dirname(os.path.abspath(config_path)); self.backend = backend; self._cache = {}

    def graph(self, key):
        if key not in self._cache:
            spec = dict(self.cfg[key]); spec.setdefault("name", key)
            self._cache[key] = make_graph(self.dir, spec, self.backend)
        return self._cache[key]

    def files(self, keys):
        ext = "onnx" if self.backend == "cpu" else "ctx"
        return [os.path.join(self.dir, self.cfg[k][ext]) for k in keys]

    def asr(self, lang):
        if lang == "vi": return AsrVi(self.graph("asr_vi"), os.path.join(ASSETS, "zipformer_bpe.model"))
        if "asr_xx" in self.cfg: return AsrXx(self.graph("asr_xx"), None, lang)
        return AsrXx(self.graph("asr_xx_frontend"), self.graph("asr_xx_encoder"), lang)

    def mt(self):
        if "mt" in self.cfg: return Nllb(self.graph("mt"), ASSETS)
        from .mt import NllbSplit
        return NllbSplit(self.graph("mt_enc"), self.graph("mt_dec"), ASSETS)

    def tts(self, lang):
        if lang == "vi": return TtsVi(self.graph("tts_vi"))
        if lang == "zh": return TtsZh(self.graph("tts_zh"))
        return TtsEnKo(self.graph("tts_enko"), lang, ASSETS)

    def components_for(self, src, tgt):
        asr = ["asr_vi"] if src == "vi" else (["asr_xx"] if "asr_xx" in self.cfg else ["asr_xx_frontend", "asr_xx_encoder"])
        mt = ["mt"] if "mt" in self.cfg else ["mt_enc", "mt_dec"]
        return asr + mt + [{"vi": "tts_vi", "zh": "tts_zh"}.get(tgt, "tts_enko")]
