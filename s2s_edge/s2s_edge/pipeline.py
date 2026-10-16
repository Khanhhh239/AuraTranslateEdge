# -*- coding: utf-8 -*-
"""Speech-to-speech pipeline: audio -> utterance segments -> ASR -> MT -> TTS, with per-stage timing (host wall time and graph time)."""
import time
import numpy as np
from .audio import SR, segment


class Translator:
    def __init__(self, registry, src, tgt):
        self.r = registry; self.src, self.tgt = src, tgt
        self.asr = registry.asr(src); self.mt = registry.mt(); self.tts = registry.tts(tgt)
        self.graphs = [registry.graph(k) for k in registry.components_for(src, tgt)]

    def _gtime(self): return sum(g.total_s for g in self.graphs), sum(g.calls for g in self.graphs)

    def translate_segment(self, wav):
        t = {}; g0, c0 = self._gtime(); a = time.perf_counter()
        text = self.asr(wav); b = time.perf_counter(); t["asr_s"] = b - a
        mt_text = self.mt(text, self.src, self.tgt) if text.strip() else ""; c = time.perf_counter(); t["mt_s"] = c - b
        out, sr = self.tts.synthesize(mt_text) if mt_text.strip() else (np.zeros(0, np.float32), self.tts.sr); d = time.perf_counter(); t["tts_s"] = d - c
        g1, c1 = self._gtime(); t["total_s"] = d - a; t["graph_s"] = g1 - g0; t["graph_calls"] = c1 - c0; t["host_s"] = t["total_s"] - t["graph_s"]
        t["audio_in_s"] = len(wav) / SR; t["audio_out_s"] = len(out) / sr
        return dict(asr=text, mt=mt_text, wav=out, sr=sr, timing=t)

    def translate(self, wav, max_seg_s=None):
        max_seg = max_seg_s or min(getattr(self.asr, "max_s", 12.0), 12.0)
        segs = segment(wav, max_s=max_seg); out = []
        for a, b in segs:
            r = self.translate_segment(wav[a:b]); r["span"] = (a / SR, b / SR); out.append(r)
        return out
