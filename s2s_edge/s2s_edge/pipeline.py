# -*- coding: utf-8 -*-
"""Fixed-direction speech-to-speech pipeline: audio -> utterance segments -> ASR -> MT -> TTS, with per-stage timing (host wall time and graph time).
The work is done by `conversation.Engine` (shared with the multilingual conversation router)."""
from .audio import SR, effective_max_s, segment
from .conversation import Engine
from .registry import resolve_runtime_profile


class Translator:
    def __init__(self, registry, src, tgt, denoise=False, vad_profile=None):
        self.r = registry; self.src, self.tgt = src, tgt; self.e = Engine(registry, denoise)
        self.vad_profile = resolve_runtime_profile(vad_profile=vad_profile if vad_profile is not None else getattr(registry, "vad_profile", None))["vad_profile"]
        self.asr = self.e.asr(src); self.mt = self.e.mt; self.tts = self.e.tts(tgt)
        self.graphs = self.e.graphs_for(src, tgt)

    def translate_segment(self, wav, breaks=None):
        kw = {"breaks": breaks} if breaks else {}
        return self.e.translate_segment(wav, self.src, self.tgt, **kw)

    def _segments(self, wav, max_seg_s=None, vad_profile=None):
        prof = resolve_runtime_profile(vad_profile=vad_profile if vad_profile is not None else self.vad_profile)["vad_profile"]
        max_seg = effective_max_s(getattr(self.asr, "max_s", 12.0), vad_profile=prof, max_seg_s=max_seg_s)
        res = segment(wav, max_s=max_seg, vad_profile=prof, return_breaks=True)
        if isinstance(res, tuple):
            return res
        return res, [[] for _ in res]

    def translate(self, wav, max_seg_s=None, vad_profile=None):
        spans, brs = self._segments(wav, max_seg_s=max_seg_s, vad_profile=vad_profile)
        out = []
        for (a, b), br in zip(spans, brs):
            r = self.translate_segment(wav[a:b], breaks=br or None)
            r["span"] = (a / SR, b / SR); out.append(r)
        return out

    def translate_stream(self, wav, max_seg_s=None, vad_profile=None):
        """Generator for live output: one result per sentence as soon as its audio is ready (Engine.translate_segment_stream)."""
        spans, brs = self._segments(wav, max_seg_s=max_seg_s, vad_profile=vad_profile)
        for (a, b), br in zip(spans, brs):
            kw = {"breaks": br} if br else {}
            for r in self.e.translate_segment_stream(wav[a:b], self.src, self.tgt, **kw):
                r["span"] = (a / SR, b / SR); yield r
