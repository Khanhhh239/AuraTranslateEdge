# -*- coding: utf-8 -*-
"""Fixed-direction speech-to-speech pipeline: audio -> utterance segments -> ASR -> MT -> TTS, with per-stage timing (host wall time and graph time).
The work is done by `conversation.Engine` (shared with the multilingual conversation router)."""
from .audio import SR, segment, SEG_MAX_S, vad_mode
from .conversation import Engine


class Translator:
    def __init__(self, registry, src, tgt, denoise=False):
        self.r = registry; self.src, self.tgt = src, tgt; self.e = Engine(registry, denoise)
        self.asr = self.e.asr(src); self.mt = self.e.mt; self.tts = self.e.tts(tgt)
        self.graphs = self.e.graphs_for(src, tgt)

    def translate_segment(self, wav, breaks=None):
        return self.e.translate_segment(wav, self.src, self.tgt, breaks=breaks)

    def _segments(self, wav, max_seg_s=None):
        max_seg = max_seg_s or min(getattr(self.asr, "max_s", SEG_MAX_S), SEG_MAX_S)
        if vad_mode() == "hybrid":
            segs, brks = segment(wav, max_s=max_seg, return_breaks=True)
            return list(zip(segs, brks))
        return [(s, None) for s in segment(wav, max_s=max_seg)]

    def translate(self, wav, max_seg_s=None):
        out = []
        for (a, b), brk in self._segments(wav, max_seg_s):
            r = self.translate_segment(wav[a:b], breaks=brk); r["span"] = (a / SR, b / SR); out.append(r)
        return out

    def translate_stream(self, wav, max_seg_s=None):
        """Generator for live output: one result per sentence as soon as its audio is ready (Engine.translate_segment_stream)."""
        for (a, b), brk in self._segments(wav, max_seg_s):
            for r in self.e.translate_segment_stream(wav[a:b], self.src, self.tgt, breaks=brk):
                r["span"] = (a / SR, b / SR); yield r
