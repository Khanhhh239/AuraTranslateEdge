# -*- coding: utf-8 -*-
"""Multilingual conversation: every utterance is routed on its own, so speakers may switch language between turns.
 pair mode : two languages (one is Vietnamese): vi -> X and X -> vi.   e.g. --pair vi,ko
 auto mode : Vietnamese plus any of en / zh / ko; the partner language is detected from the other speaker (sticky, default `partner`).
Language identification per utterance (cheap, no extra model):
 1. the Vietnamese CTC model always runs first (it is the cheapest ASR): high confidence and mostly valid Vietnamese words  =>  Vietnamese;
 2. otherwise, with one candidate left that language is used; with several (auto mode) SenseVoice runs in `auto` language mode and the script of its
    text decides (Hangul -> ko, Han -> zh, Latin -> en)."""
import re
import numpy as np
from .audio import SR, segment, normalize_level
from .asr import AsrXx
from .names import NameTransliterator
import time

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ]"); HAN = re.compile(r"[一-鿿]"); LATIN = re.compile(r"[A-Za-z]")


def script_lang(text):
    h, c, l = len(HANGUL.findall(text)), len(HAN.findall(text)), len(LATIN.findall(text))
    if h + c + l == 0: return None
    return "ko" if h >= max(c, l) else ("zh" if c >= l else "en")


class Engine:
    """Shared models for all directions (loaded lazily): ASR per language, one MT, TTS per language."""

    def __init__(self, registry, denoise=False):
        self.r = registry; self._asr = {}; self._tts = {}; self._mt = None; self.names = None; self.denoise = denoise; self._den = None

    def prep(self, wav):
        """Level normalisation (+ optional GTCRN speech enhancement on the CPU) before ASR."""
        if self.denoise:
            if self._den is None:
                import os
                from .denoise import Gtcrn
                from .registry import ASSETS
                self._den = Gtcrn(os.path.join(ASSETS, "gtcrn_dns3.onnx"))
            wav = self._den(wav)
        return normalize_level(wav)

    def asr(self, lang):
        if lang not in self._asr: self._asr[lang] = self.r.asr(lang)
        return self._asr[lang]

    def tts(self, lang):
        if lang not in self._tts: self._tts[lang] = self.r.tts(lang)
        return self._tts[lang]

    @property
    def mt(self):
        if self._mt is None: self._mt = self.r.mt(); self.names = NameTransliterator(self._mt, self.r.glossary)
        return self._mt

    def graphs_for(self, src, tgt): return [self.r.graph(k) for k in self.r.components_for(src, tgt)]

    def gtime(self, graphs): return sum(g.total_s for g in graphs), sum(g.calls for g in graphs)

    def translate_segment(self, wav, src, tgt, text=None, prepped=False):
        """text: pre-computed ASR text (language identification already ran the ASR)."""
        wav = wav if prepped else self.prep(wav); gr = self.graphs_for(src, tgt); g0, c0 = self.gtime(gr); t = {}; a = time.perf_counter()
        if text is None: text = self.asr(src)(wav)
        for pat, rep in (self.r.glossary.get("_asr_fix", {}).get(src, {}) or {}).items():      # domain fixes: how the ASR (mis)spells a term -> the term
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        b = time.perf_counter(); t["asr_s"] = b - a
        if len(re.sub(r"[\W_]+", "", text)) < 2: text = ""                                 # nothing intelligible (noise, breath, a lone "I.")
        mt_text = self.mt(text, src, tgt) if text.strip() else ""
        if tgt in ("zh", "ko") and mt_text: mt_text = self.names(mt_text, tgt)
        c = time.perf_counter(); t["mt_s"] = c - b; tts = self.tts(tgt)
        out, sr = tts.synthesize(mt_text) if mt_text.strip() else (np.zeros(0, np.float32), tts.sr); d = time.perf_counter(); t["tts_s"] = d - c
        g1, c1 = self.gtime(gr); t["total_s"] = d - a; t["graph_s"] = g1 - g0; t["graph_calls"] = c1 - c0; t["host_s"] = t["total_s"] - t["graph_s"]
        t["audio_in_s"] = len(wav) / SR; t["audio_out_s"] = len(out) / sr
        return dict(src=src, tgt=tgt, asr=text, mt=mt_text, wav=out, sr=sr, timing=t)


class LanguageID:
    """vi / not-vi from confidence statistics of the Vietnamese CTC model: logistic regression on six features (assets/lid_vi.json, trained on FLEURS
    clean speech: 10-fold CV 100 % on 120 clips, see docs/conversation.md). `bias` shifts the decision (positive = more willing to call it Vietnamese)."""

    def __init__(self, engine, model_path=None, bias=0.0):
        import json, os
        from .registry import ASSETS
        self.e = engine; self.bias = bias; self.m = json.load(open(model_path or os.path.join(ASSETS, "lid_vi.json"), encoding="utf-8"))

    def vi_prob(self, st):
        x = np.array([st["conf_nb"], st["known"], st["nb_ratio"], st.get("conf_all", st["conf_nb"]), st.get("blank_p", 1 - st["nb_ratio"]), min(st["nwords"], 30) / 30.0])
        z = float(((x - np.array(self.m["mean"])) / np.array(self.m["scale"])) @ np.array(self.m["coef"]) + self.m["intercept"] + self.bias)
        return 1.0 / (1.0 + np.exp(-z))

    def is_vi(self, stats):
        return stats["nwords"] >= 1 and self.vi_prob(stats) >= 0.5

    def other(self, wav, candidates):
        if len(candidates) == 1: return candidates[0], None
        sv = self.e.asr("en"); lang0 = sv.lang; sv.lang = "auto"
        try:
            text = sv(wav)
        finally: sv.lang = lang0
        return (script_lang(text) or candidates[0]), text


class Conversation:
    def __init__(self, registry, pair=None, auto=False, partner="en", denoise=False):
        self.e = Engine(registry, denoise); self.lid = LanguageID(self.e); self.auto = auto; self.partner = partner
        self.others = [l for l in (pair or ("vi", partner)) if l != "vi"] if not auto else ["en", "zh", "ko"]
        if not auto: self.partner = self.others[0]

    def route(self, wav):
        """-> (src, tgt, vi_text_or_None)"""
        asr_vi = self.e.asr("vi"); text = asr_vi(wav); st = dict(asr_vi.stats)
        if self.lid.is_vi(st): return "vi", self.partner, text, st
        src, text = self.lid.other(wav, self.others)                   # auto mode: SenseVoice text is reused, no second pass
        if self.auto and text and len(re.sub(r"[\W_]+", "", text)) >= 3: self.partner = src       # junk fragments must not change the partner language
        return src, "vi", text, st

    def translate(self, wav, max_seg_s=12.0):
        out = []
        for a, b in segment(wav, max_s=max_seg_s):
            seg = self.e.prep(wav[a:b]); src, tgt, text, st = self.route(seg); r = self.e.translate_segment(seg, src, tgt, text=text, prepped=True)
            r["span"] = (a / SR, b / SR); r["lid"] = st
            if r["mt"]: out.append(r)
        return out
