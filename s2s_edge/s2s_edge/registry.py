# -*- coding: utf-8 -*-
"""Model registry: models.json maps component names to files. Paths are relative to the json file (or absolute for development).
  {"asr_vi": {"onnx": ..., "ctx": ..., "name": ...}, "asr_xx": ..., "mt": ..., "tts_vi": ..., "tts_zh": ..., "tts_enko": ...}
ASR windows: extra static windows of the same ASR graph are keys "<base>@<window>" (Zipformer: fbank frames, 100/s; SenseVoice encoder: LFR
frames, 60 ms each), e.g. "asr_vi@500", "asr_xx_encoder@100"; the base key is the largest window (1500 / 500, or its "window" field).
SenseVoice features: "_sv_frontend": "host" (exact FunASR features on the CPU, no frontend graph) or "graph" (the fp16 NPU frontend graph,
default when "asr_xx_frontend" is in the file); environment variable S2S_SV_FRONTEND overrides it."""
import json, os
from .backends import make_graph
from .asr import AsrVi, AsrXx, T_FRAMES, SV_FRAMES, load_cmvn
from .mt import Nllb
from .tts import TtsVi, TtsZh, TtsEnKo

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DIRECTIONS = [("vi", "en"), ("vi", "zh"), ("vi", "ko"), ("en", "vi"), ("zh", "vi"), ("ko", "vi")]


class Registry:
    def __init__(self, config_path, backend="npu", glossary_path=None):
        self.cfg = json.load(open(config_path, encoding="utf-8")); self.dir = os.path.dirname(os.path.abspath(config_path)); self.backend = backend; self._cache = {}
        gp = glossary_path or os.environ.get("S2S_GLOSSARY"); self.glossary = json.load(open(gp, encoding="utf-8")) if gp and os.path.exists(gp) else {}

    def graph(self, key):
        if key not in self._cache:
            spec = dict(self.cfg[key]); spec.setdefault("name", key)
            self._cache[key] = make_graph(self.dir, spec, self.backend)
        return self._cache[key]

    def files(self, keys):
        """Model files of the given keys, each once (ASR windows linked into one context binary share the file)."""
        ext = "onnx" if self.backend == "cpu" else "ctx"
        return list(dict.fromkeys(os.path.join(self.dir, self.cfg[k][ext]) for k in keys))

    def windows(self, base, default):
        """[(window, key)] ascending: the base key (its "window" field, else `default`) and every "<base>@<window>" key."""
        out = []
        for k, spec in self.cfg.items():
            if k == base: out.append((int(spec.get("window", default)), k))
            elif k.startswith(base + "@"): out.append((int(k.split("@", 1)[1]), k))
        return sorted(out)

    def sv_frontend(self):
        """'graph' (NPU frontend graph) or 'host' (s2s_edge.asr.sv_features)."""
        want = (os.environ.get("S2S_SV_FRONTEND") or self.cfg.get("_sv_frontend") or ("graph" if "asr_xx_frontend" in self.cfg else "host")).lower()
        return "graph" if want == "graph" and "asr_xx_frontend" in self.cfg else "host"

    def asr_keys(self, lang):
        if lang == "vi": return [k for _, k in self.windows("asr_vi", T_FRAMES)]
        if "asr_xx" in self.cfg: return ["asr_xx"]
        return (["asr_xx_frontend"] if self.sv_frontend() == "graph" else []) + [k for _, k in self.windows("asr_xx_encoder", SV_FRAMES)]

    def transducer(self):
        """RNN-T head of the Vietnamese Zipformer on the host CPU (asr.Transducer), when models.json has "asr_vi_decoder" and "asr_vi_joiner"
        (their "onnx" files are used with every backend: 5 + 4 MB, tiny host work). S2S_VI_DECODE=ctc turns it off."""
        if os.environ.get("S2S_VI_DECODE", "rnnt").lower() == "ctc" or not ("asr_vi_decoder" in self.cfg and "asr_vi_joiner" in self.cfg): return None
        if getattr(self, "_rnnt", None) is None:
            from .asr import Transducer
            p = lambda k: os.path.join(self.dir, self.cfg[k]["onnx"])
            try: self._rnnt = Transducer(p("asr_vi_decoder"), p("asr_vi_joiner"))
            except Exception as e:                                        # never take the Vietnamese ASR down: fall back to the CTC head
                import sys; print(f"[s2s] RNN-T decoder/joiner unusable ({e!r}) -> CTC greedy", file=sys.stderr); self._rnnt = False
        return self._rnnt or None

    def asr(self, lang):
        if lang == "vi":
            known = set(open(os.path.join(ASSETS, "piper_vi_words.txt"), encoding="utf-8").read().splitlines())
            return AsrVi([(w, self.graph(k)) for w, k in self.windows("asr_vi", T_FRAMES)], os.path.join(ASSETS, "zipformer_bpe.model"), known, self.transducer())
        if "asr_xx" in self.cfg: return AsrXx(self.graph("asr_xx"), None, lang)
        enc = [(w, self.graph(k)) for w, k in self.windows("asr_xx_encoder", SV_FRAMES)]
        if self.sv_frontend() == "graph": return AsrXx(self.graph("asr_xx_frontend"), enc, lang)
        return AsrXx(None, enc, lang, load_cmvn(os.path.join(ASSETS, "sensevoice_am.mvn")))

    def release(self, key):
        """Drop a graph from the cache and free its runtime memory (MT fallback after an OOM / runtime error)."""
        import gc
        g = self._cache.pop(key, None)
        if g is not None: g.release()
        gc.collect()

    def mt_candidates(self):
        """Decoder graphs from the most beams to greedy: every mt_decB* key (sorted by "B", descending), then mt_dec.
        S2S_MT_MAXB=<n> skips beam graphs with more than n beams (S2S_MT_MAXB=1 -> greedy only)."""
        maxb = int(os.environ.get("S2S_MT_MAXB", "99"))
        beams = sorted([k for k in self.cfg if k.startswith("mt_decB") and self.cfg[k].get("B", 4) <= maxb], key=lambda k: -self.cfg[k].get("B", 4))
        return beams + (["mt_dec"] if "mt_dec" in self.cfg else [])

    def _build_mt(self, key):
        from .mt import NllbSplit, NllbBeam
        e = self.cfg["mt_enc"]; d = self.cfg[key]; kw = dict(L=d.get("L", 12), H=d.get("H", 16), dk=d.get("dk", 64))
        if key == "mt_dec": return NllbSplit(self.graph("mt_enc"), self.graph("mt_dec"), ASSETS, S=e.get("S", 32), Tm=d.get("Tm", 48), **kw)
        return NllbBeam(self.graph("mt_enc"), self.graph(key), ASSETS, S=e.get("S", 64), Tm=d.get("Tm", 96), B=d.get("B", 4), **kw)

    def mt(self):
        """One shared MT object per registry. With split graphs it is an MtFallback over mt_candidates() (largest beam first, released and
        degraded on OOM / runtime errors)."""
        if getattr(self, "_mt_obj", None) is not None: return self._mt_obj
        if "mt" in self.cfg: self._mt_obj = Nllb(self.graph("mt"), ASSETS); return self._mt_obj
        from .mt import MtFallback
        self._mt_obj = MtFallback(self, self.mt_candidates(), self._build_mt, probe=os.environ.get("S2S_MT_PROBE", "1") != "0")
        return self._mt_obj

    def tts(self, lang):
        if lang == "vi": return TtsVi(self.graph("tts_vi"), ASSETS, self.glossary)
        if lang == "zh": return TtsZh(self.graph("tts_zh"))
        return TtsEnKo(self.graph("tts_enko"), lang, ASSETS)

    def gold_path(self, key):
        """Golden vectors of a graph: the model set's own directory first (models.json "_golden", e.g. golden_13b for the NLLB-1.3B set), then golden/."""
        root = os.path.dirname(ASSETS)
        for d in ([self.cfg["_golden"]] if "_golden" in self.cfg else []) + ["golden"]:
            p = os.path.join(root, d, f"{key}.npz")
            if os.path.exists(p): return p
        return os.path.join(root, self.cfg.get("_golden", "golden"), f"{key}.npz")

    def warmup(self, directions, log=print):
        """Load every graph the given directions need and run each once on its golden input (golden/<key>.npz), so the first live
        utterance does not pay for loading ~2 GB of context binaries and the first-call HTP setup. Returns {key: (load_s, first_run_s)}."""
        import time, numpy as np
        out = {}; t_all = time.perf_counter()
        t0 = time.perf_counter(); m = self.mt()                                # MT first: MtFallback probes the biggest beam and degrades if it does not fit
        log(f"  warmup MT: decoder {getattr(m, 'active', 'mt')} ready in {time.perf_counter() - t0:.1f} s" + (f" (dropped: {[k for k, _ in m.failures]})" if getattr(m, "failures", None) else ""))
        keys = []
        for s, t in directions:
            for k in self.components_for(s, t):
                if k not in keys: keys.append(k)
        for k in keys:
            try:
                t0 = time.perf_counter(); g = self.graph(k); t1 = time.perf_counter(); run_s = None; gp = self.gold_path(k)
                if os.path.exists(gp):
                    z = np.load(gp); g(**{n[3:]: z[n] for n in z.files if n.startswith("in_")}); run_s = time.perf_counter() - t1
                out[k] = (round(t1 - t0, 2), None if run_s is None else round(run_s, 3))
                log(f"  warmup {k}: load {t1 - t0:.1f} s" + ("" if run_s is None else f", first run {run_s * 1000:.0f} ms"))
            except Exception as e:
                log(f"  warmup {k}: FAILED {repr(e)[:200]}")
        langs = list(dict.fromkeys("vi" if s == "vi" else "xx" for s, _ in directions))   # en/zh/ko share the SenseVoice graphs: warm them once
        for lang in langs:                                                # every ASR window once (a golden run covers only one input length)
            lang = "en" if lang == "xx" else lang
            try:
                t0 = time.perf_counter(); a = self.asr(lang); a.warmup()
                log(f"  warmup ASR {lang}: {len(getattr(a, 'ws', []) or [1])} window(s) in {time.perf_counter() - t0:.1f} s")
            except Exception as e:
                log(f"  warmup ASR {lang}: FAILED {repr(e)[:200]}")
        if any(s == "vi" for s, _ in directions):                        # Vietnamese punctuation/casing model (CPU), see conversation.restore_vi
            from .conversation import VI_PUNCT_MODE
            if VI_PUNCT_MODE == "model":
                from .punct_vi import get
                t0 = time.perf_counter(); m = get(ASSETS)
                if m is not None: m("xin chào hôm nay chúng ta họp lúc ba giờ"); log(f"  warmup vi punctuation model: {time.perf_counter() - t0:.1f} s")
                else: log("  warmup vi punctuation model: assets/vi_capu missing -> rule-based vi_punct")
        log(f"  warmup done: {len(keys)} graphs in {time.perf_counter() - t_all:.1f} s")
        return out

    def components_for(self, src, tgt):
        asr = self.asr_keys(src)
        if "mt" in self.cfg: mt = ["mt"]
        else:
            if getattr(self, "mt_active", None) is None: self.mt()            # build (probe + fallback) before anyone loads the biggest beam graph directly
            mt = ["mt_enc", self.mt_active]                                   # the decoder MtFallback settled on
        return asr + mt + [{"vi": "tts_vi", "zh": "tts_zh"}.get(tgt, "tts_enko")]
