# -*- coding: utf-8 -*-
"""NLLB-200-distilled-600M, one static graph (src_ids[1,32] i32, tgt_lang[1] i32 -> out_ids[1,16] i32 = [lang, y1..y15]).
Host: SentencePiece tokenize / detokenize and ADAPTIVE CHUNKING: a source piece whose translation is cut by the 15-token limit
(no </s> in output) is split in two and translated again, so long sentences are complete (costs extra graph calls)."""
import json, os, re, numpy as np

S, T = 32, 16
CODES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
SENT_END = re.compile(r"(?<=[\.\!\?;。！？；\n])\s*")
CLAUSE = re.compile(r"(?<=[,，、:：])\s*")


class Nllb:
    def __init__(self, graph, assets_dir):
        import sentencepiece as spm
        self.g = graph; self.sp = spm.SentencePieceProcessor(model_file=os.path.join(assets_dir, "nllb_spm.model"))
        self.ids = json.load(open(os.path.join(assets_dir, "nllb_ids.json"), encoding="utf-8")); self.lang = self.ids["lang"]
        self.calls = 0; self.truncations = 0

    def encode(self, text):
        return [3 if p == 0 else p + 1 for p in self.sp.encode(text)]            # fairseq offset: spm id + 1, <unk> -> 3

    def decode(self, ids):
        keep = [i - 1 for i in ids if i > 3 and i < 256001]
        return self.sp.decode(keep)

    def _run(self, toks, src, tgt):
        a = np.ones((1, S), np.int32); seq = [self.lang[CODES[src]]] + toks + [2]; a[0, :len(seq)] = seq
        out = np.asarray(self.g(src_ids=a, tgt_lang=np.array([self.lang[CODES[tgt]]], np.int32))[0]).reshape(-1)
        self.calls += 1; body = [int(v) for v in out[1:]]
        return body, (2 in body)

    def _split(self, text, src):
        for rx in (SENT_END, CLAUSE):
            parts = [p for p in rx.split(text) if p.strip()]
            if len(parts) > 1: return parts
        joiner = "" if src in ("zh",) else " "; units = list(text) if src in ("zh", "ko") and " " not in text.strip() else text.split()
        if len(units) < 2: return [text]
        h = len(units) // 2; return [joiner.join(units[:h]), joiner.join(units[h:])]

    def _tr(self, text, src, tgt, depth=0):
        text = text.strip()
        if not text: return ""
        toks = self.encode(text)
        if len(toks) <= S - 2:
            body, done = self._run(toks, src, tgt)
            if done or depth > 6:
                if not done: self.truncations += 1
                return self.decode(body[: body.index(2)] if 2 in body else body)
        parts = self._split(text, src)
        if len(parts) == 1:                                        # cannot split further: accept the truncated output
            self.truncations += 1; body, _ = self._run(toks[: S - 2], src, tgt); return self.decode(body[: body.index(2)] if 2 in body else body)
        sep = "" if tgt == "zh" else " "
        return sep.join(x for x in (self._tr(p, src, tgt, depth + 1) for p in parts) if x)

    def __call__(self, text, src, tgt):
        sents = [p for p in SENT_END.split(text.strip()) if p.strip()]
        sep = "" if tgt == "zh" else " "
        return sep.join(x for x in (self._tr(s, src, tgt) for s in sents) if x)


class NllbSplit(Nllb):
    """Same host logic, but the model is two static graphs (encoder, one decoder step with static KV cache) and the greedy decode loop runs on the host.
    No 15-token limit (up to Tm-1 = 47 target tokens), small files (no unrolled copies of the decoder weights)."""
    Tm = 48

    def __init__(self, enc, dec, assets_dir):
        Nllb.__init__(self, None, assets_dir); self.enc = enc; self.dec = dec; self.L, self.H, self.dk = 12, 16, 64
        self.steps = 0

    def _run(self, toks, src, tgt):
        Tm, L, H, dk = self.Tm, self.L, self.H, self.dk
        a = np.ones((1, S), np.int32); seq = [self.lang[CODES[src]]] + toks + [2]; a[0, :len(seq)] = seq; lang = self.lang[CODES[tgt]]
        ck, cv, bias = [np.asarray(x) for x in self.enc(src_ids=a)]
        sk = np.zeros((L, H, Tm, dk), np.float32); sv = sk.copy(); tok = 2; out = []
        for t in range(Tm - 1):
            oh = np.zeros(Tm, np.float32); oh[t] = 1.0; mask = np.where(np.arange(Tm) <= t, 0.0, -1e4).astype(np.float32)
            nxt, sk, sv = self.dec(tok=np.array([tok], np.int32), oh=oh, mask=mask, sk=sk, sv=sv, ck=ck, cv=cv, bias=bias)
            nxt = int(np.asarray(nxt).reshape(-1)[0]); sk = np.asarray(sk); sv = np.asarray(sv); self.steps += 1
            tok = lang if t == 0 else nxt; out.append(tok)
            if t > 0 and nxt == 2: break
        self.calls += 1; body = out[1:]
        return body, (2 in body)
