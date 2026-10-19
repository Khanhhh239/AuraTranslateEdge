# -*- coding: utf-8 -*-
"""Golden-vector check of every compiled graph.
  make:  python tools/golden.py make  --models models.dev.json            (dev machine, CPU/ONNX reference -> golden/<graph>.npz)
  check: python tools/golden.py check --models models/models.json --backend npu   (device: runs each graph on the golden inputs, compares to the CPU reference)
A graph PASSES when float outputs have cosine >= 0.99 and integer outputs match on >= 95 % of elements (argmax/ids). Takes seconds, tells which graph is wrong."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry, ASSETS
from s2s_edge.audio import load_audio
from s2s_edge.asr import _fbank, LID
from s2s_edge.mt import Nllb, S

HERE = os.path.dirname(os.path.abspath(__file__)); GOLD = os.path.join(os.path.dirname(HERE), "golden")


def inputs_for(reg, key):
    """Deterministic example inputs for graph `key` (only needs assets + one wav)."""
    wav_vi = os.path.join(os.path.dirname(HERE), "samples", "vi_0.wav"); wav_en = os.path.join(os.path.dirname(HERE), "samples", "en_0.wav")
    if key == "asr_vi":
        f = _fbank(load_audio(wav_vi)[:14 * 16000]); x = np.zeros((1, 1500, 80), np.float32); n = min(len(f), 1500); x[0, :n] = f[:n]; return dict(x=x, x_lens=np.array([n], np.int64))
    if key in ("asr_xx", "asr_xx_frontend"):
        w = load_audio(wav_en); b = np.zeros((1, 464000), np.float32); n = min(len(w), 464000); b[0, :n] = w[:n]
        d = dict(wav=b, wav_len=np.array([n], np.int32))
        return {**d, "language": np.array([LID["en"]], np.int32), "textnorm": np.array([14], np.int32)} if key == "asr_xx" else d
    if key == "mt":
        nl = Nllb(None, ASSETS); toks = nl.encode("Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé."); a = np.ones((1, S), np.int32); seq = [nl.lang["vie_Latn"]] + toks + [2]; a[0, :len(seq)] = seq
        return dict(src_ids=a, tgt_lang=np.array([nl.lang["eng_Latn"]], np.int32))
    if key == "tts_vi":
        b = "Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé.".encode("utf-8"); x = np.zeros((1, 512), np.int32); x[0, :len(b)] = list(b); return dict(text_bytes=x)
    if key == "tts_zh":
        b = "今天天气很好，我们一起去公园散步。".encode("utf-8"); x = np.zeros((1, 112), np.int32); x[0, :len(b)] = list(b); return dict(text_bytes=x)
    if key == "tts_enko":
        from s2s_edge.tts import TtsEnKo
        t = TtsEnKo(None, "en", ASSETS); s = t._clean(t.p._preprocess_text("Please wear your helmet before entering the factory.", "en")); cp = np.zeros((1, 96), np.int32); cp[0, :len(s)] = [ord(c) for c in s]
        return dict(cp=cp, style_dp=t.dp, style_ttl=t.ttl)
    if key in ("mt_enc", "mt_dec", "mt_decB"):
        S_ = reg.cfg["mt_enc"].get("S", 64); Tm = reg.cfg.get("mt_dec", reg.cfg.get("mt_decB", {})).get("Tm", 96); B = reg.cfg.get("mt_decB", {}).get("B", 4)
        nl = Nllb(None, ASSETS); toks = nl.encode("Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé."); a = np.ones((1, S_), np.int32); seq = [nl.lang["vie_Latn"]] + toks + [2]; a[0, :len(seq)] = seq
        if key == "mt_enc": return dict(src_ids=a)
        ck, cv, bias = [np.asarray(x) for x in reg.graph("mt_enc")(src_ids=a)]
        nb = B if key == "mt_decB" else None; shp = (12, nb, 16, Tm, 64) if nb else (12, 16, Tm, 64); oh = np.zeros(Tm, np.float32); oh[0] = 1; mask = np.where(np.arange(Tm) <= 0, 0.0, -1e4).astype(np.float32)
        return dict(tok=np.array([2] * (nb or 1), np.int32), oh=oh, mask=mask, sk=np.zeros(shp, np.float32), sv=np.zeros(shp, np.float32), ck=ck, cv=cv, bias=bias)
    raise KeyError(key)


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size != b.size: return "size", float("nan")
    if np.issubdtype(b.dtype, np.integer): return "int", float((a.reshape(-1).astype(np.int64) == b.reshape(-1).astype(np.int64)).mean())
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64); return "cos", float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["make", "check"]); ap.add_argument("--models", default="models/models.json"); ap.add_argument("--backend", default="npu"); ap.add_argument("--only", default="")
    a = ap.parse_args(); os.makedirs(GOLD, exist_ok=True); reg = Registry(a.models, "cpu" if a.mode == "make" else a.backend)
    keys = [k for k in a.only.split(",") if k] or [k for k in reg.cfg if k in ("asr_vi", "asr_xx", "asr_xx_frontend", "mt", "mt_enc", "mt_dec", "mt_decB", "tts_vi", "tts_zh", "tts_enko")]; rows = []
    for k in keys:
        g = reg.graph(k); gp = os.path.join(GOLD, f"{k}.npz")
        if a.mode == "check" and os.path.exists(gp): z = np.load(gp); ins = {n[3:]: z[n] for n in z.files if n.startswith("in_")}          # exactly the inputs used for the reference
        else: ins = inputs_for(reg, k)
        t = time.perf_counter(); out = g(**ins); dt = time.perf_counter() - t
        if a.mode == "make":
            np.savez_compressed(gp, **{f"in_{n}": v for n, v in ins.items()}, **{f"out_{i}": np.asarray(o) for i, o in enumerate(out)}); print("saved", gp, f"{os.path.getsize(gp)/1e6:.1f} MB"); continue
        ref = np.load(gp); status = []
        for i, o in enumerate(out):
            kind, v = compare(o, ref[f"out_{i}"]); status.append((i, kind, round(v, 5)))
        ok = all((k_ == "cos" and v >= 0.99) or (k_ == "int" and v >= 0.95) for _, k_, v in status)
        rows.append(dict(graph=k, ok=ok, outputs=status, seconds=round(dt, 3))); print(("PASS " if ok else "FAIL ") + k, status, f"{dt*1000:.0f} ms", flush=True)
    if a.mode == "check":
        os.makedirs("logs", exist_ok=True); json.dump(rows, open(f"logs/golden_{a.backend}_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1); print("all pass" if all(r["ok"] for r in rows) else "SOME FAILED")


if __name__ == "__main__": main()
