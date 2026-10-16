# -*- coding: utf-8 -*-
"""Command line: `python -m s2s_edge file|live --src vi --tgt en`."""
import argparse, os, sys
import numpy as np
from .audio import load_audio, save_wav, listen, play
from .registry import Registry, DIRECTIONS
from .pipeline import Translator


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="s2s_edge", description="Offline speech-to-speech translation (vi <-> en/zh/ko)")
    ap.add_argument("cmd", choices=["file", "live"])
    ap.add_argument("--src", required=True, choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--tgt", required=True, choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--models", default=os.environ.get("S2S_MODELS", "models/models.json"))
    ap.add_argument("--backend", default=os.environ.get("S2S_BACKEND", "npu"), choices=["npu", "cpu"])
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--no-play", action="store_true")
    a = ap.parse_args()
    if (a.src, a.tgt) not in DIRECTIONS:
        sys.exit(f"Unsupported direction {a.src}->{a.tgt}; one side must be vi. Supported: {DIRECTIONS}")
    tr = Translator(Registry(a.models, a.backend), a.src, a.tgt)
    if a.cmd == "file":
        res = tr.translate(load_audio(a.inp))
        wav = np.concatenate([r["wav"] for r in res]) if res else np.zeros(0, np.float32)
        for r in res:
            t = r["timing"]
            print(f"[{r['span'][0]:.1f}-{r['span'][1]:.1f}s] {r['asr']}\n   -> {r['mt']}\n   ({t['total_s'] * 1000:.0f} ms, graphs {t['graph_s'] * 1000:.0f} ms)")
        save_wav(a.out, wav, res[0]["sr"] if res else 16000)
        print("saved", a.out)
    else:
        print(f"Live {a.src}->{a.tgt}. Enter = speak, Ctrl+C = quit.")
        while True:
            input("\n[Enter] to speak > ")
            print("listening...")
            w = listen()
            if len(w) < 4800:
                print("(nothing heard)")
                continue
            for r in tr.translate(w):
                t = r["timing"]
                print(f"  heard : {r['asr']}\n  output: {r['mt']}\n  {t['total_s'] * 1000:.0f} ms (ASR {t['asr_s'] * 1000:.0f} | MT {t['mt_s'] * 1000:.0f} | TTS {t['tts_s'] * 1000:.0f}), graphs {t['graph_s'] * 1000:.0f} ms")
                if len(r["wav"]) and not a.no_play:
                    play(r["wav"], r["sr"])


if __name__ == "__main__":
    main()
