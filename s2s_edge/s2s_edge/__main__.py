# -*- coding: utf-8 -*-
"""Command line: `python -m s2s_edge file|live --src vi --tgt en`."""
import argparse, os, sys
import numpy as np
from .audio import load_audio, save_wav, listen, play, concat_results
from .registry import Registry, DIRECTIONS
from .pipeline import Translator


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="s2s_edge", description="Offline speech-to-speech translation (vi <-> en/zh/ko)")
    ap.add_argument("cmd", choices=["file", "live"])
    ap.add_argument("--src", choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--tgt", choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--pair", help="bidirectional conversation between two languages, e.g. vi,ko (language of each utterance is detected)")
    ap.add_argument("--auto", action="store_true", help="Vietnamese + any of en/zh/ko, partner language detected per utterance")
    ap.add_argument("--partner", default="en", help="first partner language in --auto mode")
    ap.add_argument("--glossary", default=None, help="JSON {\"Name\": {\"vi\": ..., \"zh\": ..., \"ko\": ...}} overriding name transliteration")
    ap.add_argument("--models", default=os.environ.get("S2S_MODELS", "models/models.json"))
    ap.add_argument("--backend", default=os.environ.get("S2S_BACKEND", "npu"), choices=["npu", "cpu"])
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--no-play", action="store_true")
    a = ap.parse_args()
    reg = Registry(a.models, a.backend, a.glossary)
    if a.pair or a.auto:
        from .conversation import Conversation
        pr = tuple(a.pair.split(",")) if a.pair else None
        if pr and "vi" not in pr: sys.exit("one language of the pair must be vi")
        tr = Conversation(reg, pair=pr, auto=a.auto, partner=a.partner); label = "auto" if a.auto else a.pair
    else:
        if not a.src or not a.tgt or (a.src, a.tgt) not in DIRECTIONS: sys.exit(f"give --src/--tgt (one side must be vi), or --pair / --auto. Supported: {DIRECTIONS}")
        tr = Translator(reg, a.src, a.tgt); label = f"{a.src}->{a.tgt}"
    if a.cmd == "file":
        res = tr.translate(load_audio(a.inp))
        for r in res:
            t = r["timing"]
            print(f"[{r['span'][0]:.1f}-{r['span'][1]:.1f}s] {r.get('src', a.src)}->{r.get('tgt', a.tgt)}: {r['asr']}\n   -> {r['mt']}\n   ({t['total_s'] * 1000:.0f} ms, graphs {t['graph_s'] * 1000:.0f} ms)")
        wav, sr = concat_results(res); save_wav(a.out, wav, sr)
        print("saved", a.out)
    else:
        print(f"Live {label}. Enter = speak, Ctrl+C = quit.")
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
