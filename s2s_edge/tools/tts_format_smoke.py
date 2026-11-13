"""Portable actual-graph TTS smoke test; no microphone, ASR or MT involved."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = {
    'vi': 'Hẹn lúc 3:30. Mã ZIP 04782.',
    'en': 'The meeting is at 3:30 PM. My ZIP code is 04782.',
    'ko': '회의는 3:30에 있습니다. 우편번호는 04782입니다.',
    'zh': '会议在3:30开始。邮政编码是04782。',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True, help='Existing model registry JSON; graph paths relative to that JSON')
    p.add_argument('--out', type=Path, default=ROOT/'logs/tts_format_smoke')
    p.add_argument('--lang', choices=['all']+list(CASES), default='all')
    args = p.parse_args()
    os.environ['S2S_TTS_FORMAT'] = 'hybrid'
    import numpy as np
    from s2s_edge.registry import Registry
    from s2s_edge.audio import save_wav
    reg = Registry(str(args.models.resolve()), 'cpu')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang, text in CASES.items():
        if args.lang not in ('all', lang): continue
        model = reg.tts(lang); start = time.perf_counter()
        audio, sr = model.synthesize(text)
        if not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError('Empty/non-finite audio: '+lang)
        target = args.out/(lang+'.wav'); save_wav(str(target), audio, sr)
        row = dict(lang=lang, input=text, tts_text=model.tts_input_text,
                   trace=model.format_trace, wav=target.name, sample_rate=sr,
                   output_seconds=len(audio)/sr, synthesis_seconds=time.perf_counter()-start,
                   sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        rows.append(row); print(lang, 'OK', row['tts_text'], flush=True)
    (args.out/'results.json').write_text(json.dumps({'scope':'real CPU TTS only; synthetic text; no human listening score', 'cases':rows},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
