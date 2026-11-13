# OneVoice — S2S web demo (new UI)

Zero-dependency (Python stdlib only) web UI for the ASR → MT → TTS pipeline.
Currently runs on a **mock engine**; real models plug in via `engine.py`.
Separate from `web/` (the older UI served by `tools/s2s_web.py` on port 8766).

Supported directions: vi↔en, vi↔zh, vi↔ko (the target list only shows languages valid for the chosen source).

```bash
python3 web_demo/server.py          # http://127.0.0.1:8000
```

- Mic: tap to record / tap again to translate, or hold **Space** (push-to-talk), **Esc** cancels.
- Upload or drag-drop any audio file. Browser resamples to 16 kHz mono PCM16 WAV before sending.
- Each stage streams back as it finishes (NDJSON), so the UI shows ASR → MT → TTS progress + per-stage ms.

## Plugging in real models

Everything model-related lives in [engine.py](engine.py). Implement a class with:

```python
name: str
asr(wav: bytes, lang) -> str          # 16 kHz mono WAV bytes
mt(text, src, tgt) -> str
tts(text, lang) -> bytes              # full WAV file
```

and return it from `get_engine()`. The classes in `src/step5_pipeline/pipeline_s2s.py` can be wrapped directly.
Requests are serialized with a lock in `server.py`, so models don't need to be thread-safe.
