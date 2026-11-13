"""Tiny zero-dependency web server for the S2S demo.

    python web_demo/server.py            # http://localhost:8000
    python web_demo/server.py --port 9000

Endpoints
    GET  /api/health                         engine name + supported languages
    POST /api/translate?src=vi&tgt=en        body = 16 kHz mono WAV bytes
         -> streams NDJSON, one event per stage as it finishes:
            {"stage":"asr","text":...,"ms":...}
            {"stage":"mt","text":...,"ms":...}
            {"stage":"tts","audio":<base64 wav>,"ms":...}
            {"stage":"done","total_ms":...,"input_sec":...}
            {"stage":"error","message":...}   (on failure)
"""
import argparse
import base64
import io
import json
import mimetypes
import os
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from engine import LANGS, PAIRS, get_engine

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 30 * 1024 * 1024
ENGINE = get_engine()
LOCK = threading.Lock()  # models are usually not thread-safe; serialize requests


def wav_seconds(data):
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        print("%s  %s" % (self.address_string(), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"engine": ENGINE.name, "langs": list(LANGS), "pairs": [list(p) for p in PAIRS]})
        if path == "/":
            path = "/index.html"
        fp = os.path.normpath(os.path.join(STATIC, path.lstrip("/")))
        if not fp.startswith(STATIC) or not os.path.isfile(fp):
            return self._json(404, {"error": "not found"})
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/translate":
            return self._json(404, {"error": "not found"})
        q = parse_qs(url.query)
        src, tgt = q.get("src", [""])[0], q.get("tgt", [""])[0]
        if (src, tgt) not in PAIRS:
            return self._json(400, {"error": f"unsupported pair {src}->{tgt}"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._json(400, {"error": "empty or too large body"})
        audio = self.rfile.read(n)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        t0 = time.perf_counter()
        try:
            with LOCK:
                t = time.perf_counter()
                text = ENGINE.asr(audio, src)
                emit({"stage": "asr", "text": text, "ms": round((time.perf_counter() - t) * 1000)})

                t = time.perf_counter()
                out = ENGINE.mt(text, src, tgt)
                emit({"stage": "mt", "text": out, "ms": round((time.perf_counter() - t) * 1000)})

                t = time.perf_counter()
                wav = ENGINE.tts(out, tgt)
                emit({"stage": "tts", "audio": base64.b64encode(wav).decode(),
                      "ms": round((time.perf_counter() - t) * 1000)})

            emit({"stage": "done", "total_ms": round((time.perf_counter() - t0) * 1000),
                  "input_sec": wav_seconds(audio)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                emit({"stage": "error", "message": str(e)})
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"OneVoice S2S demo  ->  http://{a.host}:{a.port}   (engine: {ENGINE.name})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
