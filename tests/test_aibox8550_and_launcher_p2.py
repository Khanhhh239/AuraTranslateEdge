import io
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "s2s_edge") not in sys.path:
    sys.path.insert(0, str(ROOT / "s2s_edge"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from s2s_edge import backends, hybrid_textnorm, identifier_tagger, neural_textnorm
import s2s_web
import start_s2s_web


def _make_wav_bytes(seconds=0.5, sr=16000):
    buf = io.BytesIO()
    n = int(seconds * sr)
    samples = (np.sin(np.linspace(0, 8 * np.pi, n)) * 3000).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class FakeProc:
    def __init__(self, pid=4242, poll_seq=None):
        self.pid = pid
        self._poll_seq = list(poll_seq or [])
        self.terminated = False

    def poll(self):
        if self._poll_seq:
            return self._poll_seq.pop(0)
        return None

    def terminate(self):
        self.terminated = True


class LauncherP2AndAiboxTests(unittest.TestCase):
    def test_spawn_refuses_foreign_pid_and_preserves_server_pid(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "web"
            out_dir.mkdir()
            pid_file = out_dir / "server.pid"
            pid_file.write_text("1111")
            log_file = Path(td) / "web.log"

            args = start_s2s_web.parse_args(["--port", "18766", "--out-dir", str(out_dir), "--log-file", str(log_file)])
            fp = {"s2s_edge_sha256": "a", "server_files_sha256": "b", "git_head": "c"}
            req_profile = start_s2s_web.requested_profile_from_env(
                start_s2s_web.build_launch_env(base_env={})
            )
            foreign_health = {
                "ok": True,
                "pid": 99999,
                "pending": 0,
                **req_profile,
                "effective_profile": dict(req_profile),
                "loaded_code_fingerprint": fp,
            }
            calls = [None, foreign_health]
            proc = FakeProc(pid=4242)

            with self.assertRaises(start_s2s_web.LauncherError) as ctx:
                start_s2s_web.launch_or_reuse(
                    args,
                    base_env={},
                    health_fn=lambda _u: calls.pop(0) if calls else foreign_health,
                    fingerprint_fn=lambda: fp,
                    popen_fn=lambda *a, **kw: proc,
                    sleep_fn=lambda _s: None,
                )

            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("Foreign server", str(ctx.exception))
            self.assertTrue(proc.terminated)
            self.assertEqual(pid_file.read_text(), "1111")

    def test_spawn_checks_child_exit_before_accepting_foreign_health(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "web"
            out_dir.mkdir()
            pid_file = out_dir / "server.pid"
            pid_file.write_text("1111")
            log_file = Path(td) / "web.log"
            args = start_s2s_web.parse_args(["--port", "18766", "--out-dir", str(out_dir), "--log-file", str(log_file)])
            fp = {"s2s_edge_sha256": "a", "server_files_sha256": "b", "git_head": "c"}
            proc = FakeProc(pid=4242, poll_seq=[1])

            calls = [None, {"ok": True, "pid": 88888}]
            with self.assertRaises(start_s2s_web.LauncherError) as ctx:
                start_s2s_web.launch_or_reuse(
                    args,
                    base_env={},
                    health_fn=lambda _u: calls.pop(0) if calls else {"ok": True, "pid": 88888},
                    fingerprint_fn=lambda: fp,
                    popen_fn=lambda *a, **kw: proc,
                    sleep_fn=lambda _s: None,
                )

            self.assertEqual(ctx.exception.code, 5)
            self.assertIn("exited", str(ctx.exception))
            self.assertEqual(pid_file.read_text(), "1111")

    def test_spawn_refuses_mismatched_effective_profile_and_preserves_pid(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "web"
            out_dir.mkdir()
            pid_file = out_dir / "server.pid"
            pid_file.write_text("1111")
            log_file = Path(td) / "web.log"
            args = start_s2s_web.parse_args(
                ["--port", "18766", "--out-dir", str(out_dir), "--log-file", str(log_file), "--source-mode", "1"]
            )
            fp = {"s2s_edge_sha256": "a", "server_files_sha256": "b", "git_head": "c"}
            stale_effective = {
                "ok": True,
                "pid": 4242,
                "pending": 0,
                "source_mode": "1",
                "vad_profile": "legacy",
                "ort_policy": "auto",
                "tts_format": "hybrid",
                "effective_profile": {
                    "source_mode": "0",
                    "vad_profile": "legacy",
                    "ort_policy": "auto",
                    "tts_format": "hybrid",
                },
                "loaded_code_fingerprint": fp,
            }
            calls = [None, stale_effective]
            proc = FakeProc(pid=4242)

            with self.assertRaises(start_s2s_web.LauncherError) as ctx:
                start_s2s_web.launch_or_reuse(
                    args,
                    base_env={},
                    health_fn=lambda _u: calls.pop(0) if calls else stale_effective,
                    fingerprint_fn=lambda: fp,
                    popen_fn=lambda *a, **kw: proc,
                    sleep_fn=lambda _s: None,
                )

            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("failed readiness verification", str(ctx.exception))
            self.assertTrue(proc.terminated)
            self.assertEqual(pid_file.read_text(), "1111")

    def test_spawn_writes_pid_only_after_ownership_and_npu_args(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "web"
            log_file = Path(td) / "web.log"
            models_json = Path(td) / "models_qcs8550.json"
            models_json.write_text("{}")

            args = start_s2s_web.parse_args(
                [
                    "--port",
                    "18766",
                    "--out-dir",
                    str(out_dir),
                    "--log-file",
                    str(log_file),
                    "--backend",
                    "npu",
                    "--models",
                    str(models_json),
                    "--keep-loaded",
                ]
            )
            fp = {"s2s_edge_sha256": "a", "server_files_sha256": "b", "git_head": "c"}
            req_profile = start_s2s_web.requested_profile_from_env(
                start_s2s_web.build_launch_env(base_env={}, backend="npu", models=str(models_json))
            )
            self.assertEqual(req_profile["source_mode"], "0")
            matching = {
                "ok": True,
                "pid": 5151,
                "pending": 0,
                "backend": "npu",
                "models": str(models_json.resolve()),
                **req_profile,
                "effective_profile": dict(req_profile),
                "loaded_code_fingerprint": fp,
            }
            calls = [None, matching]
            captured = {}

            def fake_popen(cmd, **kw):
                captured["cmd"] = list(cmd)
                captured["env"] = dict(kw["env"])
                return FakeProc(pid=5151)

            res = start_s2s_web.launch_or_reuse(
                args,
                base_env={},
                health_fn=lambda _u: calls.pop(0),
                fingerprint_fn=lambda: fp,
                popen_fn=fake_popen,
                sleep_fn=lambda _s: None,
            )

            self.assertEqual(res["action"], "started")
            self.assertEqual(res["pid"], 5151)
            self.assertEqual((out_dir / "server.pid").read_text(), "5151")
            self.assertIn("--backend", captured["cmd"])
            self.assertIn("npu", captured["cmd"])
            self.assertIn("--models", captured["cmd"])
            self.assertIn("--keep-loaded", captured["cmd"])
            self.assertEqual(captured["env"]["S2S_NEURAL_TN"], "0")

    def test_default_source_mode_and_explicit_linux_guard(self):
        self.assertEqual(neural_textnorm.default_source_mode("cpu", "darwin"), "1")
        self.assertEqual(neural_textnorm.default_source_mode("npu", "darwin"), "0")
        self.assertEqual(neural_textnorm.default_source_mode("cpu", "linux"), "0")
        self.assertEqual(neural_textnorm.default_source_mode("npu", "linux"), "0")
        self.assertEqual(start_s2s_web.build_launch_env({}, backend="npu", source_mode="1")["S2S_NEURAL_TN"], "1")

        with self.assertRaises(RuntimeError) as ctx:
            neural_textnorm.check_neural_tn_available(platform_name="linux", check_model_exists=False)
        self.assertIn("S2S_NEURAL_TN=0", str(ctx.exception))

    def test_find_crf_model_packaged_assets_and_env_override(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            (pkg / "assets").mkdir(parents=True)
            crf = pkg / "assets/identifier.crfsuite"
            crf.write_bytes(b"dummy")
            found = identifier_tagger.find_crf_model([pkg])
            self.assertEqual(found, crf)

            override = Path(td) / "custom.crfsuite"
            override.write_bytes(b"override")
            with patch.dict(os.environ, {"S2S_CRF_MODEL": str(override)}, clear=False):
                self.assertEqual(identifier_tagger.find_crf_model([pkg]), override)

        missing = Path("/nonexistent/identifier.crfsuite")
        if hasattr(identifier_tagger, "IdentifierTagger"):
            try:
                import pycrfsuite  # noqa: F401
            except ImportError:
                pass
            else:
                with self.assertRaises(FileNotFoundError):
                    identifier_tagger.IdentifierTagger(missing)

    def test_s2s_web_service_npu_keep_loaded_and_multilingual_submit(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "web_jobs"
            models_json = Path(td) / "models_npu.json"
            models_json.write_text("{}")

            clean_env = {k: v for k, v in os.environ.items() if k not in ("S2S_NEURAL_TN", "S2S_BACKEND", "S2S_MODELS")}
            with patch.dict(os.environ, clean_env, clear=True):
                svc = s2s_web.Service(out_dir, backend="npu", models=str(models_json))
                self.assertEqual(svc.profile["source_mode"], "0")
                self.assertEqual(svc.backend, "npu")
                self.assertTrue(svc.keep_loaded)

                created_regs = []
                released_keys = []

                class DummyReg:
                    def __init__(self, path, backend, **_kw):
                        self.path = path
                        self.backend = backend
                        self._cache = {"asr": object(), "mt": object(), "tts": object()}
                        created_regs.append(self)

                    def release(self, k):
                        released_keys.append(k)
                        self._cache.pop(k, None)

                    def graph_telemetry(self):
                        return {"backend": self.backend}

                class DummyCallable:
                    def __init__(self, fn):
                        self.fn = fn
                        self.calls = 0

                    def __call__(self, *a):
                        self.calls += 1
                        return self.fn(*a)

                    def synthesize(self, text):
                        self.calls += 1
                        return self.fn(text)

                    def translate_atomic(self, *a):
                        self.calls += 1
                        return self.fn(*a)

                base_asr = DummyCallable(lambda _w: "xin chào")
                base_asr.max_s = 12.0
                base_mt = DummyCallable(lambda t, *_a: f"hello {t}")
                base_tts = DummyCallable(lambda text: (np.ones(1600, dtype=np.float32), 16000))

                class DummyTranslator:
                    max_s = 12.0

                    def __init__(self, reg, src, tgt, **_kw):
                        self.reg = reg
                        self.src = src
                        self.tgt = tgt
                        self.asr = base_asr
                        self.tts = base_tts
                        self.mt = base_mt
                        self.e = type(
                            "DummyEngine",
                            (),
                            {
                                "_asr": {src: base_asr},
                                "_mt": base_mt,
                                "_tts": {tgt: base_tts},
                                "neural_enabled": staticmethod(lambda: False),
                            },
                        )()

                    def translate_segment(self, wav, **_kw):
                        raw_asr = self.e._asr[self.src](wav)
                        mt_txt = self.e._mt(raw_asr)
                        wav_out, sr = self.e._tts[self.tgt].synthesize(mt_txt)
                        return {"asr": raw_asr, "mt": mt_txt, "tts_text": mt_txt, "wav": wav_out, "sr": sr, "timing": {}}

                wav = _make_wav_bytes(0.8, 16000)
                with patch.object(svc.pool, "submit", lambda *_a, **_kw: None):
                    k1 = svc.submit(wav, "vi", tgt="ko")
                    k2 = svc.submit(wav, "vi", tgt="ko")
                    self.assertEqual(svc.snapshot(k1)["tgt"], "ko")
                    with self.assertRaises(ValueError):
                        svc.submit(wav, "vi", tgt="vi")

                with patch("s2s_edge.registry.Registry", DummyReg), patch("s2s_edge.pipeline.Translator", DummyTranslator), patch(
                    "s2s_edge.audio.segment", lambda w, **_kw: ([(0, len(w))], [[]])
                ):
                    svc.run(k1)
                    svc.run(k2)

                self.assertEqual(len(created_regs), 1)
                self.assertEqual(released_keys, [])
                self.assertIs(svc._translators[("vi", "ko")].e._asr["vi"], base_asr)
                self.assertEqual(base_asr.calls, 2)
                snap2 = svc.snapshot(k2)
                self.assertEqual(snap2["status"], "done", snap2)
                self.assertEqual(snap2["execution_backend"], "npu")
                self.assertIn("Hexagon NPU", snap2["backend"])
                svc.close()
                svc.pool.shutdown(wait=False, cancel_futures=True)

    def test_run_e2e_accepts_models_and_backend_flags(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_e2e.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--models", proc.stdout)
        self.assertIn("--backend", proc.stdout)

    def test_ort_optimization_plan_policies(self):
        _, plan_large = backends.ort_optimization_plan("auto", large_or_ext=True)
        self.assertEqual([name for name, _ in plan_large], ["ORT_DISABLE_ALL"])
        _, plan_small = backends.ort_optimization_plan("auto", large_or_ext=False)
        self.assertEqual([name for name, _ in plan_small], ["ORT_ENABLE_BASIC", "ORT_DISABLE_ALL"])

    def test_build_package_crf_and_wheelhouse_validation(self):
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            wh = Path(td) / "wheelhouse"
            wh.mkdir()
            for name in (
                "numpy-1.26.4-cp312-cp312-manylinux_2_17_aarch64.whl",
                "scipy-1.13.0-cp312-cp312-manylinux_2_17_aarch64.whl",
                "soundfile-0.12.1-py2.py3-none-any.whl",
                "sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_aarch64.whl",
                "kaldi_native_fbank-1.19.2-cp312-cp312-manylinux_2_17_aarch64.whl",
                "onnxruntime-1.18.0-cp312-cp312-manylinux_2_17_aarch64.whl",
                "psutil-5.9.8-cp312-cp312-linux_aarch64.whl",
                "qai_appbuilder-2.51.40-cp312-cp312-linux_aarch64.whl",
            ):
                (wh / name).write_bytes(b"")

            out_pkg = Path(td) / "pkg"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "s2s_edge/tools/build_package.py"),
                    "--no-models",
                    "--strict-wheelhouse",
                    "--wheelhouse",
                    str(wh),
                    "--out",
                    str(out_pkg),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            install_sh = (out_pkg / "install.sh").read_text(encoding="utf-8")
            run_sh = (out_pkg / "run_demo.sh").read_text(encoding="utf-8")
            self.assertIn("optional wheel unavailable", install_sh)
            self.assertIn("S2S_NEURAL_TN=", run_sh)
            self.assertIn("S2S_TTS_FORMAT=", run_sh)
            if (ROOT / "models/normalization/identifier.crfsuite").is_file():
                self.assertTrue((out_pkg / "assets/identifier.crfsuite").is_file())


if __name__ == "__main__":
    unittest.main()
