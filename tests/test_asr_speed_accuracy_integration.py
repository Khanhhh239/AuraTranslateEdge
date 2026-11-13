"""Integration and unit tests for the merged `asr-speed-accuracy` enhancements."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S2S_DIR = ROOT / "s2s_edge"
ASSETS_DIR = S2S_DIR / "assets"
if str(S2S_DIR) not in sys.path:
    sys.path.insert(0, str(S2S_DIR))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from s2s_edge import backends
from s2s_edge.asr import (
    AsrVi,
    AsrXx,
    LID,
    Transducer,
    _word_ends,
    fix_itn,
    load_cmvn,
    pick,
    sv_features,
    windows,
    zip_outputs,
)
from s2s_edge.audio import SR, active_rms, get_vad_options, load_audio, normalize_level, segment
from s2s_edge.conversation import Conversation, Engine, _align_breaks, _break_positions
from s2s_edge.pipeline import Translator
from s2s_edge.registry import ASSETS, Registry, resolve_runtime_profile
from s2s_edge.textnorm import vi_punct
import ablate_profiles
import evaluate_asr_integration
import start_s2s_web


def _tone(hz=220.0, dur_s=1.0, amp=0.1, sr=SR):
    t = np.arange(int(dur_s * sr), dtype=np.float32) / sr
    return (amp * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)


class AsrSpeedAccuracyUnitTests(unittest.TestCase):
    def test_upstream_host_asr_script_suite(self):
        env = dict(os.environ, S2S_VAD_PROFILE="hybrid", PYTHONPATH=str(S2S_DIR))
        proc = subprocess.run(
            [sys.executable, str(S2S_DIR / "tools/test_asr_host.py")],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("checks passed", proc.stdout)

    def test_sv_features_cmvn_and_fix_itn_guards(self):
        cmvn_path = ASSETS_DIR / "sensevoice_am.mvn"
        self.assertTrue(cmvn_path.exists())
        cmvn = load_cmvn(str(cmvn_path))
        self.assertEqual(cmvn[0].shape, (560,))
        self.assertEqual(cmvn[1].shape, (560,))

        wav = np.concatenate([np.zeros(1600, np.float32), _tone(dur_s=1.8, amp=0.08)])
        feats = sv_features(wav, cmvn)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[1], 560)
        self.assertTrue(np.isfinite(feats).all())
        self.assertGreater(len(feats), 20)

        self.assertEqual(fix_itn("Meet me at530 AM tomorrow.", "en"), "Meet me at 5:30 AM tomorrow.")
        self.assertEqual(fix_itn("Call from930am to 11am.", "en"), "Call from 9:30 am to 11am.")
        self.assertEqual(fix_itn("Wait until at 330 pm.", "en"), "Wait until at 3:30 pm.")
        # Guard against false positives
        self.assertEqual(fix_itn("1530 PM", "en"), "1530 PM")
        self.assertEqual(fix_itn("Draw 530 amps at Kat5 cable.", "en"), "Draw 530 amps at Kat5 cable.")
        self.assertEqual(fix_itn("Meet at530 AM.", "vi"), "Meet at530 AM.")

    def test_windows_pick_and_zip_outputs_permuted(self):
        ws = windows([(500, "g500"), (300, "g300"), (1200, "g1200")], 1500)
        self.assertEqual([w for w, _ in ws], [300, 500, 1200])
        self.assertEqual(pick(ws, 250), (300, "g300"))
        self.assertEqual(pick(ws, 300), (300, "g300"))
        self.assertEqual(pick(ws, 301), (500, "g500"))
        self.assertEqual(pick(ws, 5000), (1200, "g1200"))

        lg = np.zeros((1, 73, 2000), np.float32)
        el = np.array([70], np.int64)
        eo = np.ones((1, 73, 512), np.float32)
        p_lg, p_el, p_eo = zip_outputs([eo, el, lg], 300)
        self.assertEqual(p_lg.shape, (1, 73, 2000))
        self.assertEqual(int(p_el.reshape(-1)[0]), 70)
        self.assertEqual(p_eo.shape, (73, 512))

    def test_active_rms_normalize_and_hybrid_vad_profiles_and_breaks(self):
        self.assertIn("hybrid", get_vad_options("hybrid") and ("legacy", "upstream", "hybrid"))
        self.assertEqual(resolve_runtime_profile(vad_profile=None)["vad_profile"], "legacy")
        self.assertEqual(resolve_runtime_profile(vad_profile="hybrid")["vad_profile"], "hybrid")

        # CLI parsers accept hybrid without error
        self.assertEqual(start_s2s_web.parse_args(["--vad-profile", "hybrid"]).vad_profile, "hybrid")
        self.assertEqual(evaluate_asr_integration.build_parser().parse_args(["--run-id", "x", "--vad-profile", "hybrid"]).vad_profile, "hybrid")
        self.assertEqual(ablate_profiles.build_parser().parse_args(["vad", "--run-id", "x", "--fixed-vad", "hybrid"]).fixed_vad, "hybrid")

        # active_rms & normalize_level handle quiet, normal, loud, and transient click
        sil = np.zeros(SR, np.float32)
        self.assertEqual(active_rms(sil), 0.0)
        self.assertTrue(np.array_equal(normalize_level(sil), sil))

        quiet = np.concatenate([sil, _tone(amp=0.002, dur_s=1.0), sil])
        loud = _tone(amp=0.6, dur_s=1.0)
        self.assertGreater(active_rms(normalize_level(quiet)), active_rms(quiet) * 5.0)
        self.assertLess(active_rms(normalize_level(loud)), active_rms(loud))

        # hybrid VAD merges 0.45s gap & emits pause midpoint break; split mode separates it
        w = np.concatenate([
            np.zeros(int(0.4 * SR), np.float32),
            _tone(dur_s=0.9, amp=0.12),
            np.zeros(int(0.45 * SR), np.float32),
            _tone(dur_s=0.9, amp=0.12),
            np.zeros(int(0.4 * SR), np.float32),
        ])
        spans_h, br_h = segment(w, vad_profile="hybrid", return_breaks=True)
        self.assertEqual(len(spans_h), 1)
        self.assertEqual(len(br_h), 1)
        self.assertEqual(len(br_h[0]), 1)
        with patch.dict(os.environ, {"S2S_VAD_MODE": "split"}):
            spans_s = segment(w, vad_profile="hybrid")
            self.assertEqual(len(spans_s), 2)
            self.assertLessEqual(spans_s[0][1], spans_s[1][0])

    def test_vi_punct_breaks_and_alignment_preserve_numeric_guards(self):
        src_words = ["tôi", "không", "đi", "hôm", "nay", "số", "0", "hay", "-5"]
        breaks = _break_positions([int(1.9 * SR)], [15, 28, 42, 60, 82, 95, 110, 125, 140])
        self.assertEqual(breaks, [3])
        aligned = _align_breaks(src_words, ["tôi", "không", "đi", "hôm", "nay", "số", "0", "hay", "-5"], breaks)
        out = vi_punct("tôi không đi hôm nay số 0 hay -5", breaks=aligned)
        self.assertIn("đi.", out)
        self.assertTrue(out.endswith("-5.") or out.endswith("?") is False)
        self.assertFalse(vi_punct("không đi", breaks=[2]).endswith("?"))

    def test_registry_buckets_sv_frontend_dedup_files_and_rnnt_mock(self):
        with tempfile.TemporaryDirectory() as td:
            mdir = Path(td)
            cfg_path = mdir / "models.json"
            cfg_path.write_text(json.dumps({
                "_sv_frontend": "host",
                "asr_vi": {"onnx": "z1500.onnx", "ctx": "asr_vi.bin"},
                "asr_vi@300": {"onnx": "z300.onnx", "ctx": "asr_vi.bin", "graph_index": 0},
                "asr_vi@800": {"onnx": "z800.onnx", "ctx": "asr_vi.bin", "graph_index": 1},
                "asr_vi_decoder": {"onnx": "dec.onnx"},
                "asr_vi_joiner": {"onnx": "joi.onnx"},
                "asr_xx_frontend": {"onnx": "fe.onnx"},
                "asr_xx_encoder": {"onnx": "sv500.onnx", "ctx": "sv.bin"},
                "asr_xx_encoder@100": {"onnx": "sv100.onnx", "ctx": "sv.bin"},
                "mt": {"onnx": "mt.onnx"},
                "tts_vi": {"onnx": "tts_vi.onnx"},
                "tts_enko": {"onnx": "tts_enko.onnx"},
            }))

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("S2S_SV_FRONTEND", None)
                os.environ.pop("S2S_VI_DECODE", None)
                reg_cpu = Registry(str(cfg_path), "cpu", vad_profile="hybrid")
                self.assertEqual(reg_cpu.sv_frontend(), "host")
                self.assertEqual([w for w, _ in reg_cpu.windows("asr_vi", 1500)], [300, 800, 1500])
                self.assertEqual(reg_cpu.asr_keys("vi"), ["asr_vi@300", "asr_vi@800", "asr_vi"])
                self.assertEqual(reg_cpu.asr_keys("en"), ["asr_xx_encoder@100", "asr_xx_encoder"])

                with patch.dict(os.environ, {"S2S_SV_FRONTEND": "graph"}):
                    self.assertEqual(reg_cpu.asr_keys("en"), ["asr_xx_frontend", "asr_xx_encoder@100", "asr_xx_encoder"])

                reg_npu = Registry(str(cfg_path), "npu")
                self.assertEqual(
                    reg_npu.files(reg_npu.asr_keys("vi")),
                    [str(mdir / "asr_vi.bin"), str(mdir / "dec.onnx"), str(mdir / "joi.onnx")],
                )
                with patch.dict(os.environ, {"S2S_VI_DECODE": "ctc"}):
                    self.assertIsNone(reg_npu.transducer())

            # Test Transducer greedy & word_ends with stub ONNX sessions via mock
            class FakeDec:
                def get_inputs(self):
                    return [types.SimpleNamespace(name="y")]
                def get_modelmeta(self):
                    return types.SimpleNamespace(custom_metadata_map={"context_size": "2"})
                def run(self, _o, _in):
                    return [np.zeros((1, 1, 512), np.float32)]

            class FakeJoi:
                def get_inputs(self):
                    return [types.SimpleNamespace(name="enc"), types.SimpleNamespace(name="dec")]
                def run(self, _o, feeds):
                    n = feeds["enc"].shape[0]
                    out = np.zeros((n, 2000), np.float32)
                    out[:, 0] = 5.0
                    mask = feeds["enc"][:, 0] > 0.5
                    if mask.any():
                        out[mask, 12] = 10.0
                    return [out]

            class FakeOrtMod:
                SessionOptions = lambda: types.SimpleNamespace(log_severity_level=0, intra_op_num_threads=1)
                @staticmethod
                def InferenceSession(path, *_a, **_kw):
                    return FakeDec() if "dec" in path else FakeJoi()

            with patch.dict(sys.modules, {"onnxruntime": FakeOrtMod, "onnx": None}):
                tr = Transducer("dec.onnx", "joi.onnx")
                enc = np.zeros((10, 512), np.float32)
                enc[2, 0] = 1.0
                toks, frames = tr.greedy(enc, return_frames=True)
                self.assertEqual(toks, [12])
                self.assertEqual(frames, [2])
                ends = _word_ends(types.SimpleNamespace(decode=lambda p: "xin" if p else ""), toks, frames)
                self.assertEqual(ends, [2])

    def test_qnn_apply_perf_after_load_and_shared_context_graph_index(self):
        events = []

        class FakeCtx:
            def __init__(self, **kw):
                events.append(("QNNContext", kw["model_path"]))
            def getInputName(self, *a): return ["x"]
            def getInputShapes(self, *a): return [[1, 4]]
            def getInputDataType(self, *a): return ["float32"]
            def getOutputShapes(self, *a): return [[1, 2]]
            def getOutputDataType(self, *a): return ["float32"]
            def Inference(self, arrs, graphIndex=None):
                events.append(("Inference", graphIndex, len(arrs)))
                return [np.ones((1, 2), np.float32)]
            def release(self):
                events.append(("release",))

        class FakePerf:
            BURST = "BURST_ENUM"
            @staticmethod
            def SetPerfProfileGlobal(val):
                events.append(("SetPerfProfileGlobal", val))

        class FakeConfig:
            @staticmethod
            def Config(*a):
                events.append(("QNNConfig",))

        fake_qai = types.SimpleNamespace(
            __file__="/tmp/fake_qai/__init__.py",
            QNNConfig=FakeConfig,
            Runtime=types.SimpleNamespace(HTP=1),
            LogLevel=types.SimpleNamespace(WARN=2),
            ProfilingLevel=types.SimpleNamespace(OFF=0),
            PerfProfile=FakePerf,
            QNNContext=FakeCtx,
        )

        old_state = (backends._qnn_ready, backends._perf, backends._perf_done, dict(backends._contexts), dict(backends._context_refs))
        try:
            backends._qnn_ready = False
            backends._perf = None
            backends._perf_done = False
            backends._contexts.clear()
            backends._context_refs.clear()

            with patch.dict(sys.modules, {"qai_appbuilder": fake_qai}), patch.dict(os.environ, {"S2S_PERF": "BURST", "S2S_SYSTEM_QNN": "1"}):
                with tempfile.TemporaryDirectory() as td:
                    bin_path = str(Path(td) / "shared_asr.bin")
                    g0 = backends.QnnGraph(bin_path, "asr_300", graph_index=0)
                    g1 = backends.QnnGraph(bin_path, "asr_800", graph_index=1)
                    self.assertIs(g0.ctx, g1.ctx)

                    # Verify SetPerfProfileGlobal ran AFTER the first QNNContext, not before it
                    ctx_idx = events.index(("QNNContext", bin_path))
                    perf_idx = events.index(("SetPerfProfileGlobal", "BURST_ENUM"))
                    self.assertGreater(perf_idx, ctx_idx)
                    self.assertEqual(sum(1 for e in events if e[0] == "QNNContext"), 1)

                    g0.run({"x": np.zeros((1, 4), np.float32)})
                    g1.run({"x": np.zeros((1, 4), np.float32)})
                    self.assertIn(("Inference", 0, 1), events)
                    self.assertIn(("Inference", 1, 1), events)

                    g0.release()
                    self.assertNotIn(("release",), events)
                    g1.release()
                    self.assertIn(("release",), events)
        finally:
            backends._qnn_ready, backends._perf, backends._perf_done, c, r = old_state
            backends._contexts.clear(); backends._contexts.update(c)
            backends._context_refs.clear(); backends._context_refs.update(r)


class RealAsrSpeedAccuracyE2ETests(unittest.TestCase):
    def test_host_sv_frontend_and_hybrid_vad_real_audio(self):
        ev3775 = ROOT.parent / "en-vi/BTC/EV-3775_in.wav"
        en_special = ROOT / "results/e2e/audio/en_to_vi_special_input.wav"
        vi_special = ROOT / "results/e2e/audio/vi_to_en_special_input.wav"
        if not (ev3775.exists() and en_special.exists() and vi_special.exists()):
            self.skipTest("Real regression fixtures missing")

        # Host SenseVoice frontend fixes 'Gett me' -> 'Get me' and preserves 5:30 AM
        with patch.dict(os.environ, {"S2S_SV_FRONTEND": "host"}):
            reg_host = Registry(str(ROOT / "models/models.cpu.json"), "cpu", ort_policy="auto")
            self.assertEqual(reg_host.sv_frontend(), "host")
            asr_en = reg_host.asr("en")
            hyp = asr_en(load_audio(str(ev3775)))
            self.assertIn("Get me a taxi", hyp)
            self.assertIn("5:30 AM", hyp)

        # Full Translator E2E under vad_profile='hybrid' + tts_format='hybrid' + source_mode='0'
        reg_hyb = Registry(
            str(ROOT / "models/models.cpu.json"),
            "cpu",
            ort_policy="auto",
            vad_profile="hybrid",
            source_mode="0",
            tts_format="hybrid",
        )
        tr_en = Translator(reg_hyb, "en", "vi", vad_profile="hybrid")
        res_en = tr_en.translate(load_audio(str(en_special)))
        joined_asr = " ".join(r["asr"] for r in res_en)
        joined_tts = " ".join(r["tts_text"] for r in res_en)
        for token in ("3:30", "15.50", "37.5", "67297"):
            self.assertIn(token, joined_asr)
        self.assertTrue(any(len(r["wav"]) > 0 and np.isfinite(r["wav"]).all() for r in res_en))
        self.assertTrue(len(joined_tts) > 0)

        tr_vi = Translator(reg_hyb, "vi", "en", vad_profile="hybrid")
        res_vi = tr_vi.translate(load_audio(str(vi_special)))
        self.assertTrue(any(len(r["wav"]) > 0 and np.isfinite(r["wav"]).all() for r in res_vi))
        self.assertTrue(any(r["mt"] for r in res_vi))


if __name__ == "__main__":
    unittest.main()
