"""Tests for Plan 03 T4: target entity audit and provenance ledger (without silent MT mutation)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
S2S_DIR = ROOT / "s2s_edge"
if str(S2S_DIR) not in sys.path:
    sys.path.insert(0, str(S2S_DIR))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from s2s_edge.numeric_guard import audit_target_entities, extract_known_numeric_spans
import s2s_web


class NumericAuditTests(unittest.TestCase):
    def test_audit_preserved_cross_lingual_equivalents(self):
        src = "Meet at 3:30 PM on November 15, 2026. Ticket $15.50, temp 37.5 degrees Celsius, call 0912345678."
        tgt = "Gặp lúc 3:30 chiều ngày 15 tháng 11 năm 2026. Giá vé 15,50 đô la, nhiệt độ 37,5 độ C, gọi 0912 345 678."
        rep = audit_target_entities(extract_known_numeric_spans(src, "en"), tgt, "vi", src_lang="en")
        self.assertEqual(rep["counts"]["mismatch"], 0)
        self.assertEqual(rep["counts"]["unverifiable"], 0)
        self.assertEqual(rep["counts"]["preserved"], 5)

    def test_audit_mismatches_wrong_value_unit_meridiem_and_duplicates(self):
        # Wrong value & unit & meridiem
        src = "Meet at 3:30 PM, price $15.50, temp 37.5°C."
        tgt_bad = "Gặp lúc 3:30 sáng, giá vé 15,00 euro, nhiệt độ 37,5 độ F."
        rep_bad = audit_target_entities(src, tgt_bad, "vi", src_lang="en")
        self.assertEqual(rep_bad["counts"]["preserved"], 0)
        self.assertEqual(rep_bad["counts"]["mismatch"], 3)

        # Duplicate occurrences: 2x $15.50 in source, only 1x in target -> 1 preserved + 1 mismatch
        rep_dup = audit_target_entities("Pay $15.50 and $15.50.", "Trả 15,50 đô la.", "vi", src_lang="en")
        self.assertEqual(rep_dup["counts"]["preserved"], 1)
        self.assertEqual(rep_dup["counts"]["mismatch"], 1)

        # 3:30 PM vs 15:30 or missing meridiem -> mismatch (no silent 12h/24h coercion)
        rep_24h = audit_target_entities("Meet at 3:30 PM.", "Gặp lúc 15:30.", "vi", src_lang="en")
        self.assertEqual(rep_24h["counts"]["mismatch"], 1)

    def test_audit_spoken_target_and_opaque_are_unverifiable(self):
        # Target writes numbers in words -> unverifiable, NOT pass or mismatch
        rep_spoken = audit_target_entities("Ticket is $15.50.", "Giá vé là mười lăm đô la năm mươi xen.", "vi", src_lang="en")
        self.assertEqual(rep_spoken["counts"]["unverifiable"], 1)
        self.assertEqual(rep_spoken["counts"]["preserved"], 0)
        self.assertEqual(rep_spoken["counts"]["mismatch"], 0)

        # Opaque/ambiguous source span -> unverifiable
        rep_opaque = audit_target_entities("Coeff 1e-3.", "Hệ số 1e-3.", "vi", src_lang="en")
        self.assertEqual(rep_opaque["counts"]["unverifiable"], 1)

    def test_s2s_web_records_audit_ledger_without_mutating_mt(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            wav_path = td_path / "in.wav"
            t = np.arange(8000, dtype=np.float32) / 16000.0
            sf.write(str(wav_path), 0.1 * np.sin(2 * np.pi * 220.0 * t), 16000)

            class FakeReg:
                def __init__(self, *a, **kw):
                    pass
                def graph_telemetry(self):
                    return {}
                def release(self, k):
                    pass

            class FakeAsr:
                max_s = 12.0
                def __call__(self, _w):
                    return "Ticket costs $15.50 at 3:30 PM."

            class FakeMt:
                S = 64
                def encode(self, s):
                    return s.split()
                def translate_atomic(self, s, src, tgt):
                    # Deliberately wrong MT output (15,00 đô la, 3:30 sáng) -> must NOT be mutated
                    return "Giá vé 15,00 đô la lúc 3:30 sáng."
                def __call__(self, s, src, tgt):
                    return self.translate_atomic(s, src, tgt)

            class FakeTts:
                sr = 16000
                tts_input_text = "giá vé mười lăm phẩy không không đô la lúc ba giờ ba mươi phút sáng."
                format_trace = {"stage": "tts_hybrid"}
                def synthesize(self, txt):
                    return np.ones(800, dtype=np.float32) * 0.05, self.sr

            class FakeTranslator:
                def __init__(self, reg, src, tgt, vad_profile=None):
                    self.asr = FakeAsr()
                    self.mt = FakeMt()
                    self.tts = FakeTts()
                    self.e = types.SimpleNamespace(_asr={src: self.asr}, _mt=self.mt, _tts={tgt: self.tts}, neural_enabled=lambda: False)
                def translate_segment(self, seg, **kw):
                    raw = self.e._asr["en"](seg)
                    mt_out = self.e._mt.translate_atomic(raw, "en", "vi")
                    self.e._tts["vi"].synthesize(mt_out)
                    audit = audit_target_entities(raw, mt_out, "vi", src_lang="en")
                    return {
                        "src": "en", "tgt": "vi", "asr": raw, "asr_before_normalization": raw,
                        "mt": mt_out, "tts_text": self.tts.tts_input_text, "tts_format": self.tts.format_trace,
                        "source_guard": {"status": "verified", "valid": True},
                        "target_audit": audit,
                        "wav": np.ones(800, dtype=np.float32) * 0.05, "sr": 16000, "timing": {},
                    }

            srv = s2s_web.Service(td_path / "jobs", source_mode="0", backend="npu", vad_profile="legacy", ort_policy="basic")
            job_id = "a" * 32
            jdir = srv.out / job_id
            jdir.mkdir(parents=True)
            (jdir / "input.wav").write_bytes(wav_path.read_bytes())
            srv.jobs[job_id] = {"id": job_id, "status": "queued", "src": "en", "tgt": "vi"}

            import s2s_edge.audio
            prev_tn_audit = os.environ.get("S2S_TN_AUDIT")
            try:
                with patch("s2s_edge.registry.Registry", FakeReg), patch("s2s_edge.pipeline.Translator", FakeTranslator), patch("s2s_edge.audio.segment", return_value=([(0, 8000)], [[]])):
                    srv.run(job_id)
            finally:
                if prev_tn_audit is None:
                    os.environ.pop("S2S_TN_AUDIT", None)
                else:
                    os.environ["S2S_TN_AUDIT"] = prev_tn_audit

            res = json.loads((jdir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(res["status"], "done")
            self.assertEqual(res["source_mode"], "0")
            self.assertEqual(res["execution_backend"], "npu")
            # MT must NOT be silently mutated
            self.assertEqual(res["translation"], "Giá vé 15,00 đô la lúc 3:30 sáng.")
            self.assertEqual(res["target_audit_summary"]["mismatch"], 2)
            self.assertEqual(len(res["audit_ledger"]), 1)
            entry = res["audit_ledger"][0]
            self.assertEqual(entry["asr_raw"], "Ticket costs $15.50 at 3:30 PM.")
            self.assertEqual(entry["mt"], "Giá vé 15,00 đô la lúc 3:30 sáng.")
            self.assertTrue((jdir / "audit_ledger.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
