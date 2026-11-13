"""Unit and integration tests for Plan 03 T1: semantic numeric guard (source_mode=0 & source_mode=1)."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S2S_DIR = ROOT / "s2s_edge"
if str(S2S_DIR) not in sys.path:
    sys.path.insert(0, str(S2S_DIR))

from s2s_edge.numeric_guard import (
    GuardReport,
    NumericSpan,
    extract_known_numeric_spans,
    guard_source_numeric_changes,
)
from s2s_edge.conversation import Engine
from s2s_edge import neural_textnorm


class NumericGuardTests(unittest.TestCase):
    def test_extract_pure_python_fast_and_works_without_crf(self):
        text = "Meet at 3:30 PM on 21/12/2027, price $15.50, -12.50%, 37.5°C, flight VN 123, phone 0912345678."
        with patch.dict(sys.modules, {"pycrfsuite": None}):
            t0 = time.perf_counter()
            for _ in range(20):
                spans = extract_known_numeric_spans(text, "en")
            elapsed_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
        self.assertLess(elapsed_ms, 5.0)
        self.assertTrue(all(isinstance(sp, NumericSpan) for sp in spans))
        raws = [sp.raw for sp in spans]
        self.assertTrue(any("3:30 PM" in r for r in raws))
        self.assertTrue(any("21/12/2027" in r for r in raws))
        self.assertTrue(any("15.50" in r for r in raws))
        self.assertTrue(any("0912345678" in r for r in raws))

    def test_report_statuses_verified_not_applicable_unknown_invalid(self):
        r_na = guard_source_numeric_changes("hello world", "Hello world.", "en")
        self.assertIsInstance(r_na, GuardReport)
        self.assertTrue(r_na.valid)
        self.assertEqual(r_na.status, "not_applicable")
        self.assertEqual(r_na.issues, [])
        self.assertEqual(r_na.unknowns, [])

        r_ok = guard_source_numeric_changes(
            "the meeting is at 3:30 pm and costs $15.50",
            "The meeting is at 3:30 PM, and costs $15.50.",
            "en",
        )
        self.assertTrue(r_ok.valid)
        self.assertEqual(r_ok.status, "verified")
        self.assertTrue(r_ok.token_identity)

        # Unknown opaque/ambiguous preserved verbatim -> valid=False, status='unknown'
        for txt in ("ratio 3-5", "coeff 1e-3", "date 03/04/2027", "date 03/04"):
            with self.subTest(txt=txt):
                r_unk = guard_source_numeric_changes(txt, txt + ".", "en")
                self.assertFalse(r_unk.valid)
                self.assertEqual(r_unk.status, "unknown")
                self.assertEqual(r_unk.issues, [])
                self.assertTrue(r_unk.unknowns)

        # Vietnamese spoken numeral sequence -> unknown, single negation -> not_applicable
        r_neg = guard_source_numeric_changes("tôi không biết khi nào", "Tôi không biết khi nào.", "vi")
        self.assertTrue(r_neg.valid)
        self.assertEqual(r_neg.status, "not_applicable")

        r_spoken = guard_source_numeric_changes("nhiệt độ ba bảy năm độ c", "Nhiệt độ ba bảy năm độ C.", "vi")
        self.assertFalse(r_spoken.valid)
        self.assertEqual(r_spoken.status, "unknown")
        self.assertEqual(r_spoken.issues, [])

    def test_frozen_mutation_suite_caught_100_percent(self):
        mutations = [
            # Lost negative sign
            ("en", "discount -12.50%", "discount 12.50%"),
            ("vi", "nhiệt độ -5 độ", "nhiệt độ 5 độ"),
            # Decimal -> time
            ("en", "price is 15.50", "price is 15:50"),
            # Date -> integer or dropped year
            ("en", "date 21/12/2027", "date 21122027"),
            ("en", "date 21/12/2027", "date 21/12"),
            # Lost leading zero
            ("en", "phone 0912345678", "phone 912345678"),
            ("en", "ZIP 04782", "ZIP 4782"),
            # Swapped / duplicated / dropped digits
            ("en", "phone 0912345678", "phone 0912345768"),
            ("en", "phone 0912345678", "phone 09123456788"),
            ("en", "phone 0912345678", "phone 091234567"),
            # Unit / currency / precision changes
            ("en", "temp 37.5°C", "temp 37.5°F"),
            ("en", "temp 37.5 degrees Celsius", "temp 37.5 degrees Fahrenheit"),
            ("en", "price $15.50", "price €15.50"),
            ("en", "price $15.50", "price $15.5"),
            ("en", "distance 10km", "distance 10m"),
            # Meridiem / 12h vs 24h mismatch
            ("en", "meet at 3:30 PM", "meet at 3:30 AM"),
            ("en", "meet at 3:30 PM", "meet at 3:30"),
            ("en", "meet at 3:30 PM", "meet at 15:30"),
            # Opaque mutations
            ("en", "coeff 1e-3", "coeff 1e3"),
            ("en", "ratio 3-5", "ratio 35"),
            ("en", "date 03/04", "date 03:04"),
            ("en", "date 03/04/2027", "date 2027-03-04"),
            # Punctuation inside identifier span
            ("vi", "chuyến bay VN 123", "chuyến bay VN, 123"),
            # Non-numeric suffix dropped
            ("vi", "giá vé $15.50 kết thúc tốt đẹp", "giá vé $15.50"),
        ]
        for lang, before, after in mutations:
            with self.subTest(lang=lang, before=before, after=after):
                rep = guard_source_numeric_changes(before, after, lang)
                self.assertFalse(rep.valid)
                self.assertEqual(rep.status, "invalid")
                self.assertTrue(rep.issues)

    def test_330_pm_typed_context_rule(self):
        # Explicit meridiem context allows 330 pm -> 3:30 PM
        ok = guard_source_numeric_changes("meet at 330 pm", "Meet at 3:30 PM.", "en")
        self.assertTrue(ok.valid)
        self.assertEqual(ok.status, "verified")

        # Without meridiem context, 330 -> 3:30 is NOT allowed
        bad_no_meridiem = guard_source_numeric_changes("amount 330", "amount 3:30", "en")
        self.assertFalse(bad_no_meridiem.valid)
        self.assertEqual(bad_no_meridiem.status, "invalid")

        # Invalid 12h hour with PM (1530 pm) is NOT allowed
        bad_1530 = guard_source_numeric_changes("meet at 1530 pm", "Meet at 3:30 PM.", "en")
        self.assertFalse(bad_1530.valid)
        self.assertEqual(bad_1530.status, "invalid")

        # Meridiem flip 330 pm -> 3:30 AM is NOT allowed
        bad_flip = guard_source_numeric_changes("meet at 330 pm", "Meet at 3:30 AM.", "en")
        self.assertFalse(bad_flip.valid)
        self.assertEqual(bad_flip.status, "invalid")

    def test_t1a_source_mode_0_fallback_on_corrupted_edge_prep(self):
        class FakeMt:
            S = 64
            def encode(self, s):
                return [ord(c) for c in s]
            def translate_atomic(self, s, src, tgt):
                return s

        class FakeReg:
            source_mode = "0"
            glossary = {}
            def mt(self):
                return FakeMt()
            def components_for(self, s, t):
                return []

        eng = Engine(FakeReg())
        self.assertFalse(eng.neural_enabled())
        raw = "giá vé $15.50 lúc 3:30 PM gọi 0912345678"
        with patch("s2s_edge.conversation.restore_vi", return_value="giá vé $15.50 lúc 3:30 AM gọi 912345678"):
            prepared, parts = eng.prepare_text(raw, "vi")
        self.assertEqual(prepared, raw)
        self.assertEqual(eng.last_source_guard["fallback_policy"], "fallback_to_raw_source")
        self.assertEqual(parts, [raw])

    def test_t1b_neural_textnorm_guards_canonical_and_qwen_outputs(self):
        # Construct NeuralNormalizer without loading Apple Metal weights
        norm = neural_textnorm.NeuralNormalizer.__new__(neural_textnorm.NeuralNormalizer)
        norm.records = []
        norm.cache = {}

        # 1) canonical_identifiers guard catches corruption before Qwen
        with patch.object(norm, "canonical_identifiers", return_value=("price $15.5", [{"source": "$15.50", "canonical": "$15.5"}])):
            with patch.object(norm, "_generate", return_value=('{"segments": ["price $15.5"]}', types.SimpleNamespace(finish_reason="stop"))):
                # Since canonical_identifiers corrupted $15.50 -> $15.5, guard falls back to original "$15.50",
                # then Qwen receives original "$15.50" and fails 3 retries -> falls back to span-safe split preserving "$15.50".
                parts = norm.prepare_source("price $15.50", "en", 62, lambda s: s.split())
        self.assertEqual(" ".join(parts), "price $15.50")
        self.assertTrue(any(r.get("fallback_policy") == "fallback_to_raw_source" for r in norm.records))

        # 2) Qwen mutation (e.g. 1e-3 -> 1e3) is rejected by numeric_guard and retried, then verified when fixed
        norm.records.clear()
        responses = [
            ('{"segments": ["Coeff 1e3."]}', types.SimpleNamespace(finish_reason="stop")),
            ('{"segments": ["Coeff 1e-3."]}', types.SimpleNamespace(finish_reason="stop")),
        ]
        with patch.object(norm, "canonical_identifiers", side_effect=lambda t, l: (t, [])):
            with patch.object(norm, "_generate", side_effect=responses):
                parts = norm.prepare_source("coeff 1e-3", "en", 62, lambda s: s.split())
        self.assertEqual(parts, ["Coeff 1e-3."])
        seg_rec = [r for r in norm.records if r["stage"] == "asr_segmentation"][-1]
        self.assertNotIn("content_preserved", seg_rec)
        self.assertTrue(seg_rec["token_identity"])
        self.assertEqual(seg_rec["numeric_validation"]["status"], "unknown")
        self.assertEqual(seg_rec["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
