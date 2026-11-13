"""Tests for Plan 03 T3: span-preserving source segmentation & MT/conversation S=64 integration."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S2S_DIR = ROOT / "s2s_edge"
ASSETS_DIR = S2S_DIR / "assets"
if str(S2S_DIR) not in sys.path:
    sys.path.insert(0, str(S2S_DIR))

from s2s_edge.numeric_guard import extract_known_numeric_spans
from s2s_edge.source_segmentation import SpanBudgetError, load_nllb_encode, split_preserving_spans
from s2s_edge.mt import MtFallback, Nllb
from s2s_edge.conversation import Engine, split_sentences


class SourceSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encode = staticmethod(load_nllb_encode(str(ASSETS_DIR)))

    def test_preserves_failing_spans_and_normal_sentence_boundaries(self):
        text = (
            "Meet at 3:30 PM on 21/12/2027. "
            "The ticket costs $18.25 at 36.8°C! "
            "Call 0908 246 135 for flight VN247. Have a safe trip."
        )
        spans = extract_known_numeric_spans(text, "en")
        chunks = split_preserving_spans(text, spans, token_limit=62, encode=self.encode, lang="en")
        self.assertGreaterEqual(len(chunks), 3)
        joined = " ".join(chunks)
        for token in ("3:30 PM", "21/12/2027", "$18.25", "36.8°C", "0908 246 135", "VN247"):
            self.assertTrue(any(token in c for c in chunks), f"{token!r} was split across {chunks}")
            self.assertIn(token, joined)
        for c in chunks:
            self.assertLessEqual(len(self.encode(c)), 62)

    def test_tight_budget_never_splits_inside_protected_spans(self):
        # Put protected spans right across tight chunk boundaries (budget=12 tokens)
        text = (
            "Please check the invoice total $18.25 and temperature 36.8°C at 3:30 PM "
            "on 21/12/2027 for phone 0908 246 135 and flight VN247 before boarding."
        )
        spans = extract_known_numeric_spans(text, "en")
        chunks = split_preserving_spans(text, spans, token_limit=12, encode=self.encode, lang="en")
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(len(self.encode(c)), 12)
        for sp in spans:
            self.assertTrue(
                any(sp.raw in c for c in chunks),
                f"Span {sp.raw!r} was broken across chunks: {chunks}",
            )

    def test_oversized_single_span_raises_explicit_error(self):
        text = "Call 090824613512345678901234567890 now."
        spans = extract_known_numeric_spans(text, "en")
        with self.assertRaises(SpanBudgetError):
            split_preserving_spans(text, spans, token_limit=4, encode=self.encode, lang="en")

    def test_nllb_and_mt_fallback_do_not_split_decimals_or_drop_graph_on_budget_error(self):
        seen_inputs = []

        class FakeGraph:
            def __init__(self, nllb_ref):
                self.nllb_ref = nllb_ref
            def __call__(self, src_ids, tgt_lang):
                seq = [int(x) for x in src_ids[0] if int(x) != 1]
                # Decode tokens (skipping lang prefix and trailing 2)
                inner = seq[1:-1]
                seen_inputs.append(self.nllb_ref.decode(inner))
                out = [int(tgt_lang[0])] + inner + [2]
                return [np.array(out, dtype=np.int32)]

        nllb = Nllb(None, str(ASSETS_DIR), S=64)
        nllb.g = FakeGraph(nllb)

        out = nllb("The ticket costs $15.50. The temperature is 37.5°C.", "en", "vi")
        self.assertIn("$15.50", out)
        self.assertIn("37.5", out)
        self.assertFalse(any(s.strip().endswith("$15.") for s in seen_inputs))
        self.assertFalse(any(s.strip().endswith("37.") for s in seen_inputs))

        # MtFallback must re-raise ValueError/SpanBudgetError instead of releasing/dropping active decoder
        class FakeReg:
            mt_active = None
            def release(self, k):
                raise AssertionError("MtFallback must not drop decoder on SpanBudgetError")

        fb = MtFallback(FakeReg(), ["mt_decB"], lambda k: nllb, probe=False)
        nllb_tight = Nllb(FakeGraph(nllb), str(ASSETS_DIR), S=6)
        fb.impl = nllb_tight
        with self.assertRaises(SpanBudgetError):
            fb("Call 09082461351234567890.", "en", "vi")
        self.assertEqual(fb.failures, [])

    def test_batch_and_stream_use_identical_span_safe_segmentation(self):
        recorded_batch = []
        recorded_stream = []

        class MockMt:
            S = 18
            def __init__(self, sink):
                self.encode = load_nllb_encode(str(ASSETS_DIR))
                self.sink = sink
            def translate_atomic(self, text, src, tgt):
                self.sink.append(text)
                return text

        class FakeAsr:
            max_s = 12.0
            def __call__(self, _w):
                return "Price is $18.25 and temp is 36.8°C. Flight VN247 departs at 3:30 PM."

        class FakeTts:
            sr = 16000
            tts_input_text = ""
            format_trace = {}
            def synthesize(self, text):
                self.tts_input_text = text
                return np.ones(160, dtype=np.float32), self.sr

        class FakeReg:
            source_mode = "0"
            glossary = {}
            def __init__(self, sink):
                self._mt_inst = MockMt(sink)
            def mt(self):
                return self._mt_inst
            def asr(self, lang):
                return FakeAsr()
            def tts(self, lang):
                return FakeTts()
            def components_for(self, s, t):
                return []

        wav = np.ones(1600, dtype=np.float32) * 0.05
        eng_b = Engine(FakeReg(recorded_batch))
        eng_s = Engine(FakeReg(recorded_stream))

        res_b = eng_b.translate_segment(wav, "en", "vi", prepped=True)
        res_s = list(eng_s.translate_segment_stream(wav, "en", "vi", prepped=True))

        self.assertEqual(recorded_batch, recorded_stream)
        self.assertGreaterEqual(len(recorded_batch), 2)
        self.assertIn("$18.25", res_b["mt"])
        self.assertIn("36.8", res_b["mt"])
        self.assertEqual(len(res_s), len(recorded_stream))

    def test_split_sentences_backward_compatible_and_span_safe(self):
        parts = split_sentences("Price is $15.50. Temp is 37.5°C.", "en")
        self.assertEqual(parts, ["Price is $15.50.", "Temp is 37.5°C."])


if __name__ == "__main__":
    unittest.main()
