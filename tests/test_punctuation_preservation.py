"""Tests for Plan 03 T2: CAPU overlap preservation, numeric/negation safety, and vi_punct fallback."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
S2S_DIR = ROOT / "s2s_edge"
if str(S2S_DIR) not in sys.path:
    sys.path.insert(0, str(S2S_DIR))

from s2s_edge.numeric_guard import GuardReport
from s2s_edge.punct_vi import LABELS, ViCapu, _apply, get as get_capu
from s2s_edge.textnorm import vi_punct
from s2s_edge.conversation import restore_vi


class PunctuationPreservationTests(unittest.TestCase):
    def test_apply_blocks_verb_digit_rewrite_merge_and_internal_punct(self):
        tokens = ["gọi", "số", "không", "chín", "một", "hai", "mã", "vn", "123", "nhiệt", "độ", "37.5", "độ", "c"]
        # Simulate CAPU edits: verb transform "không"->"0", merge "vn"+"123", append ":" inside "không chín", append "." inside "37.5 độ c"
        edits = [
            (2, 3, "$TRANSFORM_VERB_VB_VBN"),
            (3, 3, ","),
            (7, 8, "$TRANSFORM_CASE_UPPER"),
            (7, 7, "$MERGE_SPACE"),
            (11, 11, "."),
        ]
        out = _apply(tokens, edits, verbs={"không_VB_VBN": "0"})
        # Token count and non-punct words must stay 1-to-1 with source
        self.assertEqual([t.lower() for t in out if t not in {",", ".", ":", "?"}], tokens)
        self.assertNotIn("0", out)
        # No "," inserted between "chín" and "một" and no "." inserted between "37.5" and "độ"
        self.assertEqual(out[2], "không")
        self.assertNotEqual(out[12] if len(out) > 12 else "", ".")

    def test_mock_merge_fail_over_48_words_preserves_suffix_and_overlap(self):
        capu = get_capu(str(S2S_DIR / "assets"))
        if capu is None:
            self.skipTest("assets/vi_capu not present")

        prefix_words = [f"từ{i}" for i in range(38)]
        overlap_words = ["mã", "vn", "123", "số", "0912345678", "giá", "$15.50", "lúc", "3:30", "pm", "không", "đổi"]
        suffix_words = ["nhưng", "chúng", "ta", "vẫn", "tiếp", "tục", "kết", "thúc", "tốt", "đẹp", "ngày", "21/12/2027"]
        all_words = prefix_words + overlap_words + suffix_words
        self.assertGreater(len(all_words), 48)
        text = " ".join(all_words)

        with patch.object(capu, "_merge", side_effect=ValueError("no overlap")):
            out = capu(text)

        self.assertIsInstance(capu.last_validation, GuardReport)
        self.assertTrue(capu.last_validation.token_identity)
        self.assertNotEqual(capu.last_validation.status, "invalid")
        self.assertTrue(any(w.get("code") == "overlap_merge_fallback" for w in capu.last_validation.warnings))
        # Suffix must not be dropped and overlap must not be duplicated
        self.assertIn("21/12/2027", out)
        self.assertIn("kết thúc tốt đẹp", out)
        self.assertEqual(out.lower().count("0912345678"), 1)
        self.assertEqual(out.lower().count("vn 123"), 1)
        self.assertEqual(out.lower().count("$15.50"), 1)

    def test_last_validation_resets_per_call_on_capu_and_fallback(self):
        capu = get_capu(str(S2S_DIR / "assets"))
        if capu is not None:
            capu("nhiệt độ ba bảy năm độ c")
            first = capu.last_validation
            self.assertEqual(first.status, "unknown")
            capu("tôi đi làm hôm nay")
            second = capu.last_validation
            self.assertEqual(second.status, "not_applicable")
            self.assertIsNot(first, second)
            self.assertEqual(second.warnings, [])

        # Fallback path when assets/vi_capu is missing
        with patch("s2s_edge.punct_vi.get", return_value=None):
            out1 = restore_vi("tôi không biết khi nào")
            r1 = restore_vi.last_validation
            self.assertEqual(out1, "Tôi không biết khi nào.")
            self.assertEqual(r1.status, "not_applicable")
            self.assertTrue(r1.valid)

            out2 = restore_vi("nhiệt độ ba bảy năm độ c")
            r2 = restore_vi.last_validation
            self.assertEqual(r2.status, "unknown")
            self.assertFalse(r2.valid)
            self.assertIn("ba bảy năm", out2.lower())

    def test_negation_khong_and_zero_and_spoken_numeral_controls(self):
        # Rule-based vi_punct fallback controls
        self.assertEqual(vi_punct("tôi không biết khi nào"), "Tôi không biết khi nào.")
        self.assertEqual(vi_punct("không sao"), "Không sao.")
        self.assertEqual(vi_punct("gọi số không"), "Gọi số không.")
        self.assertEqual(vi_punct("năm hai nghìn không"), "Năm hai nghìn không.")
        self.assertEqual(vi_punct("anh có số không."), "Anh có số không?")

        # Pause breaks must never split inside protected spans
        broken = vi_punct("chuyến bay VN 123 gọi 0908 246 135 hôm nay", breaks=[3, 6])
        self.assertIn("VN 123", broken)
        self.assertIn("0908 246 135", broken)

        # Real vibert_capu_int8 ONNX on a long (>60 word) Vietnamese transcript
        capu = get_capu(str(S2S_DIR / "assets"))
        if capu is not None:
            long_text = (
                "xin chào anh chị hôm nay chúng ta họp lúc 14:45 ngày 21/12/2027 để kiểm tra đơn hàng vn 123 "
                "giá vé là $15.50 với mức giảm 12,50% và nhiệt độ phòng đo được là 37.5 độ c "
                "nếu có thắc mắc xin gọi số 0912345678 hoặc số không chín một hai ba bốn năm "
                "tôi không biết liệu chuyến bay có bị hoãn hay không nhưng mọi thứ vẫn ổn định"
            )
            out = capu(long_text)
            self.assertIsInstance(capu.last_validation, GuardReport)
            self.assertTrue(capu.last_validation.token_identity)
            self.assertNotEqual(capu.last_validation.status, "invalid")
            for must_keep in ("14:45", "21/12/2027", "$15.50", "12,50%", "37.5 độ c", "0912345678", "không biết"):
                self.assertIn(must_keep, out.lower())
            self.assertNotIn("0 chín", out.lower())


if __name__ == "__main__":
    unittest.main()
