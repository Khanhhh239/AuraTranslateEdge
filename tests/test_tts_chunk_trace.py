import unittest
from unittest.mock import patch

import numpy as np

from s2s_edge.tts import TtsEnKo, TtsVi, TtsZh
from s2s_edge.tts_format import adaptive, chunks


class _FakeViGraph:
    def __init__(self, samples_per_call: int = 1000, oov_per_unknown: int = 1) -> None:
        self.samples_per_call = samples_per_call
        self.oov_per_unknown = oov_per_unknown
        self.seen_texts: list[str] = []

    def __call__(self, text_bytes: np.ndarray):
        raw = bytes(int(x) for x in text_bytes[0] if x > 0).decode('utf-8', errors='replace')
        self.seen_texts.append(raw)
        n_oov = self.oov_per_unknown if 'unknownword' in raw else 0
        audio = np.zeros((1, self.samples_per_call), dtype=np.float32)
        y_len = np.array([10], dtype=np.int32)
        oov = np.array([float(n_oov)], dtype=np.float32)
        return [audio, y_len, oov]


class _FakeEnKoGraph:
    def __init__(self, max_len: int = 96) -> None:
        self.max_len = max_len
        self.seen_cps: list[list[int]] = []

    def __call__(self, cp: np.ndarray, style_dp=None, style_ttl=None):
        assert cp.shape == (1, self.max_len), cp.shape
        nonzero = [int(x) for x in cp[0] if x > 0]
        assert len(nonzero) <= self.max_len
        self.seen_cps.append(nonzero)
        wav = np.zeros((1, 460800), dtype=np.float32)
        lat = np.array([10.0], dtype=np.float32)
        return [wav, lat]


class _FakeZhGraph:
    def __init__(self) -> None:
        self.seen_bytes: list[bytes] = []

    def __call__(self, text_bytes: np.ndarray):
        raw = bytes(int(x) for x in text_bytes[0] if x > 0)
        assert len(raw) <= 112, f'exceeded 112 UTF-8 bytes: {len(raw)}'
        self.seen_bytes.append(raw)
        audio = np.zeros((1, 204800), dtype=np.float32)
        y_len = np.array([10], dtype=np.int32)
        return [audio, y_len]


class TestAdaptiveChunkTrace(unittest.TestCase):
    def test_planned_vs_attempted_vs_synthesized_under_cap_retry(self) -> None:
        text = 'This is sentence one with many words. And sentence two has many words too.'
        rec: dict = {}
        calls: list[str] = []

        def fake_synth(piece: str):
            calls.append(piece)
            if piece == calls[0] and len(calls) == 1:
                return np.zeros(100, dtype=np.float32), True
            return np.zeros(50, dtype=np.float32), False

        audios = adaptive(text, 'en', max_units=10, unit='word', synth=fake_synth, record=rec)
        self.assertGreaterEqual(len(audios), 2)
        self.assertIn('planned_chunks', rec)
        self.assertIn('attempted_chunks', rec)
        self.assertIn('synthesized_chunks', rec)
        first_attempted = rec['attempted_chunks'][0]
        self.assertTrue(first_attempted['capped'])
        self.assertEqual(first_attempted['depth'], 0)
        self.assertEqual(rec['audio_cap_retries'], [first_attempted['text']])
        self.assertNotIn(first_attempted['text'], [s['text'] for s in rec['synthesized_chunks']])
        for s in rec['synthesized_chunks']:
            self.assertFalse(s['capped'])
            self.assertGreater(s['utf8_bytes'], 0)
            self.assertGreater(s['word_count'], 0)

    def test_duplicate_spoken_occurrence_span_protection_and_unmapped_warning(self) -> None:
        text = 'Code AB-012 and code AB-012.'
        rec: dict = {}
        pieces = chunks(
            text,
            'vi',
            max_units=26,
            unit='word',
            record=rec,
            span_ranges=[(5, 11), (21, 27)],
        )
        self.assertEqual(pieces, [text])

        # Duplicate spoken text after transformation where spans must never share index 0
        rec2: dict = {}
        chunks(
            'một hai ba AB-012 rồi lại một hai ba AB-012',
            'vi',
            max_units=6,
            unit='word',
            record=rec2,
            span_ranges=[(11, 17), (36, 42)],
        )
        # Every resulting piece containing 'AB-012' keeps 'AB-012' intact without cutting inside it
        self.assertEqual(''.join(rec2['planned_chunks']).count('AB-012'), 2)

        # Duplicate spoken substring with missing/non-matching norm offsets triggers unmapped warning
        # instead of guessing via str.find.
        rec3 = {
            'input': '12 and 12',
            'output': 'different output',
            'spans': [
                {'raw': '12', 'spoken': 'mười hai'},
                {'raw': '12', 'spoken': 'mười hai'},
            ],
            'protected_ranges': [(99, 107), (120, 128)],
            'warnings': [],
        }
        chunks('mười hai và mười hai', 'vi', max_units=10, unit='word', record=rec3)
        self.assertTrue(
            any('unmapped' in w['reason'] for w in rec3['warnings']),
            rec3['warnings'],
        )
        self.assertTrue(any(m['status'] == 'unmapped' for m in rec3['span_mapping']))


class TestTtsModelContracts(unittest.TestCase):
    def test_ttsvi_duplicate_occurrence_latin_to_vi_and_oov_delta(self) -> None:
        graph = _FakeViGraph(oov_per_unknown=1)
        tts = TtsVi(graph, assets_dir=None, glossary=None, tts_format='hybrid')
        # Simulate VietnameseNames loaded (`self.vn`) so latin_to_vi runs on segments
        tts.vn = object()
        with patch('s2s_edge.tts.latin_to_vi', side_effect=lambda s, *_a: s.replace('Video', 'vi đê ô').replace('video', 'vi đê ô')):
            audio, sr = tts.synthesize('Video AB-012 và video AB-012 unknownword')
            self.assertEqual(sr, 22050)
            self.assertGreater(len(audio), 0)
            trace = tts.format_trace
            self.assertEqual(trace['oov_delta'], 1)
            self.assertEqual(tts.last_oov_delta, 1)
            self.assertIn('synthesized_chunks', trace)
            self.assertIn('attempted_chunks', trace)
            self.assertIn('planned_chunks', trace)
            joined = ' '.join(graph.seen_texts)
            self.assertEqual(joined.count('vi đê ô'), 2)
            # Duplicate span ranges map distinctly at their two separate positions
            self.assertEqual(len(trace['tts_input_protected_ranges']), 2)
            r0, r1 = trace['tts_input_protected_ranges']
            self.assertLess(r0[1], r1[0])
            for sc in trace['synthesized_chunks']:
                self.assertLessEqual(sc['word_count'], 26)
                self.assertLessEqual(sc['utf8_bytes'], 512)

            # Second call with zero OOV must report oov_delta == 0 (not cumulative)
            tts.synthesize('chào bạn')
            self.assertEqual(tts.format_trace['oov_delta'], 0)
            self.assertEqual(tts.oov, 1)

    def test_ttsvi_512_byte_limit_splits_long_words(self) -> None:
        graph = _FakeViGraph()
        tts = TtsVi(graph, assets_dir=None, glossary=None, tts_format='hybrid')
        long_vi = ' '.join(['người' * 5] * 18)
        tts.synthesize(long_vi)
        self.assertGreaterEqual(len(graph.seen_texts), 2)
        for sent in graph.seen_texts:
            self.assertLessEqual(len(sent.encode('utf-8')), 512)
            self.assertLessEqual(len(sent.split()), 26)
        for sc in tts.format_trace['synthesized_chunks']:
            self.assertLessEqual(sc['utf8_bytes'], 512)
            self.assertLessEqual(sc['word_count'], 26)

    def test_ttszh_byte_and_char_limits_and_indivisible_raise(self) -> None:
        graph = _FakeZhGraph()
        tts = TtsZh(graph, tts_format='hybrid')
        tts.synthesize('这是一个非常长的中文测试句子用于验证字节和字符上限不会超出范围。')
        self.assertGreaterEqual(len(graph.seen_bytes), 2)
        for raw in graph.seen_bytes:
            self.assertLessEqual(len(raw), 112)
            self.assertLessEqual(len(raw.decode('utf-8')), 20)
        for sc in tts.format_trace['synthesized_chunks']:
            self.assertLessEqual(sc['utf8_bytes'], 112)
            self.assertLessEqual(len(sc['text']), 20)

        with self.assertRaises(ValueError):
            tts.synthesize('A' * 120)

    def test_ttsenko_preprocessed_n96_contract_and_retry_trace(self) -> None:
        graph = _FakeEnKoGraph(max_len=96)

        class _FakeProc:
            def __init__(self, *_a):
                self.N = 96
                self.indexer = list(range(256))

            def _preprocess_text(self, s: str, _lang: str) -> str:
                return f'<en>{s}</en>'

        fake_styles = {
            'M1_dp': np.zeros((1, 256), dtype=np.float32),
            'M1_ttl': np.zeros((1, 256), dtype=np.float32),
        }
        with patch('s2s_edge.vendor.supertonic_text.UnicodeProcessor', _FakeProc), patch(
            's2s_edge.tts.np.load', return_value=fake_styles
        ):
            tts = TtsEnKo(graph, 'en', assets_dir='.', tts_format='hybrid')
            long_en = 'First clause with numbers 123, ' + 'detail ' * 25 + 'end.'
            tts.synthesize(long_en)
            trace = tts.format_trace
            self.assertGreaterEqual(len(trace['synthesized_chunks']), 2)
            self.assertEqual(len(trace['preprocessed_chunk_lengths']), len(trace['synthesized_chunks']))
            for length in trace['preprocessed_chunk_lengths']:
                self.assertLessEqual(length, 96)
            for sc in trace['synthesized_chunks']:
                self.assertLessEqual(sc['preprocessed_length'], 96)


if __name__ == '__main__':
    unittest.main()
