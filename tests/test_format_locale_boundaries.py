import random
import unittest

from s2s_edge.hybrid_textnorm import (
    HybridNormalizer,
    extract_spans,
    numeric_fields,
)


class TestLocaleAndPunctuationBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.norm = HybridNormalizer(learned_identifiers=False)

    def test_cjk_comma_separates_numbers_cleanly(self) -> None:
        text = '1000000000000元，1000000001000，3-5天，系数1e-3。'
        out, rec = self.norm(text, 'zh')
        self.assertIn('一万亿元', out)
        self.assertIn('一万亿零一千', out)
        self.assertIn('，', out)
        self.assertEqual(
            [(s['raw'], s['kind']) for s in rec['spans'] if s['raw'] in ('3-5', '1e-3')],
            [('3-5', 'unresolved'), ('1e-3', 'unresolved')],
        )
        unsupported_warnings = [w for w in rec['warnings'] if 'unsupported numeric notation' in w['reason']]
        self.assertEqual([w['raw'] for w in unsupported_warnings], ['3-5', '1e-3'])

    def test_scientific_and_range_adjacent_to_cjk_stay_whole(self) -> None:
        cases = [
            ('zh', '3-5天', '3-5'),
            ('zh', '系数1e-3。', '1e-3'),
            ('ko', '기간 3-5일입니다.', '3-5'),
            ('ko', '수량 1.5e+10개', '1.5e+10'),
        ]
        for lang, text, token in cases:
            with self.subTest(lang=lang, text=text):
                out, rec = self.norm(text, lang)
                unresolved = [s for s in rec['spans'] if s['kind'] == 'unresolved']
                self.assertEqual([s['raw'] for s in unresolved], [token])
                self.assertTrue(
                    any(w['raw'] == token and 'unsupported numeric notation' in w['reason'] for w in rec['warnings'])
                )
                # Adjacent text must not be swallowed into the unresolved span
                if text.endswith(('天', '。', '개')):
                    self.assertTrue(out.endswith(text[-1]))

    def test_numeric_fields_locale_separators(self) -> None:
        self.assertEqual(numeric_fields('1,234.50', 'en'), {'sign': '', 'whole': '1234', 'fraction': '50'})
        self.assertEqual(numeric_fields('1.234,50', 'vi'), {'sign': '', 'whole': '1234', 'fraction': '50'})
        self.assertEqual(numeric_fields('1,234.50', 'zh'), {'sign': '', 'whole': '1234', 'fraction': '50'})
        self.assertEqual(numeric_fields('1,234.50', 'ko'), {'sign': '', 'whole': '1234', 'fraction': '50'})
        self.assertEqual(numeric_fields('-12,50', 'vi'), {'sign': '-', 'whole': '12', 'fraction': '50'})
        self.assertEqual(numeric_fields('３６．８', 'zh'), {'sign': '', 'whole': '36', 'fraction': '8'})
        # Invalid/malformed grouping raises explicit ValueError (caught as unresolved by normalizer)
        for bad in ('12.34.56', '1,23,45', '1234,567,890'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                numeric_fields(bad, 'en')

    def test_fullwidth_digits_and_negative_decimal_and_currency(self) -> None:
        zh_out, _ = self.norm('０８：００和３６．８℃。', 'zh')
        self.assertIn('八点', zh_out)
        self.assertIn('三十六点八摄氏度', zh_out)

        en_out, _ = self.norm('Budget €1,250.50 (-12.5%).', 'en')
        self.assertIn('one thousand two hundred fifty euros fifty cents', en_out)
        self.assertIn('minus twelve point five percent', en_out)

        vi_out, _ = self.norm('Giá €1.250,50, âm -12,50%.', 'vi')
        self.assertIn('một nghìn hai trăm năm mươi euro năm mươi xen', vi_out)
        self.assertIn('âm mười hai phẩy năm không phần trăm', vi_out)

    def test_urls_emails_versions_and_codes_not_partially_captured(self) -> None:
        cases = [
            ('en', 'Visit https://example.com/2027/03/04?id=123&v=2 now.', 'https://example.com/2027/03/04?id=123&v=2'),
            ('en', 'Email a1@example.com or v1.2.3 release.', 'a1@example.com'),
            ('en', 'Upgrade to v1.2.3 today.', 'v1.2.3'),
            ('en', 'Install version 1.2.3 carefully.', '1.2.3'),
            ('vi', 'Xem https://test.vn/03/04/2027 và mã AB-012.', 'https://test.vn/03/04/2027'),
        ]
        for lang, text, preserved in cases:
            with self.subTest(lang=lang, text=text):
                spans = extract_spans(text, lang, learned_identifiers=False)
                for s in spans:
                    self.assertNotIn(s['raw'], preserved if s['raw'] != preserved else '')
                out, _ = self.norm(text, lang)
                self.assertIn(preserved, out)

    def test_ambiguous_separator_policy_recorded_and_flagged(self) -> None:
        for text in ('Value 12.34.56.', 'Count 1,23,45.', 'Big 1234,567,890.'):
            with self.subTest(text=text):
                _, rec = self.norm(text, 'en')
                self.assertIn('separator_policy', rec)
                self.assertTrue(rec['warnings'])
                self.assertTrue(any(s['kind'] == 'unresolved' for s in rec['spans']))
                self.assertTrue(
                    any('ambiguous' in w['reason'] for w in rec['warnings']),
                    rec['warnings'],
                )


class TestSeededLocaleGrouping(unittest.TestCase):
    def test_seeded_en_vi_thousands_and_decimals(self) -> None:
        rng = random.Random(11132026)
        norm = HybridNormalizer(learned_identifiers=False)
        for _ in range(10):
            thousands = rng.randint(1, 9)
            rem = rng.randint(100, 999)
            frac = rng.randint(10, 99)
            en_tok = f'{thousands},{rem}.{frac}'
            vi_tok = f'{thousands}.{rem},{frac}'
            self.assertEqual(numeric_fields(en_tok, 'en'), {'sign': '', 'whole': f'{thousands}{rem}', 'fraction': str(frac)})
            self.assertEqual(numeric_fields(vi_tok, 'vi'), {'sign': '', 'whole': f'{thousands}{rem}', 'fraction': str(frac)})
            en_out, en_rec = norm(f'Total {en_tok}.', 'en')
            vi_out, vi_rec = norm(f'Tổng {vi_tok}.', 'vi')
            self.assertFalse([w for w in en_rec['warnings'] if w.get('raw') is not None])
            self.assertFalse([w for w in vi_rec['warnings'] if w.get('raw') is not None])
            self.assertNotIn(en_tok, en_out)
            self.assertNotIn(vi_tok, vi_out)


if __name__ == '__main__':
    unittest.main()
