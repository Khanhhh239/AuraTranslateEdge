import random
import unittest

from s2s_edge.hybrid_textnorm import (
    HybridNormalizer,
    extract_spans,
    identifier_candidates,
)

# 17 frozen edge cases extracted from reports/tts_format_s2s_eval_2026-11-13.md
# and its companion tts_format_4langs_report.json.
# P1 items (marked p1_deferred) are tracked in the frozen suite per PLAN.md
# (14h/14h45, độ C, Korean 3시 30분, VI ZIP->dip) without blocking P0 assertions.
FROZEN_EDGE_CASES = [
    {
        'id': 'vi_01',
        'lang': 'vi',
        'text': 'Chuyến bay của tôi là VN 123, và mã địa chỉ là 67297.',
        'p1_deferred': (),
    },
    {
        'id': 'vi_02',
        'lang': 'vi',
        'text': 'Nhiệt độ là 37.5 độ C. Cuộc họp lúc 14h hoặc 14h45, vé 50.000 đồng.',
        'p1_deferred': ('14h', '37.5 độ C'),
    },
    {
        'id': 'vi_03',
        'lang': 'vi',
        'text': 'Ngày 03/04/2027 lúc 00:05 và 12:00 PM. Tốc độ 100km/h, nặng 1kg.',
        'p1_deferred': (),
    },
    {
        'id': 'vi_04',
        'lang': 'vi',
        'text': 'Gọi +84-908-246-135, code AB-012. Tỷ lệ 3-5, hệ số 1e-3, âm -12,50%.',
        'p1_deferred': (),
    },
    {
        'id': 'vi_05',
        'lang': 'vi',
        'text': 'Từ 08:00 đến 15:30 PM tại 734 Main Street, giá €1.250,50.',
        'p1_deferred': (),
    },
    {
        'id': 'en_01',
        'lang': 'en',
        'text': 'Flight VN 123, ZIP 67297, phone (202) 555-0123, +1-800-555-0199.',
        'p1_deferred': (),
    },
    {
        'id': 'en_02',
        'lang': 'en',
        'text': 'Meeting at 12:00 PM or 00:05 on March 3, 2027, budget €1,250.50 (-12.5%).',
        'p1_deferred': (),
    },
    {
        'id': 'en_03',
        'lang': 'en',
        'text': 'Speed 100 km/h, weight 1kg, temp 37.5 degrees Celsius, range 3-5 days, 1e-3.',
        'p1_deferred': (),
    },
    {
        'id': 'en_04',
        'lang': 'en',
        'text': 'Ambiguous date 03/04/2027 and invalid 15:30 PM, ticket $1.01.',
        'p1_deferred': (),
    },
    {
        'id': 'ko_01',
        'lang': 'ko',
        'text': '회의는 오후 3시 30분, 2027년 12월 21일 08:00에 있습니다. 12:00 PM.',
        'p1_deferred': ('3시 30분',),
    },
    {
        'id': 'ko_02',
        'lang': 'ko',
        'text': '항공편 VN 247, 전화번호 +82-10-1234-5678, 우편번호 04782, 코드 AB-012.',
        'p1_deferred': (),
    },
    {
        'id': 'ko_03',
        'lang': 'ko',
        'text': '가격은 50000원, €18.25, -12.5% 할인, 속도 100km/h, 1kg.',
        'p1_deferred': (),
    },
    {
        'id': 'ko_04',
        'lang': 'ko',
        'text': (
            '긴 문장 테스트입니다: 회의는 이천이십칠 년 십이 월 이십일 일 오후 세 시 삼십 분에 '
            '시작하며, 전화번호는 영 구 영 팔 이 사 육 일 삼 오이고 항공편은 브이 엔 이 사 칠입니다. '
            '반드시 모든 정보를 정확하게 확인하세요.'
        ),
        'p1_deferred': (),
    },
    {
        'id': 'zh_01',
        'lang': 'zh',
        'text': '会议在2027年12月21日 08:00和15:30开始，０８：００和３６．８℃。',
        'p1_deferred': (),
    },
    {
        'id': 'zh_02',
        'lang': 'zh',
        'text': '价格是18.25美元或500元，折扣12.5%，速度100km/h，1kg。',
        'p1_deferred': (),
    },
    {
        'id': 'zh_03',
        'lang': 'zh',
        'text': '航班号VN247，航班 VN 247，代码 AB-012，邮编04782，电话0908-246-135。',
        'p1_deferred': (),
    },
    {
        'id': 'zh_04',
        'lang': 'zh',
        'text': '1000000000000元，1000000001000，3-5天，系数1e-3。',
        'p1_deferred': (),
    },
]


class TestFrozenEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.norm = HybridNormalizer(learned_identifiers=False)

    def test_seventeen_frozen_cases_recorded_and_p0_fixes(self) -> None:
        self.assertEqual(len(FROZEN_EDGE_CASES), 17)
        outputs = {}
        for case in FROZEN_EDGE_CASES:
            out, rec = self.norm(case['text'], case['lang'])
            outputs[case['id']] = (out, rec)
            self.assertIsInstance(out, str)
            self.assertTrue(len(out) > 0, case['id'])
            self.assertEqual(rec['lang'], case['lang'])

        # vi_01: VN 123 (cue-backed spaced prefix) + 67297 (code cue)
        vi01, _ = outputs['vi_01']
        self.assertIn('vê en một hai ba', vi01)
        self.assertIn('sáu bảy hai chín bảy', vi01)

        # vi_02: 50.000 đồng formatted cleanly even while 14h / độ C are P1-deferred
        vi02, _ = outputs['vi_02']
        self.assertIn('năm mươi nghìn đồng', vi02)

        # vi_03: 12:00 PM. keeps trailing sentence period and ambiguous date warns
        vi03, rec_vi03 = outputs['vi_03']
        self.assertIn('mười hai giờ trưa.', vi03)
        self.assertTrue(any('ambiguous date' in w['reason'] for w in rec_vi03['warnings']))

        # vi_04: +84 phone, AB-012 protected as identifier, 3-5 & 1e-3 kept whole as unresolved, -12,50%
        vi04, rec_vi04 = outputs['vi_04']
        self.assertIn('cộng tám bốn gạch ngang chín không tám', vi04)
        ab_spans_vi = [s for s in rec_vi04['spans'] if s['raw'] == 'AB-012']
        self.assertEqual(len(ab_spans_vi), 1)
        self.assertEqual(ab_spans_vi[0]['kind'], 'identifier')
        self.assertEqual(
            [(s['raw'], s['kind']) for s in rec_vi04['spans'] if s['raw'] in ('3-5', '1e-3')],
            [('3-5', 'unresolved'), ('1e-3', 'unresolved')],
        )
        self.assertIn('âm mười hai phẩy năm không phần trăm', vi04)
        self.assertTrue(any('unsupported numeric notation' in w['reason'] for w in rec_vi04['warnings']))

        # vi_05: invalid 15:30 PM warns and preserves 734 cardinal + €1.250,50
        vi05, rec_vi05 = outputs['vi_05']
        self.assertTrue(any('invalid clock time' in w['reason'] for w in rec_vi05['warnings']))
        self.assertIn('một nghìn hai trăm năm mươi euro năm mươi xen', vi05)
        self.assertIn('bảy trăm ba mươi bốn', vi05)

        # en_01: spaced flight, ZIP, balanced paren phone, +1 phone
        en01, _ = outputs['en_01']
        self.assertIn('two zero two', en01)
        self.assertIn('plus one', en01)
        self.assertIn('six seven two nine seven', en01)
        self.assertIn('vee en one two three', en01)

        # en_02: negative decimal in parens + euro decimal
        en02, _ = outputs['en_02']
        self.assertIn('minus twelve point five percent', en02)
        self.assertIn('one thousand two hundred fifty euros fifty cents', en02)

        # en_03: 3-5 days and 1e-3 kept whole as unresolved with unsupported warning
        _, rec_en03 = outputs['en_03']
        self.assertEqual(
            [(s['raw'], s['kind']) for s in rec_en03['spans'] if s['raw'] in ('3-5', '1e-3')],
            [('3-5', 'unresolved'), ('1e-3', 'unresolved')],
        )
        self.assertTrue(any('unsupported numeric notation' in w['reason'] for w in rec_en03['warnings']))

        # en_04: invalid 15:30 PM warning, ambiguous date warning, $1.01
        en04, rec_en04 = outputs['en_04']
        self.assertTrue(any('invalid clock time' in w['reason'] for w in rec_en04['warnings']))
        self.assertTrue(any('ambiguous date' in w['reason'] for w in rec_en04['warnings']))
        self.assertIn('one dollar one cent', en04)

        # ko_01: 12:00 PM. keeps sentence-final period (3시 30분 is P1-deferred)
        ko01, _ = outputs['ko_01']
        self.assertTrue(ko01.endswith('.'))
        self.assertIn('오후 열두 시.', ko01)

        # ko_02: 항공편 VN 247, +82 phone, 우편번호 04782, AB-012 identifier
        ko02, rec_ko02 = outputs['ko_02']
        self.assertIn('플러스 팔 이 하이픈 일 영', ko02)
        self.assertIn('영 사 칠 팔 이', ko02)
        self.assertIn('이 사 칠', ko02)
        self.assertTrue(any(s['raw'] == 'AB-012' and s['kind'] == 'identifier' for s in rec_ko02['spans']))

        # ko_03: negative percent & speed/weight
        ko03, _ = outputs['ko_03']
        self.assertIn('마이너스 십이 점 오 퍼센트', ko03)

        # zh_01: fullwidth time and decimal temperature
        zh01, _ = outputs['zh_01']
        self.assertIn('八点', zh01)
        self.assertIn('三十六点八摄氏度', zh01)

        # zh_03: 航班号VN247, 航班 VN 247, AB-012 identifier, 邮编04782, 电话0908-246-135
        zh03, rec_zh03 = outputs['zh_03']
        self.assertIn('零九零八杠二四六杠一三五', zh03)
        self.assertIn('零四七八二', zh03)
        self.assertTrue(any(s['raw'] == 'AB-012' and s['kind'] == 'identifier' for s in rec_zh03['spans']))

        # zh_04: CJK comma splits huge numbers cleanly; 3-5 & 1e-3 kept whole as unresolved
        zh04, rec_zh04 = outputs['zh_04']
        self.assertIn('一万亿元', zh04)
        self.assertIn('一万亿零一千', zh04)
        self.assertEqual(
            [(s['raw'], s['kind']) for s in rec_zh04['spans'] if s['raw'] in ('3-5', '1e-3')],
            [('3-5', 'unresolved'), ('1e-3', 'unresolved')],
        )
        self.assertTrue(any('unsupported numeric notation' in w['reason'] for w in rec_zh04['warnings']))


class TestIdentifierCandidatesAndBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.norm = HybridNormalizer(learned_identifiers=False)

    def test_identifier_candidates_plus_country_parens_and_second_phone(self) -> None:
        text = 'Phone (202) 555-0123, +1-800-555-0199.'
        cands = identifier_candidates(text, 'en')
        raws = [c['raw'] for c in cands]
        self.assertIn('(202) 555-0123', raws)
        self.assertIn('+1-800-555-0199', raws)
        for c in cands:
            self.assertEqual(text[c['start']:c['end']], c['raw'])
            self.assertIn('reason', c)

        # Second phone after cue via separator
        vi_text = 'Gọi 0908-246-135 hoặc 0912-345-678.'
        vi_cands = identifier_candidates(vi_text, 'vi')
        self.assertEqual([c['raw'] for c in vi_cands], ['0908-246-135', '0912-345-678'])

    def test_identifier_candidates_space_grouping_and_cjk_cues(self) -> None:
        vi_space = 'Số điện thoại 0908 246 135.'
        self.assertEqual([c['raw'] for c in identifier_candidates(vi_space, 'vi')], ['0908 246 135'])

        zh_cues = '电话0908-246-135，邮编04782，航班 VN 247。'
        zh_cands = identifier_candidates(zh_cues, 'zh')
        self.assertEqual([c['raw'] for c in zh_cands], ['0908-246-135', '04782', 'VN 247'])

        ko_cues = '전화번호 010-1234-5678, 우편번호 04782, 항공편 VN 247.'
        ko_cands = identifier_candidates(ko_cues, 'ko')
        self.assertEqual([c['raw'] for c in ko_cands], ['010-1234-5678', '04782', 'VN 247'])

    def test_counter_examples_protected_from_identifier_capture_and_unit_misfire(self) -> None:
        cases = [
            ('vi', 'Giảm -12.5% hôm nay.'),
            ('vi', 'Trong 3-5 ngày tới.'),
            ('vi', 'Giá 330 đồng một cái.'),
            ('vi', 'Số nhà 734 đường Lê Lợi.'),
            ('en', 'Meeting starts at 12:30 sharp.'),
            ('en', 'Random 12345 without any cue.'),
        ]
        for lang, text in cases:
            with self.subTest(lang=lang, text=text):
                self.assertEqual(identifier_candidates(text, lang), [])

        # AB-012 is classified cleanly as a single alphanumeric identifier, not split range/digits
        ab_cands = identifier_candidates('Reference AB-012 only.', 'en')
        self.assertEqual([(c['raw'], c['reason']) for c in ab_cands], [('AB-012', 'alphanumeric grammar')])

        # Confirm Số nhà 734 stays cardinal, and Vietnamese words starting with unit letters (gọi, lần)
        # are NOT misparsed as measure g/l (source_guard boundary fix)
        vi_house, _ = self.norm('Số nhà 734.', 'vi')
        self.assertEqual(vi_house, 'Số nhà bảy trăm ba mươi bốn.')

        vi_unit_collision, rec_uc = self.norm('Chuyến bay VN 123 gọi lúc 2 lần.', 'vi')
        self.assertFalse(any(s['kind'] == 'measure' for s in rec_uc['spans']), rec_uc['spans'])
        self.assertIn('vê en một hai ba gọi lúc hai lần.', vi_unit_collision)

    def test_unbalanced_parens_or_hyphens_emit_unresolved_warning(self) -> None:
        bad_cases = [
            ('en', 'Phone (202 555-0123.', 'ambiguous_unbalanced_parens'),
            ('en', 'Phone 202) 555-0123.', 'ambiguous_unbalanced_parens'),
            ('vi', 'Gọi 0908--246-135.', 'ambiguous_malformed_hyphen'),
            ('vi', 'Gọi 0908-246-.', 'ambiguous_malformed_hyphen'),
        ]
        for lang, text, expected_reason in bad_cases:
            with self.subTest(lang=lang, text=text):
                cands = identifier_candidates(text, lang)
                self.assertTrue(any(c['reason'] == expected_reason for c in cands), cands)
                spans = extract_spans(text, lang, learned_identifiers=False)
                self.assertTrue(any(s['kind'] == 'unresolved' for s in spans), spans)
                _, rec = self.norm(text, lang)
                self.assertTrue(any(w['reason'] == expected_reason for w in rec['warnings']), rec['warnings'])


class TestAmPmPeriodBoundary(unittest.TestCase):
    def setUp(self) -> None:
        self.norm = HybridNormalizer(learned_identifiers=False)

    def test_sentence_final_period_kept_across_four_languages(self) -> None:
        cases = [
            ('en', 'At 12:00 PM.', "twelve o'clock pee em."),
            ('en', 'At 4:00 p.m.', "four o'clock pee em."),
            ('vi', 'Lúc 12:00 PM.', 'mười hai giờ trưa.'),
            ('vi', 'Lúc 4:00 p.m.', 'bốn giờ chiều.'),
            ('ko', '오후 12:00 PM.', '오후 열두 시.'),
            ('zh', '12:00 PM.', '下午十二点.'),
        ]
        for lang, text, expected_fragment in cases:
            with self.subTest(lang=lang, text=text):
                out, rec = self.norm(text, lang)
                self.assertTrue(out.endswith('.'), f'{lang}: {out!r}')
                self.assertIn(expected_fragment, out)
                time_spans = [s for s in rec['spans'] if s['kind'] == 'time']
                self.assertEqual(len(time_spans), 1)
                self.assertFalse(time_spans[0]['raw'].endswith('.'))

    def test_mid_sentence_ampm_and_next_sentence_boundary(self) -> None:
        out_next, _ = self.norm('Meet at 4:00 p.m. Bring notes.', 'en')
        self.assertEqual(out_next, "Meet at four o'clock pee em. Bring notes.")

        out_mid, _ = self.norm('Meet at 4:00 p.m. today.', 'en')
        self.assertEqual(out_mid, "Meet at four o'clock pee em today.")

        out_comma, _ = self.norm('Meet at 4:00 p.m., please.', 'en')
        self.assertEqual(out_comma, "Meet at four o'clock pee em, please.")

    def test_sentence_final_scientific_and_range_before_period(self) -> None:
        out_sci, rec_sci = self.norm('Coeff 1e-3.', 'en')
        self.assertEqual([(s['raw'], s['kind']) for s in rec_sci['spans']], [('1e-3', 'unresolved')])
        self.assertTrue(out_sci.endswith('.'))

        out_rng, rec_rng = self.norm('Ratio 3-5.', 'en')
        self.assertEqual([(s['raw'], s['kind']) for s in rec_rng['spans']], [('3-5', 'unresolved')])
        self.assertTrue(out_rng.endswith('.'))


class TestSeededBoundaries(unittest.TestCase):
    def test_seeded_phone_and_ampm_boundaries(self) -> None:
        rng = random.Random(20261113)
        norm = HybridNormalizer(learned_identifiers=False)
        for _ in range(12):
            a = rng.randint(200, 999)
            b = rng.randint(100, 999)
            c = rng.randint(1000, 9999)
            hr = rng.randint(1, 11)
            mn = rng.randint(10, 59)
            text = f'Call ({a}) {b}-{c} at {hr}:{mn:02d} p.m.'
            out, rec = norm(text, 'en')
            self.assertTrue(out.endswith('.'), out)
            self.assertTrue(any(s['kind'] == 'identifier' for s in rec['spans']))
            self.assertTrue(any(s['kind'] == 'time' for s in rec['spans']))


class TestCrfActiveAndMissingFallback(unittest.TestCase):
    def test_crf_active_and_missing_pycrfsuite_agree_on_protected_boundaries(self) -> None:
        import sys
        from unittest.mock import patch
        import s2s_edge.hybrid_textnorm as ht

        crf_norm = HybridNormalizer(learned_identifiers=True)
        for case in FROZEN_EDGE_CASES:
            out_crf, rec_crf = crf_norm(case['text'], case['lang'])
            # Status warning must not be emitted when CRF loads, and -12.5% / 3-5 must not be captured as identifier
            id_raws = [s['raw'] for s in rec_crf['spans'] if s['kind'] == 'identifier']
            self.assertNotIn('-12.5%', id_raws)
            self.assertNotIn('3-5', id_raws)
            self.assertTrue(len(out_crf) > 0)

        saved_crf = ht._CRF
        try:
            ht._CRF = None
            with patch.dict(sys.modules, {'pycrfsuite': None}):
                fallback_norm = HybridNormalizer(learned_identifiers=True)
                out_fb, rec_fb = fallback_norm(
                    'Flight VN 123, ZIP 67297, phone (202) 555-0123, +1-800-555-0199.',
                    'en',
                )
                self.assertIn('two zero two', out_fb)
                self.assertIn('six seven two nine seven', out_fb)
                self.assertEqual(rec_fb['identifier_model'], 'unavailable; context grammars only')
                self.assertTrue(any(w.get('reason', '').startswith('unavailable') for w in rec_fb['warnings']))
                # No raw numeric span warnings on valid input
                self.assertEqual([w for w in rec_fb['warnings'] if w.get('raw') is not None], [])
        finally:
            ht._CRF = saved_crf


if __name__ == '__main__':
    unittest.main()
