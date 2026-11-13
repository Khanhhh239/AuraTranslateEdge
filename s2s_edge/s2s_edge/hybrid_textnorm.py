"""Typed, value-preserving TN. Grammars recognize classes, never fixture strings.

An optional existing CRF proposes identifier spans; it cannot generate text.
All rendering is deterministic from immutable numeric fields. Unknown/ambiguous
input is retained or spelled conservatively and reported, rather than guessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
import json
import os
import re
import time
import unicodedata

from .format_locales import (CURRENCY, CURRENCY_ALIASES, DIGITS, IDENTIFIER_CUES,
                             KO_HOURS, LETTERS, MONTHS_EN, UNITS)
from .textnorm import _vi3, _zh4, en_num

LANGS = ('vi', 'en', 'ko', 'zh')
NUM = r'[+\-−]?\d+(?:[.,．]\d+)*'
_CRF = None
_CALL_ONLY_CUES = {'gọi', 'call'}
_EXCLUDED_URL_TRAIL = '.,!?;:)\'"]}。，！？；：、'
_URL_RX = re.compile(r'(?:https?://|www\.)[^\s<>"\']+', re.IGNORECASE)
_EMAIL_RX = re.compile(r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])')
_VERSION_PREFIX_RX = re.compile(r'(?<![A-Za-z0-9])[vV]\d+(?:\.\d+){1,3}(?:-[A-Za-z0-9._-]+)?(?![A-Za-z0-9.])')
_VERSION_CUE_RX = re.compile(
    r'(?:(?<![A-Za-z])(?:version|ver\.?|release)|phiên bản|버전|版本)\s*(\d+(?:\.\d+){1,3}(?:-[A-Za-z0-9._-]+)?)(?![A-Za-z0-9.])',
    re.IGNORECASE,
)
_SENT_SPLIT_RX = re.compile(r'[.!?。！？;；\n]')
_LIST_SEP_RX = re.compile(r'(?:\s*[,，、]\s*(?:(?:and|or|và|hoặc|또는|및|和|或)\s+)?|\s+(?:and|or|và|hoặc|또는|및|和|或)\s+)$', re.IGNORECASE)
_LIST_ITEM_RX = re.compile(
    r'(?:\+?\s*(?:\d+\s*)?(?:\(\d{1,5}\)\s*)?\d[\d\s()\-]*\d|[A-Z]{2,4}\s+\d+(?:-\d+)*|[A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*|\d{3,})$'
)
_SPACED_PREFIX_RX = re.compile(r'(?<![A-Za-z\d])[A-Z]{2,4}\s+\d+(?:-\d+)*(?![A-Za-z\d])')
_ALNUM_CODE_RX = re.compile(r'(?<![A-Za-z\d])(?:[A-Za-z]+[A-Za-z\d\-]*\d[A-Za-z\d\-]*|\d+[A-Za-z][A-Za-z\d\-]*)(?![A-Za-z\d])')
_PHONE_OR_NUM_ID_RX = re.compile(
    r'(?<![A-Za-z\d])(?:\+\s*)?(?:\(\d[\d\s\-]*\)\s*[\d\s()\-]*\d|\(\d[\d\s\-]+\d|\d[\d\s()\-]*\d|\d+)(?![A-Za-z\d])'
)
_SCI_EXACT_RX = re.compile(r'[+\-−]?\d+(?:[.,．]\d+)?[eE][+\-−]?\d+')
_AMPM_GROUP = (
    r'(?:'
    r'[aApP]\.[mM]\.(?=[,;:\)!?\]\}\'"’”°。！？，、；：]|\.(?:\s|$)|\s+(?-i:[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]))'
    r'|[aApP]\.?[mM]'
    r')'
)
_CUE_WORDS = frozenset(c.casefold() for c in IDENTIFIER_CUES)
_SEPARATOR_POLICY = (
    'locale decimal (. for en/ko/zh, , for vi); '
    'CJK fullwidth comma (，) is a clause boundary; '
    'ambiguous repeated or malformed separators spell characters with warning'
)
_CURRENCY_ALT = '|'.join(re.escape(s) for s in sorted(CURRENCY_ALIASES, key=len, reverse=True))
_UNITS_ALT = '|'.join(re.escape(s) for s in sorted(UNITS, key=len, reverse=True))
_TYPED_PATTERNS = (
    ('date', r'(?<!\d)(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)|\d{4}年\s*\d{1,2}月\s*\d{1,2}日|\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}/\d{1,2}(?:/\d{2,4})?(?!\d)|\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?)'),
    ('date', r'\b(?:' + '|'.join(MONTHS_EN) + r')\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b'),
    ('year', r'(?<!\d)\d{4}\s*(?:年|년)'),
    ('time', r'(?<![\d:：])(?P<hour>\d{1,2})\s*[:：hH]\s*(?P<minute>\d{2})(?:[:：](?P<second>\d{2}))?(?:\s*(?P<ampm>' + _AMPM_GROUP + r')(?![A-Za-z\u00C0-\u024F\u1E00-\u1EFF\d]))?(?!\d)'),
    ('money', r'(?P<currency>' + _CURRENCY_ALT + r')\s*(?P<number>' + NUM + r')(?!\d)'),
    ('money', r'(?P<number>' + NUM + r')\s*(?P<currency>' + _CURRENCY_ALT + r')(?![A-Za-z\u00C0-\u024F\u1E00-\u1EFF])'),
    ('percent', r'(?P<number>' + NUM + r')\s*[%％]'),
    ('measure', r'(?P<number>' + NUM + r')\s*(?P<unit>' + _UNITS_ALT + r')(?![A-Za-z\u00C0-\u024F\u1E00-\u1EFF])'),
    ('unsupported', r'(?<![A-Za-z0-9_.\-/])[+\-−]?\d+(?:[.,．]\d+)?[eE][+\-−]?\d+(?![A-Za-z0-9_\-/]|\.\d)'),
    ('unsupported', r'(?<![A-Za-z0-9_.\-/])\d+(?:[.,．]\d+)?(?:\s*[-–]\s*\d+(?:[.,．]\d+)?)+(?![A-Za-z0-9_\-/]|\.\d)'),
)
_TYPED_COMPILED = [(kind, re.compile(pat, flags=re.IGNORECASE)) for kind, pat in _TYPED_PATTERNS]
_PRIORITY = {'date': 0, 'year': 1, 'time': 1, 'money': 2, 'percent': 2, 'measure': 2, 'identifier': 3, 'unsupported': 4, 'number': 5}


def join(lang, words):
    return ('' if lang == 'zh' else ' ').join(w for w in words if w)


def digits(raw, lang):
    return join(lang, [DIGITS[lang][int(c)] for c in raw])


def cardinal(n, lang):
    if n == 0:
        return DIGITS[lang][0]
    if n < 0:
        return join(lang, [dict(vi='âm', en='minus', ko='마이너스', zh='负')[lang], cardinal(-n, lang)])
    if lang == 'en':
        for scale, word in [(10**18, 'quintillion'), (10**15, 'quadrillion'), (10**12, 'trillion')]:
            if n >= scale:
                return cardinal(n//scale, lang) + ' ' + word + (' ' + cardinal(n%scale, lang) if n%scale else '')
        return en_num(n)
    if lang == 'zh':
        groups = []; i = 0
        while n:
            n, r = divmod(n, 10000)
            if r: groups.append((i, r))
            i += 1
        out = ''; previous = None
        for i, r in reversed(groups):
            if previous is not None and (previous-i > 1 or r < 1000): out += '零'
            scale = ('万' if i % 2 else '') + '亿'*(i//2)
            out += _zh4(r, first=not out) + scale
            previous = i
        return out
    if lang == 'vi':
        parts = []; i = 0
        while n:
            n, r = divmod(n, 1000)
            unit = (('', 'nghìn', 'triệu')[i % 3] + (' tỷ' * (i//3))).strip()
            if r:
                parts.append((_vi3(r, bool(n)) + ' ' + unit).strip())
            i += 1
        return ' '.join(reversed(parts))
    parts = []; i = 0
    scales = ['', '만', '억', '조', '경', '해']
    while n:
        n, r = divmod(n, 10000)
        if r:
            group = ''
            for p, c in enumerate(str(r).zfill(4)):
                d = int(c); pos = 3-p
                if d:
                    group += ('' if d == 1 and pos else DIGITS['ko'][d]) + ['', '십', '백', '천'][pos]
            parts.append(group + scales[i])
        i += 1
    return ''.join(reversed(parts))


def characters(raw, lang):
    table = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', LETTERS[lang]))
    symbols = {
        '+': ('cộng', 'plus', '플러스', '加'), '-': ('gạch ngang', 'dash', '하이픈', '杠'),
        '/': ('gạch chéo', 'slash', '슬래시', '斜杠'), ':': ('hai chấm', 'colon', '콜론', '冒号'),
        '.': ('chấm', 'dot', '점', '点'), ',': ('phẩy', 'comma', '쉼표', '逗号'),
        '(': ('mở ngoặc', 'open parenthesis', '괄호 열기', '左括号'),
        ')': ('đóng ngoặc', 'close parenthesis', '괄호 닫기', '右括号'),
    }
    out = []
    for c in raw:
        if c.isdecimal(): out.append(DIGITS[lang][int(c)])
        elif c.upper() in table: out.append(table[c.upper()])
        elif c in symbols: out.append(symbols[c][LANGS.index(lang)])
        elif not c.isspace(): out.append(c)
    return join(lang, out)


def numeric_fields(raw, lang):
    """Parse separators without float conversion; preserve fractional precision."""
    raw = ''.join(str(int(c)) if c.isdecimal() else c for c in raw).translate(str.maketrans({'，':',', '．':'.'}))
    sign = '-' if raw.startswith(('-', '−')) else '+' if raw.startswith('+') else ''
    s = raw.lstrip('+-−')
    decimal = ',' if lang == 'vi' else '.'
    if '.' in s and ',' in s:
        sep = '.' if s.rfind('.') > s.rfind(',') else ','
        other = ',' if sep == '.' else '.'
        whole, frac = s.rsplit(sep, 1)
        if sep in whole or not re.fullmatch(r'\d{1,3}(?:' + re.escape(other) + r'\d{3})+', whole):
            raise ValueError('invalid mixed number separators')
        whole = whole.replace(other, '')
    elif '.' in s or ',' in s:
        sep = '.' if '.' in s else ','
        groups = s.split(sep)
        if sep != decimal and all(len(g) == 3 for g in groups[1:]) and 1 <= len(groups[0]) <= 3:
            whole, frac = ''.join(groups), ''
        elif len(groups) == 2:
            whole, frac = groups
        else:
            raise ValueError('ambiguous or invalid repeated decimal separator')
    else:
        whole, frac = s, ''
    return {'sign': sign, 'whole': whole, 'fraction': frac}


def number_words(value, lang):
    whole, frac = value['whole'], value['fraction']
    # Very large values are intentionally spelled rather than overrun scale tables.
    out = cardinal(int(whole), lang) if len(whole) <= 18 else digits(whole, lang)
    if frac:
        out = join(lang, [out, dict(vi='phẩy', en='point', ko='점', zh='点')[lang], digits(frac, lang)])
    if value['sign']:
        prefix = dict(vi='âm', en='minus', ko='마이너스', zh='负') if value['sign'] == '-' else dict(vi='dương', en='plus', ko='플러스', zh='正')
        out = join(lang, [prefix[lang], out])
    return out


def ordinal_en(n):
    special = {1:'first', 2:'second', 3:'third', 5:'fifth', 8:'eighth', 9:'ninth', 12:'twelfth'}
    if n in special: return special[n]
    if n < 20: return cardinal(n, 'en') + 'th'
    if n % 10 == 0: return cardinal(n, 'en')[:-1] + 'ieth'
    return cardinal(n//10*10, 'en') + ' ' + ordinal_en(n%10)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    raw: str
    value: dict
    spoken: str
    source: str = 'grammar'


def _overlaps_any(a: int, b: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a < rb and b > ra for ra, rb in ranges)


def _excluded_regions(text: str) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    for m in _URL_RX.finditer(text):
        a, b = m.start(), m.end()
        while b > a and text[b - 1] in _EXCLUDED_URL_TRAIL:
            b -= 1
        if b > a:
            regions.append((a, b))
    for m in _EMAIL_RX.finditer(text):
        regions.append((m.start(), m.end()))
    for m in _VERSION_PREFIX_RX.finditer(text):
        regions.append((m.start(), m.end()))
    for m in _VERSION_CUE_RX.finditer(text):
        regions.append((m.start(1), m.end(1)))
    regions.sort()
    return regions


def _match_cue(clause_prefix: str) -> str | None:
    tail = clause_prefix.casefold()[-64:]
    for cue in IDENTIFIER_CUES:
        cjk = any('\u3400' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in cue)
        pattern = re.escape(cue) if cjk else r'(?<!\w)' + re.escape(cue) + r'(?!\w)'
        m = re.search(pattern + r'([^\d,，;；.!?。！？\n]{0,24})$', tail)
        if not m:
            continue
        connector = m.group(1)
        if cue == 'gọi' and re.match(r'^\s+là(?:\s|$)', connector):
            continue
        return cue
    return None


def _has_cue_or_list_context(text: str, start: int) -> tuple[bool, str, str | None]:
    clause = _SENT_SPLIT_RX.split(text[:start])[-1]
    direct = _match_cue(clause)
    if direct is not None:
        return True, 'context', direct
    cur = clause
    for _ in range(4):
        sep_m = _LIST_SEP_RX.search(cur)
        if not sep_m:
            break
        before_sep = cur[:sep_m.start()].rstrip()
        item_m = _LIST_ITEM_RX.search(before_sep)
        if not item_m:
            break
        cur = before_sep[:item_m.start()]
        cue = _match_cue(cur)
        if cue is not None:
            return True, 'context_list', cue
    return False, '', None


def _parens_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _check_paren_and_hyphen(text: str, a: int, b: int) -> tuple[int, int, str | None]:
    raw = text[a:b]
    if raw.startswith('(') and raw.count('(') == 1 and raw.count(')') == 0 and b < len(text) and text[b] == ')':
        a += 1
        raw = text[a:b]
    if '--' in raw:
        return a, b, 'ambiguous_malformed_hyphen'
    if b < len(text) and text[b] == '-' and (b + 1 >= len(text) or not text[b + 1].isalnum()):
        return a, b + 1, 'ambiguous_malformed_hyphen'
    if not _parens_balanced(raw):
        return a, b, 'ambiguous_unbalanced_parens'
    return a, b, None


def _is_phone_structure(raw: str, digit_count: int, groups: list[str]) -> bool:
    if digit_count < 7 or digit_count > 15:
        return False
    if raw.startswith('+') and (len(groups) >= 2 or '(' in raw):
        return True
    if re.search(r'\(\d{2,4}\)', raw) and len(groups) >= 2:
        return True
    if '-' in raw and ' ' not in raw and '(' not in raw and len(groups) >= 3 and digit_count >= 9:
        if all(2 <= len(g) <= 5 for g in groups):
            return True
    return False


def _typed_candidates(text: str, excluded: list[tuple[int, int]]) -> list[tuple[int, int, str, re.Match, str]]:
    out: list[tuple[int, int, str, re.Match, str]] = []
    for kind, rx in _TYPED_COMPILED:
        for m in rx.finditer(text):
            a, b = m.start(), m.end()
            if _overlaps_any(a, b, excluded):
                continue
            raw = text[a:b]
            if kind in ('money', 'percent', 'measure', 'unsupported') and raw.startswith(('-', '−', '+')) and a > 0 and text[a - 1].isdecimal():
                continue
            out.append((a, b, kind, m, 'grammar'))
    return out


def _higher_priority_blocked(typed_cands: list[tuple[int, int, str, re.Match, str]], excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blocked = list(excluded)
    selected_hi: list[tuple[int, int]] = []
    for a, b, kind, _, _ in sorted(typed_cands, key=lambda c: (_PRIORITY[c[2]], c[0], -(c[1] - c[0]))):
        if _PRIORITY[kind] >= _PRIORITY['identifier']:
            continue
        if _overlaps_any(a, b, selected_hi):
            continue
        selected_hi.append((a, b))
        blocked.append((a, b))
    return blocked


def _collect_identifier_candidates(text: str, blocked: list[tuple[int, int]]) -> list[tuple[int, int, str, str]]:
    cands: list[tuple[int, int, str, str]] = []
    # 1. Spaced uppercase prefix codes with cue (e.g., "Flight VN 123", "Chuyến bay của tôi là VN 123").
    for m in _SPACED_PREFIX_RX.finditer(text):
        a, b = m.start(), m.end()
        if _overlaps_any(a, b, blocked):
            continue
        if m.group().split()[0].casefold() in _CUE_WORDS:
            continue
        has_ctx, ctx_reason, _ = _has_cue_or_list_context(text, a)
        if has_ctx:
            reason = 'spaced_prefix_code' if ctx_reason == 'context' else 'context_list'
            cands.append((a, b, 'identifier', reason))

    # 2. Alphanumeric codes (e.g., "AB-012", "VN247", "X9Q7").
    for m in _ALNUM_CODE_RX.finditer(text):
        a, b = m.start(), m.end()
        raw = m.group().rstrip('-')
        b = a + len(raw)
        if b <= a or _overlaps_any(a, b, blocked):
            continue
        if _SCI_EXACT_RX.fullmatch(raw):
            continue
        letters = re.findall(r'[A-Za-z]', raw)
        digits_in = re.findall(r'\d', raw)
        if not (letters and digits_in):
            continue
        a, b, err = _check_paren_and_hyphen(text, a, b)
        if not err and a > 0 and text[a - 1] == '(' and (b >= len(text) or text[b] != ')'):
            if ')' not in text[b:b + 12]:
                a -= 1
                err = 'ambiguous_unbalanced_parens'
        if err:
            cands.append((a, b, 'ambiguous_identifier', err))
            continue
        has_ctx, ctx_reason, _ = _has_cue_or_list_context(text, a)
        if has_ctx or len(letters) >= 2:
            cands.append((a, b, 'identifier', ctx_reason if has_ctx else 'alphanumeric grammar'))

    # 3. Phone and numeric identifier sequences.
    for m in _PHONE_OR_NUM_ID_RX.finditer(text):
        a, b = m.start(), m.end()
        if _overlaps_any(a, b, blocked):
            continue
        raw = text[a:b].rstrip(' -')
        b = a + len(raw)
        if b <= a:
            continue
        if ' -' in raw or '- ' in raw:
            continue
        a, b, err = _check_paren_and_hyphen(text, a, b)
        raw = text[a:b]
        groups = re.findall(r'\d+', raw)
        digit_count = sum(c.isdecimal() for c in raw)
        has_ctx, ctx_reason, cue = _has_cue_or_list_context(text, a)
        phone_shape = _is_phone_structure(raw, digit_count, groups)
        if err:
            if has_ctx or phone_shape or raw.startswith(('+', '(')) or digit_count >= 7 or '--' in raw:
                cands.append((a, b, 'ambiguous_identifier', err))
            continue
        if phone_shape:
            cands.append((a, b, 'identifier', ctx_reason if has_ctx else 'phone_grammar'))
            continue
        if has_ctx:
            if cue in _CALL_ONLY_CUES and digit_count < 3 and '+' not in raw and '(' not in raw:
                continue
            is_range_shape = (
                '-' in raw and ' ' not in raw and not raw.startswith('+') and '(' not in raw
                and len(groups) == 2 and digit_count < 6 and not any(g.startswith('0') for g in groups)
            )
            if is_range_shape and cue in ('phone', 'telephone', 'mobile', 'hotline', 'tel', 'fax',
                                         'phone number', 'telephone number', 'số điện thoại', 'điện thoại',
                                         'di động', 'gọi số', 'gọi', '전화번호', '전화', '휴대폰', '연락처',
                                         '电话号码', '电话', '手机号', '手机', '热线', '座机', '拨打', '致电'):
                continue
            cands.append((a, b, 'identifier', ctx_reason))
    return cands


def identifier_candidates(text: str, lang: str = 'en') -> list[dict]:
    """Return structured identifier candidates {start, end, raw, reason} using shared lexical boundaries."""
    if lang not in LANGS:
        raise ValueError('unsupported language')
    excluded = _excluded_regions(text)
    typed_cands = _typed_candidates(text, excluded)
    blocked = _higher_priority_blocked(typed_cands, excluded)
    raw_cands = _collect_identifier_candidates(text, blocked)
    selected: list[tuple[int, int, str]] = []
    for a, b, _, reason in sorted(raw_cands, key=lambda c: (c[0], -(c[1] - c[0]))):
        if _overlaps_any(a, b, [(s[0], s[1]) for s in selected]):
            continue
        selected.append((a, b, reason))
    selected.sort(key=lambda s: s[0])
    return [{'start': a, 'end': b, 'raw': text[a:b], 'reason': reason} for a, b, reason in selected]


def extract_spans(text: str, lang: str, *, date_order: str | None = None, learned_identifiers: bool = False) -> list[dict]:
    """Pure-Python typed span extraction sharing formatter lexical and separator policies."""
    norm = HybridNormalizer(date_order=date_order, learned_identifiers=learned_identifiers)
    _, record = norm(text, lang)
    out: list[dict] = []
    for sp in record['spans']:
        val = sp['value']
        reason = val.get('reason') or val.get('warning') if isinstance(val, dict) else None
        out.append({
            'start': sp['start'],
            'end': sp['end'],
            'kind': sp['kind'],
            'raw': sp['raw'],
            'value': val,
            'source': sp['source'],
            'reason': reason,
        })
    return out


class HybridNormalizer:
    def __init__(self, date_order=None, learned_identifiers=True):
        self.date_order = date_order
        self.learned_identifiers = learned_identifiers
        self.records = []
        self.identifier_model_status = 'not_requested'

    @staticmethod
    def _context_identifier(text, start):
        has_ctx, _, _ = _has_cue_or_list_context(text, start)
        return has_ctx

    def _learned_spans(self, text, lang, excluded=None):
        global _CRF
        if not self.learned_identifiers or lang not in ('en', 'vi'):
            return []
        try:
            if _CRF is None:
                from .identifier_tagger import IdentifierTagger, find_crf_model
                _CRF = IdentifierTagger(find_crf_model())
            self.identifier_model_status = 'CRF loaded'
            found = []; cursor = 0
            for raw in _CRF(text, lang):
                a = text.find(raw, cursor)
                if a < 0: continue
                cursor = a+len(raw)
                if excluded and _overlaps_any(a, cursor, excluded):
                    continue
                if re.fullmatch(r'[A-Za-z\d+() -]+', raw) and any(c.isdecimal() for c in raw):
                    if _parens_balanced(raw) and '--' not in raw and not re.fullmatch(r'\d+\s*-\s*\d+', raw):
                        found.append((a, cursor))
            return found
        except (ImportError, OSError, RuntimeError, ValueError):
            self.identifier_model_status = 'unavailable; context grammars only'
            return []

    def render(self, kind, raw, m, lang):
        if kind == 'unsupported':
            raise ValueError('unsupported numeric notation; do not infer its semantic value')
        if kind == 'ambiguous_identifier':
            reason = m if isinstance(m, str) else 'ambiguous identifier boundary'
            raise ValueError(reason)
        if kind == 'date':
            nums = re.findall(r'\d+', raw)
            order = self.date_order or os.environ.get('S2S_DATE_ORDER', '')
            month_word = next((i+1 for i, name in enumerate(MONTHS_EN) if name.casefold() in raw.casefold()), None)
            if month_word:
                month = month_word; day = int(nums[0]); y = int(nums[1]) if len(nums) == 2 else 2000
                has_year = len(nums) == 2
            elif len(nums[0]) == 4:
                y, month, day = map(int, nums)
                has_year = True
            else:
                if 'tháng' in raw.casefold(): order = 'DMY'
                if not order:
                    if int(nums[0]) > 12: order = 'DMY'
                    elif int(nums[1]) > 12: order = 'MDY'
                    else: raise ValueError('ambiguous date; configure S2S_DATE_ORDER=DMY or MDY')
                if order not in ('DMY', 'MDY'): raise ValueError('unsupported date order')
                day, month = map(int, nums[:2]) if order == 'DMY' else map(int, reversed(nums[:2]))
                y = int(nums[2]) if len(nums) == 3 else 2000  # validation only for yearless dates
                if len(nums) == 3 and len(nums[2]) != 4: raise ValueError('two-digit year is ambiguous')
                has_year = len(nums) == 3
            date(y, month, day)
            if lang == 'vi': spoken = join(lang, ['ngày', cardinal(day, lang), 'tháng', cardinal(month, lang), 'năm' if has_year else '', cardinal(y, lang) if has_year else ''])
            elif lang == 'en': spoken = MONTHS_EN[month-1] + ' ' + ordinal_en(day) + (' ' + cardinal(y, lang) if has_year else '')
            elif lang == 'ko': spoken = (cardinal(y, lang)+' 년 ' if has_year else '') + cardinal(month, lang)+' 월 '+cardinal(day, lang)+' 일'
            else: spoken = (digits(str(y), lang)+'年' if has_year else '') + cardinal(month, lang)+'月'+cardinal(day, lang)+'日'
            return {'year': y if has_year else None, 'month': month, 'day': day}, spoken
        if kind == 'year':
            year = re.search(r'\d+', raw).group()
            return {'year': int(year)}, (digits(year, lang) if lang == 'zh' else cardinal(int(year), lang)) + ('年' if lang == 'zh' else ' 년')
        if kind == 'time':
            h, minute = int(m['hour']), int(m['minute'])
            second = int(m['second']) if m.groupdict().get('second') else None
            ampm = (m.groupdict().get('ampm') or '').replace('.', '').upper()
            if not 0 <= minute < 60 or not 0 <= h < 24 or (ampm and not 1 <= h <= 12) or (second is not None and not 0 <= second < 60):
                raise ValueError('invalid clock time')
            tail = {'vi': {'AM':'sáng', 'PM':'chiều'}, 'en': {'AM':'ay em', 'PM':'pee em'}, 'ko': {'AM':'오전', 'PM':'오후'}, 'zh': {'AM':'上午', 'PM':'下午'}}[lang].get(ampm, '')
            if lang == 'vi' and h == 12 and ampm: tail = 'trưa' if ampm == 'PM' else 'đêm'
            if lang == 'vi': spoken = join(lang, [cardinal(h, lang), 'giờ', cardinal(minute, lang) if minute else '', 'phút' if minute else '', cardinal(second, lang)+' giây' if second is not None else '', tail])
            elif lang == 'en': spoken = join(lang, [cardinal(h, lang), ('oh ' if minute < 10 else '')+cardinal(minute, lang) if minute else "o'clock", cardinal(second, lang)+' seconds' if second is not None else '', tail])
            elif lang == 'ko': spoken = join(lang, [tail, KO_HOURS[h] if 0 < h <= 12 else cardinal(h, lang), '시', cardinal(minute, lang)+' 분' if minute else '', cardinal(second, lang)+' 초' if second is not None else ''])
            else: spoken = join(lang, [tail, cardinal(h, lang), '点', cardinal(minute, lang)+'分' if minute else '', cardinal(second, lang)+'秒' if second is not None else ''])
            return {'hour': h, 'minute': minute, 'second': second, 'ampm': ampm}, spoken
        if kind == 'identifier':
            # Only whitespace is grouping; retain plus/dash/parentheses as symbols.
            canonical = ''.join(c for c in raw if not c.isspace())
            return {'characters': canonical}, characters(canonical, lang)
        if kind == 'money':
            alias = m['currency']; currency = {k.casefold():v for k,v in CURRENCY_ALIASES.items()}[alias.casefold()]
            value = numeric_fields(m['number'], lang)
            major, minor = CURRENCY[currency][lang]
            frac = value['fraction']
            if minor and len(frac) in (1, 2):
                n, cents = int(value['whole']), int(frac.ljust(2, '0'))
                if lang == 'en':
                    major = major if n == 1 else major+'s'
                    minor = ('pence' if minor == 'penny' else minor+'s') if cents != 1 else minor
                unsigned = dict(value, fraction='')
                spoken = join(lang, [number_words(unsigned, lang), major, cardinal(cents, lang), minor])
            else:
                if lang == 'en' and Decimal(value['whole']+'.'+(frac or '0')) != 1 and currency in ('USD', 'EUR', 'GBP'): major += 's'
                spoken = join(lang, [number_words(value, lang), major])
            return {'currency': currency, 'number': value}, spoken
        if kind in ('percent', 'measure'):
            value = numeric_fields(m['number'], lang)
            if kind == 'percent':
                unit = ('phần trăm', 'percent', '퍼센트', '百分之')[LANGS.index(lang)]
                spoken = join(lang, [unit, number_words(value, lang)] if lang == 'zh' else [number_words(value, lang), unit])
            else:
                unit = m['unit'].casefold().replace(' ', '')
                label = UNITS[unit][LANGS.index(lang)]
                if lang == 'en' and Decimal(value['whole']+'.'+(value['fraction'] or '0')) == 1:
                    head, *rest = label.split()
                    if head.endswith('s'): head = head[:-1]
                    label = ' '.join([head]+rest)
                spoken = join(lang, [number_words(value, lang), label])
            return {'number': value, 'unit': '%' if kind == 'percent' else unit}, spoken
        value = numeric_fields(raw, lang)
        return value, number_words(value, lang)

    def __call__(self, text, lang):
        if lang not in LANGS: raise ValueError('unsupported language')
        start_time = time.perf_counter()
        excluded = _excluded_regions(text)
        candidates = _typed_candidates(text, excluded)
        blocked = _higher_priority_blocked(candidates, excluded)
        for a, b, kind, reason in _collect_identifier_candidates(text, blocked):
            candidates.append((a, b, kind, reason if kind == 'ambiguous_identifier' else None, reason))
        for a, b in self._learned_spans(text, lang, excluded):
            candidates.append((a, b, 'identifier', None, 'CRF proposal'))
        for m in re.finditer(NUM, text):
            a, b = m.start(), m.end()
            if _overlaps_any(a, b, excluded):
                continue
            if a > 0 and text[a - 1].isascii() and text[a - 1].isalpha():
                continue
            if b < len(text) and text[b].isascii() and text[b].isalpha():
                continue
            if text[a:b].startswith(('-', '−', '+')) and a > 0 and text[a - 1].isdecimal():
                continue
            candidates.append((a, b, 'number', m, 'grammar'))
        priority = dict(_PRIORITY, ambiguous_identifier=_PRIORITY['identifier'])
        selected = []
        for a, b, kind, match, source in sorted(candidates, key=lambda c: (priority[c[2]], c[0], -(c[1]-c[0]))):
            if any(a < s.end and b > s.start for s in selected): continue
            raw = text[a:b]
            try:
                value, spoken = self.render(kind, raw, match, lang)
                if kind == 'date' and lang == 'vi' and re.search(r'\bngày\s*$', text[:a], re.IGNORECASE):
                    spoken = spoken.removeprefix('ngày ')
                if kind == 'number' and value['whole'].startswith('0') and len(value['whole']) > 1 and not value['fraction']:
                    kind = 'identifier'
                    value = {'characters': raw, 'warning': 'untyped leading-zero sequence; preserve characters'}
                    spoken = characters(raw, lang)
            except (ValueError, IndexError, KeyError) as exc:
                value, spoken = {'reason': str(exc)}, characters(raw, lang)
                kind = 'unresolved'
            selected.append(Span(a, b, kind, raw, value, spoken, source))
        selected.sort(key=lambda s: s.start)
        cursor = 0; parts = []; spoken_ranges = []; warnings = []
        for s in selected:
            assert s.start >= cursor and text[s.start:s.end] == s.raw
            parts.append(text[cursor:s.start]); begin = sum(len(p) for p in parts)
            parts.append(s.spoken); spoken_ranges.append((begin, begin+len(s.spoken)))
            if s.kind == 'unresolved': warnings.append({'raw': s.raw, 'reason': s.value['reason'], 'policy': 'spell characters, do not infer value'})
            if 'warning' in s.value:
                warnings.append({'raw': s.raw, 'reason': s.value['warning'], 'policy': 'spell characters'})
            cursor = s.end
        if self.identifier_model_status.startswith('unavailable'):
            warnings.append({'reason': self.identifier_model_status, 'policy': 'install python-crfsuite for learned identifier proposals'})
        parts.append(text[cursor:]); output = ''.join(parts)
        record = {'stage': 'tts_hybrid', 'lang': lang, 'input': text, 'output': output,
                  'spans': [asdict(s) for s in selected], 'protected_ranges': spoken_ranges,
                  'warnings': warnings, 'seconds': time.perf_counter()-start_time,
                  'identifier_model': self.identifier_model_status,
                  'separator_policy': _SEPARATOR_POLICY,
                  'validation': 'raw span identity; typed deterministic renderer; no generative numeric rewriting'}
        self.records.append(record)
        if len(self.records) > 128: del self.records[:-128]
        return output, record


_NORMALIZER = None


def normalize(text, lang):
    global _NORMALIZER
    if _NORMALIZER is None: _NORMALIZER = HybridNormalizer()
    result, record = _NORMALIZER(text, lang)
    log = os.environ.get('S2S_TN_AUDIT')
    if log:
        with open(log, 'a', encoding='utf-8') as f: f.write(json.dumps(record, ensure_ascii=False)+'\n')
    return result, record
