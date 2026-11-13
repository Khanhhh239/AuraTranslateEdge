"""Normalization dispatch and chunk planning shared by every TTS language."""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from . import hybrid_textnorm, neural_textnorm
from .textnorm import normalize as legacy_normalize


def resolve_mode(tts_format=None):
    value = os.environ.get('S2S_TTS_FORMAT', 'hybrid') if tts_format is None else str(tts_format)
    if value not in ('hybrid', 'legacy', 'neural'):
        raise ValueError('S2S_TTS_FORMAT must be hybrid, legacy or neural')
    return value


def mode(tts_format=None):
    return resolve_mode(tts_format)


def prepare(text, lang, mode=None):
    selected = resolve_mode(mode)
    if selected == 'hybrid':
        return hybrid_textnorm.normalize(text, lang)
    if selected == 'neural' and lang in ('en', 'vi'):
        result = neural_textnorm.get()(text, lang, stage='tts')
    else:
        result = legacy_normalize(text, lang)
    return result, {'stage': 'tts_'+selected, 'input': text, 'output': result,
                    'spans': [], 'protected_ranges': [], 'warnings': []}


def _map_protected_ranges(record, text, lang):
    """Map normalized span ranges onto `text` without ambiguous str.find guessing."""
    if not record:
        return [], []
    spans = record.get('spans', [])
    if not spans:
        return [], []
    if record.get('tts_input_text') == text and 'tts_input_protected_ranges' in record:
        mapped = []
        ranges = []
        for idx, (sp, rng) in enumerate(zip(spans, record['tts_input_protected_ranges'])):
            if rng is None:
                mapped.append({'span_index': idx, 'raw': sp.get('raw', ''), 'tts_range': None, 'status': 'unmapped'})
            else:
                a, b = int(rng[0]), int(rng[1])
                ranges.append((a, b))
                mapped.append({'span_index': idx, 'raw': sp.get('raw', ''), 'tts_range': [a, b], 'status': 'mapped'})
        return ranges, mapped

    norm_out = record.get('output')
    norm_ranges = record.get('protected_ranges') or []
    if norm_out is None or len(norm_ranges) != len(spans):
        # Record lacks normalized offsets; only map when each span's spoken text occurs uniquely.
        ranges = []
        mapped = []
        cursor = 0
        for idx, sp in enumerate(spans):
            spoken = sp['spoken'].lower() if lang == 'vi' else sp['spoken']
            occurrences = text.count(spoken)
            if occurrences == 1:
                a = text.find(spoken, cursor)
                if a >= 0:
                    b = a + len(spoken)
                    cursor = b
                    ranges.append((a, b))
                    mapped.append({'span_index': idx, 'raw': sp.get('raw', ''), 'tts_range': [a, b], 'status': 'mapped'})
                    continue
            reason = 'ambiguous duplicate spoken text without span offsets' if occurrences > 1 else 'unmapped span without normalized offsets'
            record.setdefault('warnings', []).append({
                'raw': sp.get('raw', ''),
                'reason': reason,
                'policy': 'do not guess occurrence via str.find; record unmapped span',
            })
            mapped.append({'span_index': idx, 'raw': sp.get('raw', ''), 'tts_range': None, 'status': 'ambiguous' if occurrences > 1 else 'unmapped'})
        return ranges, mapped

    if text == norm_out or (lang == 'vi' and len(text) == len(norm_out) and text == norm_out.lower()):
        ranges = [(int(a), int(b)) for a, b in norm_ranges]
        mapped = [
            {'span_index': idx, 'raw': sp.get('raw', ''), 'norm_range': [int(a), int(b)], 'tts_range': [int(a), int(b)], 'status': 'mapped'}
            for idx, (sp, (a, b)) in enumerate(zip(spans, ranges))
        ]
        return ranges, mapped

    opcodes = difflib.SequenceMatcher(None, norm_out, text, autojunk=False).get_opcodes()
    ranges = []
    mapped = []
    for idx, (sp, (na, nb)) in enumerate(zip(spans, norm_ranges)):
        na, nb = int(na), int(nb)
        ta = tb = None
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal' and i1 <= na and nb <= i2:
                ta = j1 + (na - i1)
                tb = j1 + (nb - i1)
                break
        if ta is None:
            # Check if span endpoints lie cleanly at opcode boundaries (e.g., whole span replaced without crossing plain gaps).
            start_mapped = end_mapped = None
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'equal' and i1 <= na <= i2:
                    start_mapped = j1 + (na - i1)
                elif na == i1 and tag != 'equal':
                    start_mapped = j1
                if tag == 'equal' and i1 <= nb <= i2:
                    end_mapped = j1 + (nb - i1)
                elif nb == i2 and tag != 'equal':
                    end_mapped = j2
            if start_mapped is not None and end_mapped is not None and start_mapped <= end_mapped:
                expected_spoken = sp['spoken'].lower() if lang == 'vi' else sp['spoken']
                candidate_slice = text[start_mapped:end_mapped]
                if candidate_slice in (sp['spoken'], expected_spoken):
                    ta, tb = start_mapped, end_mapped
        if ta is not None and tb is not None and tb > ta:
            ranges.append((ta, tb))
            mapped.append({'span_index': idx, 'raw': sp.get('raw', ''), 'norm_range': [na, nb], 'tts_range': [ta, tb], 'status': 'mapped'})
        else:
            record.setdefault('warnings', []).append({
                'raw': sp.get('raw', ''),
                'reason': 'ambiguous or unmapped span after G2P/text transform',
                'policy': 'do not guess occurrence via str.find; record unmapped span',
            })
            mapped.append({'span_index': idx, 'raw': sp.get('raw', ''), 'norm_range': [na, nb], 'tts_range': None, 'status': 'unmapped'})
    return ranges, mapped


def _plan_chunks(text, lang, max_units, unit, record=None, counter=None, *, _record_planned=True, span_ranges=None):
    count = counter or ((lambda s: len(s.split())) if unit == 'word' else len)
    if span_ranges is not None:
        raw_ranges = list(span_ranges)
        mapped = None
    else:
        raw_ranges, mapped = _map_protected_ranges(record, text, lang)
    ranges = [(a, b) for a, b in raw_ranges if 0 <= a < b <= len(text) and count(text[a:b]) <= max_units]

    boundaries = {0, len(text)}
    for i in range(1, len(text)):
        left, right = text[i-1], text[i]
        if unicodedata.combining(right): continue
        if left.isspace() or right.isspace() or left in '.!?。！？;；,，':
            boundaries.add(i)
        elif lang == 'zh' and not (left.isascii() and right.isascii() and left.isalnum() and right.isalnum()):
            boundaries.add(i)
    boundaries = sorted(i for i in boundaries if not any(a < i < b for a, b in ranges))
    planned = []
    start = 0
    while start < len(text):
        options = [b for b in boundaries if b > start and count(text[start:b]) <= max_units]
        if not options:
            raise ValueError('An indivisible TTS token exceeds the graph input budget')
        end = max(options)
        sentence = [b for b in options if b > start and text[start:b].rstrip().endswith(('.', '!', '?', '。', '！', '？', ';', '；'))]
        if end < len(text) and sentence: end = max(sentence)
        raw_slice = text[start:end]
        lstrip_len = len(raw_slice) - len(raw_slice.lstrip())
        piece = raw_slice.strip()
        if piece:
            pa = start + lstrip_len
            pb = pa + len(piece)
            planned.append((piece, pa, pb))
        start = end
    if record is not None and _record_planned:
        record['planned_chunks'] = [p for p, _, _ in planned]
        record['chunk_policy'] = 'never cut Latin/Hangul tokens; keep typed spans if graph budget permits'
        if mapped is not None:
            record['span_mapping'] = mapped
    return planned, ranges


def chunks(text, lang, max_units, unit, record=None, counter=None, *, span_ranges=None):
    """Preserve tokens; prefer sentence breaks and keep typed spans when they fit.

    Oversized numeric spans can exceed graph limits and are split at word/CJK
    character boundaries. These boundaries are reported, never silently truncated.
    English words are never split; an oversized word is an explicit error.
    """
    planned, _ = _plan_chunks(text, lang, max_units, unit, record=record, counter=counter, _record_planned=True, span_ranges=span_ranges)
    return [p for p, _, _ in planned]


def _adaptive_run(text, lang, max_units, unit, synth, record, depth, counter, span_ranges, preprocessed_counter):
    count = counter or ((lambda s: len(s.split())) if unit == 'word' else len)
    planned, active_ranges = _plan_chunks(
        text, lang, max_units, unit,
        record=record, counter=counter,
        _record_planned=(depth == 0),
        span_ranges=span_ranges,
    )
    out = []
    for piece, pa, pb in planned:
        wav, capped = synth(piece)
        norm_len = len(piece)
        utf8_bytes = len(unicodedata.normalize('NFC', piece).encode('utf-8'))
        word_count = len(piece.split())
        prep_len = preprocessed_counter(piece) if preprocessed_counter is not None else count(piece)
        entry = {
            'text': piece,
            'normalized_length': norm_len,
            'preprocessed_length': prep_len,
            'utf8_bytes': utf8_bytes,
            'word_count': word_count,
            'max_units': max_units,
            'depth': depth,
            'capped': bool(capped),
            'retried': bool(capped),
        }
        if record is not None:
            record.setdefault('attempted_chunks', []).append(dict(entry))
        if capped:
            smaller = max_units // 2
            if depth >= 8 or smaller < 1:
                raise RuntimeError('TTS graph audio cap reached for an indivisible token')
            if record is not None:
                record.setdefault('audio_cap_retries', []).append(piece)
            sub_ranges = [(sa - pa, sb - pa) for sa, sb in active_ranges if pa <= sa and sb <= pb]
            out += _adaptive_run(piece, lang, smaller, unit, synth, record, depth + 1, counter, sub_ranges, preprocessed_counter)
        else:
            if record is not None:
                record.setdefault('synthesized_chunks', []).append(dict(entry))
            out.append(wav)
    return out


def adaptive(text, lang, max_units, unit, synth, record=None, depth=0, counter=None, *, span_ranges=None, preprocessed_counter=None):
    if record is not None and depth == 0:
        record['attempted_chunks'] = []
        record['synthesized_chunks'] = []
        record.setdefault('audio_cap_retries', [])
    return _adaptive_run(text, lang, max_units, unit, synth, record, depth, counter, span_ranges, preprocessed_counter)
