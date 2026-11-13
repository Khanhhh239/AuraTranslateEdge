# -*- coding: utf-8 -*-
"""Span-safe source segmentation for MT (pure Python >=3.10, QCS8550 NPU S=64 compatible)."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Callable, Iterable, Sequence

# Sentence boundaries: require whitespace after ASCII [.!?;] (so $18.25 is never split at '.')
# and allow immediate split after CJK sentence-final punctuation or newlines.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；\n])\s*")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,，、:：;；—])\s+|(?<=[、，；])\s*|\s+(?=[—–-]\s)")


class SpanBudgetError(ValueError):
    """Raised when a single protected span or unsplittable unit exceeds token_limit."""


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int


def _normalize_spans(text_len: int, protected_spans: Iterable) -> list[_Interval]:
    raw_intervals: list[tuple[int, int]] = []
    for item in protected_spans or ():
        if hasattr(item, "start") and hasattr(item, "end"):
            a, b = int(item.start), int(item.end)
        elif isinstance(item, dict):
            a, b = int(item["start"]), int(item["end"])
        else:
            a, b = int(item[0]), int(item[1])
        a = max(0, min(text_len, a))
        b = max(0, min(text_len, b))
        if b > a:
            raw_intervals.append((a, b))
    if not raw_intervals:
        return []
    raw_intervals.sort(key=lambda pair: (pair[0], -(pair[1] - pair[0])))
    merged: list[_Interval] = []
    cur_a, cur_b = raw_intervals[0]
    for a, b in raw_intervals[1:]:
        if a < cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append(_Interval(cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append(_Interval(cur_a, cur_b))
    return merged


def _inside_any(pos: int, intervals: Sequence[_Interval]) -> bool:
    for iv in intervals:
        if iv.start < pos < iv.end:
            return True
    return False


def _overlaps_boundary(start: int, end: int, intervals: Sequence[_Interval]) -> bool:
    for iv in intervals:
        if iv.start < start < iv.end or iv.start < end < iv.end:
            return True
    return False


def _split_ranges_by_regex(
    text: str,
    start: int,
    end: int,
    pattern: re.Pattern[str],
    intervals: Sequence[_Interval],
    *,
    is_sentence: bool = False,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cursor = start
    sub = text[start:end]
    for m in pattern.finditer(sub):
        split_at = start + m.start()
        next_at = start + m.end()
        if _inside_any(split_at, intervals) or _inside_any(next_at, intervals) or _overlaps_boundary(split_at, next_at, intervals):
            continue
        if is_sentence and split_at > start:
            # Do not treat lowercase continuation after "a.m." / "p.m." as a sentence break.
            left_tail = text[max(start, split_at - 6):split_at].lower()
            if (left_tail.endswith("a.m.") or left_tail.endswith("p.m.")) and next_at < end and text[next_at:next_at + 1].islower():
                continue
        piece_start = cursor
        piece_end = split_at
        while piece_start < piece_end and text[piece_start].isspace():
            piece_start += 1
        while piece_end > piece_start and text[piece_end - 1].isspace():
            piece_end -= 1
        if piece_end > piece_start:
            out.append((piece_start, piece_end))
        cursor = next_at
    while cursor < end and text[cursor].isspace():
        cursor += 1
    tail_end = end
    while tail_end > cursor and text[tail_end - 1].isspace():
        tail_end -= 1
    if tail_end > cursor:
        out.append((cursor, tail_end))
    return out


def _atomic_units(text: str, start: int, end: int, intervals: Sequence[_Interval], lang: str | None) -> list[tuple[int, int]]:
    """Split [start, end) into atomic character-span units without ever cutting inside a protected interval."""
    units: list[tuple[int, int]] = []
    pos = start
    local_ivs = [iv for iv in intervals if iv.end > start and iv.start < end]
    iv_idx = 0
    while pos < end:
        if iv_idx < len(local_ivs) and pos >= local_ivs[iv_idx].start:
            iv = local_ivs[iv_idx]
            iv_idx += 1
            a = max(start, iv.start)
            b = min(end, iv.end)
            if b > a:
                units.append((a, b))
            pos = max(pos, b)
            continue
        next_iv_start = local_ivs[iv_idx].start if iv_idx < len(local_ivs) else end
        seg_end = min(end, next_iv_start)
        seg = text[pos:seg_end]
        if lang in ("zh", "ko") and " " not in seg.strip():
            for offset, ch in enumerate(seg):
                if not ch.isspace():
                    units.append((pos + offset, pos + offset + 1))
        else:
            for m in re.finditer(r"\S+", seg):
                units.append((pos + m.start(), pos + m.end()))
        pos = seg_end
    return units


def _pack_ranges(
    text: str,
    ranges: Sequence[tuple[int, int]],
    *,
    token_limit: int,
    encode: Callable[[str], Sequence[int]],
) -> list[tuple[int, int]]:
    packed: list[tuple[int, int]] = []
    cur_a: int | None = None
    cur_b: int | None = None
    for a, b in ranges:
        piece = text[a:b].strip()
        if not piece:
            continue
        if cur_a is None:
            cur_a, cur_b = a, b
            continue
        candidate = text[cur_a:b].strip()
        if len(encode(candidate)) <= token_limit:
            cur_b = b
        else:
            packed.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    if cur_a is not None and cur_b is not None:
        packed.append((cur_a, cur_b))
    return packed


def _is_small_fragment(piece: str, lang: str | None) -> bool:
    if lang in ("zh", "ko"):
        return len(re.sub(r"\W", "", piece)) < 3
    return len(piece.split()) < 2


def split_preserving_spans(
    text: str,
    protected_spans: Iterable,
    *,
    token_limit: int,
    encode: Callable[[str], Sequence[int]],
    lang: str | None = None,
    merge_small: bool = False,
) -> list[str]:
    """Split `text` into in-order chunks respecting sentence boundaries and `token_limit`, never splitting `protected_spans`."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if token_limit < 1:
        raise ValueError(f"token_limit must be >= 1, got {token_limit}")
    if not text.strip():
        return []

    intervals = _normalize_spans(len(text), protected_spans)

    # Validate each individual protected span fits within token_limit before splitting.
    for iv in intervals:
        span_text = text[iv.start:iv.end].strip()
        if span_text:
            n_toks = len(encode(span_text))
            if n_toks > token_limit:
                raise SpanBudgetError(
                    f"Protected span {span_text!r} requires {n_toks} tokens, exceeding budget token_limit={token_limit}"
                )

    sent_ranges = _split_ranges_by_regex(text, 0, len(text), _SENT_SPLIT_RE, intervals, is_sentence=True)
    if not sent_ranges:
        return []

    if merge_small and len(sent_ranges) > 1:
        merged_sents: list[tuple[int, int]] = []
        for a, b in sent_ranges:
            piece = text[a:b].strip()
            if merged_sents and (_is_small_fragment(piece, lang) or _is_small_fragment(text[merged_sents[-1][0]:merged_sents[-1][1]].strip(), lang)):
                prev_a, _ = merged_sents[-1]
                candidate = text[prev_a:b].strip()
                if len(encode(candidate)) <= token_limit:
                    merged_sents[-1] = (prev_a, b)
                    continue
            merged_sents.append((a, b))
        sent_ranges = merged_sents

    chunks: list[str] = []
    for sa, sb in sent_ranges:
        sent_text = text[sa:sb].strip()
        if not sent_text:
            continue
        if len(encode(sent_text)) <= token_limit:
            chunks.append(sent_text)
            continue

        clause_ranges = _split_ranges_by_regex(text, sa, sb, _CLAUSE_SPLIT_RE, intervals, is_sentence=False)
        if len(clause_ranges) > 1:
            packed_clauses = _pack_ranges(text, clause_ranges, token_limit=token_limit, encode=encode)
        else:
            packed_clauses = [(sa, sb)]

        for ca, cb in packed_clauses:
            clause_text = text[ca:cb].strip()
            if not clause_text:
                continue
            if len(encode(clause_text)) <= token_limit:
                chunks.append(clause_text)
                continue

            units = _atomic_units(text, ca, cb, intervals, lang)
            for ua, ub in units:
                utext = text[ua:ub]
                n_toks = len(encode(utext))
                if n_toks > token_limit:
                    raise SpanBudgetError(
                        f"Unsplittable unit {utext!r} requires {n_toks} tokens, exceeding budget token_limit={token_limit}"
                    )
            packed_units = _pack_ranges(text, units, token_limit=token_limit, encode=encode)
            for pa, pb in packed_units:
                ptext = text[pa:pb].strip()
                if ptext:
                    if len(encode(ptext)) > token_limit:
                        raise SpanBudgetError(
                            f"Chunk {ptext!r} requires {len(encode(ptext))} tokens, exceeding budget token_limit={token_limit}"
                        )
                    chunks.append(ptext)

    return chunks


def load_nllb_encode(assets_dir: str | None = None) -> Callable[[str], list[int]]:
    """Build the SentencePiece encoder matching Nllb.encode (fairseq id+1 offset, <unk> -> 3)."""
    import sentencepiece as spm
    if assets_dir is None:
        from .registry import ASSETS
        assets_dir = ASSETS
    sp = spm.SentencePieceProcessor(model_file=os.path.join(assets_dir, "nllb_spm.model"))
    return lambda text: [3 if p == 0 else p + 1 for p in sp.encode(text)]
