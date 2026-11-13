# -*- coding: utf-8 -*-
"""Pure-Python semantic numeric guard and target entity auditor (Python >=3.10, zero Metal/PyTorch)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Any, Sequence

from .hybrid_textnorm import HybridNormalizer

_PURE_HYBRID: HybridNormalizer | None = None

_TEMP_DEG_RE = re.compile(
    r"^\s*(?:degrees?\s*(celsius|fahrenheit)|độ\s*([cf])(?![A-Za-zÀ-ỹ])|°\s*([cf])(?![A-Za-z]))",
    re.IGNORECASE,
)
_AMPM_TAIL_RE = re.compile(r"^\s*([aApP]\.?[mM]\.?)(?![A-Za-z])")

_VI_SPOKEN_NUM_WORDS = frozenset({
    "không", "một", "mốt", "hai", "ba", "tư", "bốn", "năm", "lăm", "nhăm",
    "sáu", "bảy", "bẩy", "tám", "chín", "mười", "mươi", "linh", "lẻ",
    "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "phẩy", "âm", "rưỡi",
})
_VI_SPOKEN_CUES = frozenset({"số", "mã", "lúc", "giờ", "ngày", "tháng", "năm", "nhiệt", "giảm", "tăng", "giá"})


@dataclass(frozen=True)
class NumericSpan:
    start: int
    end: int
    kind: str
    raw: str
    signature: tuple | None


@dataclass
class GuardReport:
    valid: bool
    issues: list[dict[str, Any]]
    unknowns: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    status: str  # "verified" | "invalid" | "unknown" | "not_applicable"
    fallback_policy: str | None = None
    token_identity: bool | None = None
    source_spans: list[dict[str, Any]] = field(default_factory=list)
    target_spans: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "issues": list(self.issues),
            "unknowns": list(self.unknowns),
            "warnings": list(self.warnings),
            "fallback_policy": self.fallback_policy,
            "token_identity": self.token_identity,
            "source_spans": list(self.source_spans),
            "target_spans": list(self.target_spans),
        }


def _get_hybrid() -> HybridNormalizer:
    global _PURE_HYBRID
    if _PURE_HYBRID is None:
        _PURE_HYBRID = HybridNormalizer(learned_identifiers=False)
    return _PURE_HYBRID


def _canonical_unit(unit_raw: str) -> str:
    u = re.sub(r"\s+", " ", unit_raw.strip().casefold())
    u_compact = u.replace(" ", "")
    if u_compact in ("°c", "℃", "c", "celsius", "degreescelsius", "degreecelsius", "độc", "摄氏度"):
        return "degC"
    if u_compact in ("°f", "f", "fahrenheit", "degreesfahrenheit", "degreefahrenheit", "độf", "华氏度"):
        return "degF"
    if u_compact in ("°", "độ", "degrees", "degree", "도", "度"):
        return "deg"
    return u_compact


def _canonical_id(raw: str) -> str:
    return "".join(c for c in raw if not c.isspace()).upper()


def _canonical_whole(whole: str) -> str:
    # Preserve exact zero if whole is all zeros; otherwise strip redundant leading zeros on parsed numbers
    if not whole:
        return "0"
    stripped = whole.lstrip("0")
    return stripped or "0"


def _span_signature(kind: str, raw: str, value: dict[str, Any], lang: str) -> tuple | None:
    if kind in ("unsupported", "unresolved"):
        return None
    if kind == "date":
        return ("date", value.get("year"), value.get("month"), value.get("day"))
    if kind == "year":
        return ("year", value.get("year"), lang)
    if kind == "time":
        return (
            "time",
            value.get("hour"),
            value.get("minute"),
            value.get("second"),
            value.get("ampm") or "",
            lang,
        )
    if kind == "money":
        num = value.get("number") or {}
        return (
            "money",
            value.get("currency"),
            num.get("sign") or "",
            _canonical_whole(str(num.get("whole", "0"))),
            str(num.get("fraction", "")),
            lang,
        )
    if kind == "percent":
        num = value.get("number") or {}
        return (
            "percent",
            num.get("sign") or "",
            _canonical_whole(str(num.get("whole", "0"))),
            str(num.get("fraction", "")),
            lang,
        )
    if kind == "measure":
        num = value.get("number") or {}
        unit = _canonical_unit(str(value.get("unit", "")))
        return (
            "measure",
            unit,
            num.get("sign") or "",
            _canonical_whole(str(num.get("whole", "0"))),
            str(num.get("fraction", "")),
            lang,
        )
    if kind == "identifier":
        chars = str(value.get("characters") or _canonical_id(raw))
        return ("identifier", chars.upper())
    if kind == "number":
        return (
            "number",
            value.get("sign") or "",
            _canonical_whole(str(value.get("whole", "0"))),
            str(value.get("fraction", "")),
            lang,
        )
    return None


def _run_formatter_spans(text: str, lang: str, *, include_crf: bool = False) -> list[dict[str, Any]]:
    """Adapter calling formatter's HybridNormalizer / extract_spans without duplicating regexes."""
    if not text or lang not in ("vi", "en", "ko", "zh"):
        return []
    h = HybridNormalizer(learned_identifiers=True) if include_crf else _get_hybrid()
    if hasattr(h, "extract_spans"):
        return list(h.extract_spans(text, lang))
    _, record = h(text, lang)
    return list(record.get("spans", []))


def extract_known_numeric_spans(text: str, lang: str, *, include_crf: bool = False) -> list[NumericSpan]:
    """Extract typed NumericSpans using formatter's lexical boundary & type policy (<5ms CPU)."""
    if not text or lang not in ("vi", "en", "ko", "zh"):
        return []

    raw_spans = _run_formatter_spans(text, lang, include_crf=include_crf)
    out: list[NumericSpan] = []

    for s in raw_spans:
        a = int(s["start"])
        b = int(s["end"])
        kind = str(s["kind"])
        raw = str(s["raw"])
        value = dict(s.get("value") or {})

        # Typed context promotion for ASR unpunctuated 12h time with explicit meridiem (e.g. "330 pm" -> 3:30 PM)
        if kind == "number" and not value.get("sign") and not value.get("fraction"):
            whole = str(value.get("whole", ""))
            if whole.isdecimal() and len(whole) in (3, 4) and not whole.startswith("0"):
                m_ampm = _AMPM_TAIL_RE.match(text[b:])
                if m_ampm:
                    hour = int(whole[:-2])
                    minute = int(whole[-2:])
                    if 1 <= hour <= 12 and 0 <= minute < 60:
                        ampm = m_ampm.group(1).replace(".", "").upper()
                        new_b = b + m_ampm.end()
                        new_raw = text[a:new_b]
                        sig = ("time", hour, minute, None, ampm, lang)
                        out.append(NumericSpan(a, new_b, "time", new_raw, sig))
                        continue

        # Attach spelled temperature units immediately following a number (e.g. "37.5 degrees Celsius", "37.5 độ C")
        if kind == "number":
            m_temp = _TEMP_DEG_RE.match(text[b:])
            if m_temp:
                unit_str = (m_temp.group(1) or m_temp.group(2) or m_temp.group(3) or "").strip()
                c_unit = _canonical_unit(unit_str if unit_str.lower() in ("celsius", "fahrenheit") else f"°{unit_str}")
                new_b = b + m_temp.end()
                new_raw = text[a:new_b]
                sig = (
                    "measure",
                    c_unit,
                    value.get("sign") or "",
                    _canonical_whole(str(value.get("whole", "0"))),
                    str(value.get("fraction", "")),
                    lang,
                )
                out.append(NumericSpan(a, new_b, "measure", new_raw, sig))
                continue

        sig = _span_signature(kind, raw, value, lang)
        out.append(NumericSpan(a, b, kind, raw, sig))

    out.sort(key=lambda sp: sp.start)
    return out


def _detect_unknown_candidates(text: str, spans: Sequence[NumericSpan], lang: str) -> list[dict[str, Any]]:
    unknowns: list[dict[str, Any]] = []

    for sp in spans:
        if sp.signature is None:
            unknowns.append({
                "kind": sp.kind,
                "raw": sp.raw,
                "start": sp.start,
                "end": sp.end,
                "reason": "opaque_or_ambiguous_span",
            })

    # Uncovered written digits not inside any extracted span
    for m in re.finditer(r"\d+", text):
        ma, mb = m.span()
        if not any(sp.start <= ma and mb <= sp.end for sp in spans):
            unknowns.append({
                "kind": "uncovered_digits",
                "raw": m.group(0),
                "start": ma,
                "end": mb,
                "reason": "unparsed_numeric_token",
            })

    # Spoken Vietnamese numeral sequences (exclude single negation "không" or lone weekday/year words)
    if lang == "vi":
        tokens = list(re.finditer(r"[^\W\d_]+", text, re.UNICODE))
        words_low = [t.group(0).casefold() for t in tokens]
        i = 0
        while i < len(tokens):
            w = words_low[i]
            if w in _VI_SPOKEN_NUM_WORDS:
                j = i + 1
                while j < len(tokens) and words_low[j] in _VI_SPOKEN_NUM_WORDS:
                    j += 1
                run_len = j - i
                prev_word = words_low[i - 1] if i > 0 else ""
                is_cued_zero = (run_len == 1 and w == "không" and prev_word == "số")
                if run_len >= 2 or is_cued_zero:
                    sa = tokens[i].start()
                    sb = tokens[j - 1].end()
                    if not any(sp.start <= sa and sb <= sp.end for sp in spans):
                        unknowns.append({
                            "kind": "spoken_numeral",
                            "raw": text[sa:sb],
                            "start": sa,
                            "end": sb,
                            "reason": "unparsed_spoken_numeral",
                        })
                i = j
            else:
                i += 1

    return unknowns


def _strip_outside_punct_and_space(text: str, spans: Sequence[NumericSpan]) -> str:
    """Return casefolded non-span text with outside whitespace and punctuation removed for token-identity checks."""
    out: list[str] = []
    cursor = 0
    for sp in spans:
        if sp.start > cursor:
            out.append(text[cursor:sp.start])
        out.append("\u0000")
        cursor = sp.end
    if cursor < len(text):
        out.append(text[cursor:])
    joined = "".join(out)
    cleaned: list[str] = []
    for ch in joined:
        if ch == "\u0000":
            cleaned.append(ch)
        elif ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        else:
            cleaned.append(ch.casefold())
    return "".join(cleaned)


def _opaque_equivalent(raw_before: str, raw_after: str) -> bool:
    """Opaque spans (signature=None) require exact characters (only optional case/whitespace normalization if non-punct)."""
    if raw_before == raw_after:
        return True
    # Only allow pure case difference if digits/punctuation/order are 100% identical
    return raw_before.casefold() == raw_after.casefold()


def _date_order_compatible(before_raw: str, after_raw: str) -> bool:
    # Guard against numeric slash date flip like 21/12/2027 -> 12/21/2027 in source prep
    mb = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s*", before_raw)
    ma = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s*", after_raw)
    if mb and ma:
        return mb.groups() == ma.groups()
    return True


def _signatures_match(before_sp: NumericSpan, after_sp: NumericSpan) -> tuple[bool, str | None]:
    if before_sp.signature is None or after_sp.signature is None:
        if before_sp.signature is not None or after_sp.signature is not None:
            return False, "opaque_vs_typed_mismatch"
        if not _opaque_equivalent(before_sp.raw, after_sp.raw):
            return False, "opaque_span_mutated"
        return True, None

    if before_sp.kind != after_sp.kind:
        return False, f"kind_changed:{before_sp.kind}->{after_sp.kind}"
    if before_sp.signature != after_sp.signature:
        return False, "numeric_signature_mismatch"
    if before_sp.kind == "date" and not _date_order_compatible(before_sp.raw, after_sp.raw):
        return False, "date_order_changed"
    return True, None


def guard_source_numeric_changes(
    before: str,
    after: str,
    lang: str,
    *,
    allow_outside_punct_and_case: bool = True,
    include_crf: bool = False,
) -> GuardReport:
    """Validate that `after` preserves all numeric/identifier spans and non-span token identity of `before`."""
    before_str = before or ""
    after_str = after or ""
    src_spans = extract_known_numeric_spans(before_str, lang, include_crf=include_crf)
    dst_spans = extract_known_numeric_spans(after_str, lang, include_crf=include_crf)

    src_unknowns = _detect_unknown_candidates(before_str, src_spans, lang)
    dst_unknowns = _detect_unknown_candidates(after_str, dst_spans, lang)

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if len(src_spans) != len(dst_spans):
        issues.append({
            "code": "span_count_mismatch",
            "before_count": len(src_spans),
            "after_count": len(dst_spans),
            "before_spans": [sp.raw for sp in src_spans],
            "after_spans": [sp.raw for sp in dst_spans],
        })
    else:
        for idx, (bsp, asp) in enumerate(zip(src_spans, dst_spans)):
            ok, reason = _signatures_match(bsp, asp)
            if not ok:
                issues.append({
                    "code": reason or "span_mismatch",
                    "index": idx,
                    "before": bsp.raw,
                    "after": asp.raw,
                    "before_kind": bsp.kind,
                    "after_kind": asp.kind,
                })

    # Check non-span text identity (catches dropped suffix chunks, added/deleted words, CAPU word->digit edits)
    if allow_outside_punct_and_case:
        outside_before = _strip_outside_punct_and_space(before_str, src_spans)
        outside_after = _strip_outside_punct_and_space(after_str, dst_spans)
        token_identity = (outside_before == outside_after) and (len(issues) == 0)
        if outside_before != outside_after:
            issues.append({
                "code": "token_identity_mismatch",
                "message": "Non-span text or token order changed beyond allowed casing/punctuation/whitespace",
            })
    else:
        token_identity = (before_str == after_str) and (len(issues) == 0)
        if before_str != after_str:
            issues.append({
                "code": "strict_text_mismatch",
                "message": "Text modified under strict equality policy",
            })

    # Merge unknowns (deduplicated by kind+raw)
    seen_unk: set[tuple[str, str]] = set()
    unknowns: list[dict[str, Any]] = []
    for u in src_unknowns + dst_unknowns:
        key = (str(u.get("kind", "")), str(u.get("raw", "")))
        if key not in seen_unk:
            seen_unk.add(key)
            unknowns.append(u)

    if unknowns:
        warnings.append({
            "code": "unknown_numeric_candidates",
            "count": len(unknowns),
            "items": [u["raw"] for u in unknowns],
        })

    if issues:
        status = "invalid"
        valid = False
        fallback_policy = "fallback_to_raw_source"
    elif unknowns:
        status = "unknown"
        valid = False
        fallback_policy = "preserve_source_unknown_unverified"
    elif not src_spans:
        status = "not_applicable"
        valid = True
        fallback_policy = None
    else:
        status = "verified"
        valid = True
        fallback_policy = None

    return GuardReport(
        valid=valid,
        issues=issues,
        unknowns=unknowns,
        warnings=warnings,
        status=status,
        fallback_policy=fallback_policy,
        token_identity=token_identity,
        source_spans=[asdict(s) for s in src_spans],
        target_spans=[asdict(s) for s in dst_spans],
    )


def _cross_lingual_comparable(sp: NumericSpan) -> tuple | None:
    """Return locale-independent semantic key for target-side auditing (None if opaque/unverifiable)."""
    if sp.signature is None:
        return None
    kind = sp.kind
    sig = sp.signature
    if kind == "date":
        return ("date", sig[1], sig[2], sig[3])
    if kind == "year":
        return ("year", sig[1])
    if kind == "time":
        return ("time", sig[1], sig[2], sig[3], sig[4])
    if kind == "money":
        return ("money", sig[1], sig[2], sig[3], sig[4])
    if kind == "percent":
        return ("percent", sig[1], sig[2], sig[3])
    if kind == "measure":
        return ("measure", sig[1], sig[2], sig[3], sig[4])
    if kind == "identifier":
        return ("identifier", sig[1])
    if kind == "number":
        return ("number", sig[1], sig[2], sig[3])
    return None


def _numeric_decimal_value(sig: tuple) -> Decimal | None:
    kind = sig[0]
    try:
        if kind == "number":
            sign, whole, frac = sig[1], sig[2], sig[3]
        elif kind == "percent":
            sign, whole, frac = sig[1], sig[2], sig[3]
        elif kind in ("money", "measure"):
            sign, whole, frac = sig[2], sig[3], sig[4]
        else:
            return None
        s = f"{'-' if sign == '-' else ''}{whole}" + (f".{frac}" if frac else "")
        return Decimal(s)
    except (InvalidOperation, IndexError):
        return None


def _target_time_meridiem(target_text: str, sp: NumericSpan) -> str:
    """Detect explicit localized meridiem immediately adjacent to a target time span."""
    if sp.signature and len(sp.signature) >= 5 and sp.signature[4]:
        return str(sp.signature[4])
    prefix = target_text[max(0, sp.start - 12):sp.start].casefold()
    suffix = target_text[sp.end:min(len(target_text), sp.end + 16)].casefold()
    if re.search(r"\b(?:sáng|in the morning)\b", suffix) or re.search(r"(?:오전|上午|早上)\s*$", prefix):
        return "AM"
    if re.search(r"\b(?:chiều|tối|trưa|đêm|in the afternoon|in the evening|at night)\b", suffix) or re.search(r"(?:오후|下午|晚上)\s*$", prefix):
        return "PM"
    return ""


def _values_equivalent(src_sp: NumericSpan, tgt_sp: NumericSpan, target_text: str = "") -> bool:
    cs = _cross_lingual_comparable(src_sp)
    ct = _cross_lingual_comparable(tgt_sp)
    if cs is None or ct is None:
        return False
    if cs[0] == "time" and ct[0] == "time":
        tgt_ampm = ct[4] or (_target_time_meridiem(target_text, tgt_sp) if target_text else "")
        return (cs[1], cs[2], cs[3], cs[4]) == (ct[1], ct[2], ct[3], tgt_ampm)
    if cs == ct:
        return True
    # Allow Korean "도" (parsed as generic 'deg') to match 'degC' when target lang is ko
    if cs[0] == "measure" and ct[0] == "measure":
        if cs[1] in ("degC", "deg") and ct[1] in ("degC", "deg"):
            if cs[2] == ct[2] and _numeric_decimal_value(cs) == _numeric_decimal_value(ct):
                return True
    # Allow trailing-zero equivalence on decimal numbers across MT when type & unit/currency match (e.g. $15.50 -> 15,5 đô la)
    if cs[0] == ct[0] and cs[0] in ("money", "percent", "measure", "number"):
        if cs[0] in ("money", "measure") and cs[1] != ct[1]:
            return False
        ds = _numeric_decimal_value(cs)
        dt = _numeric_decimal_value(ct)
        if ds is not None and dt is not None and ds == dt:
            return True
    return False


def _has_unparsed_spoken_numeral(text: str, lang: str) -> bool:
    low = text.casefold()
    if lang == "en":
        words = set(re.findall(r"[a-z]+", low))
        return bool(words & {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
        })
    if lang == "vi":
        words = set(re.findall(r"[^\W\d_]+", low, re.UNICODE))
        return bool(words & (_VI_SPOKEN_NUM_WORDS - {"không", "năm", "ba", "tư"})) or len(words & _VI_SPOKEN_NUM_WORDS) >= 2
    if lang == "zh":
        return any(c in "零一二两三四五六七八九十百千万亿" for c in text)
    if lang == "ko":
        return any(c in "영공일이삼사오육칠팔구십백천만억조하나둘셋넷다섯여섯일곱여덟아홉열" for c in text)
    return False


def audit_target_entities(
    source_entities: Sequence[NumericSpan] | str,
    target_text: str,
    tgt_lang: str,
    *,
    src_lang: str = "en",
) -> dict[str, Any]:
    """Audit whether known source entities are preserved in MT `target_text` without mutating `target_text`.

    Returns dict with keys: preserved, mismatch, unverifiable, items, counts, target_spans.
    """
    if isinstance(source_entities, str):
        src_spans = extract_known_numeric_spans(source_entities, src_lang)
    else:
        src_spans = list(source_entities or ())

    target_str = target_text or ""
    tgt_spans = extract_known_numeric_spans(target_str, tgt_lang)
    consumed: set[int] = set()
    spoken_present = _has_unparsed_spoken_numeral(target_str, tgt_lang)

    preserved: list[dict[str, Any]] = []
    mismatch: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    for idx, src_sp in enumerate(src_spans):
        # Opaque/unknown source entities: check exact substring preservation if unambiguous, else unverifiable
        if src_sp.signature is None:
            pos = target_str.find(src_sp.raw)
            if pos >= 0:
                entry = {
                    "index": idx,
                    "kind": src_sp.kind,
                    "source_raw": src_sp.raw,
                    "target_raw": src_sp.raw,
                    "status": "unverifiable",
                    "reason": "opaque_source_span_verbatim",
                }
            else:
                entry = {
                    "index": idx,
                    "kind": src_sp.kind,
                    "source_raw": src_sp.raw,
                    "target_raw": None,
                    "status": "unverifiable",
                    "reason": "opaque_source_span_unverifiable",
                }
            unverifiable.append(entry)
            items.append(entry)
            continue

        matched_idx = None
        for t_idx, tgt_sp in enumerate(tgt_spans):
            if t_idx in consumed:
                continue
            if _values_equivalent(src_sp, tgt_sp, target_str):
                matched_idx = t_idx
                break

        if matched_idx is not None:
            consumed.add(matched_idx)
            tgt_sp = tgt_spans[matched_idx]
            entry = {
                "index": idx,
                "kind": src_sp.kind,
                "source_raw": src_sp.raw,
                "target_raw": tgt_sp.raw,
                "status": "preserved",
                "reason": "typed_signature_match",
            }
            preserved.append(entry)
            items.append(entry)
            continue

        # No exact match: check if target has unconsumed typed span or uncovered digits -> mismatch,
        # OR target spelled numbers in words without parseable digits -> unverifiable!
        consumed_ranges = [(tgt_spans[t_idx].start, tgt_spans[t_idx].end) for t_idx in consumed]
        has_unconsumed_target_digits = any(t_idx not in consumed for t_idx in range(len(tgt_spans))) or any(
            not any(sa <= m.start() < sb for sa, sb in consumed_ranges)
            for m in re.finditer(r"\d", target_str)
        )
        if not has_unconsumed_target_digits and spoken_present:
            entry = {
                "index": idx,
                "kind": src_sp.kind,
                "source_raw": src_sp.raw,
                "target_raw": None,
                "status": "unverifiable",
                "reason": "target_spoken_numeral_words",
            }
            unverifiable.append(entry)
            items.append(entry)
        else:
            # Find closest candidate in remaining target spans for diagnostics (without consuming)
            candidate_raw = next((tgt_spans[t_idx].raw for t_idx in range(len(tgt_spans)) if t_idx not in consumed), None)
            entry = {
                "index": idx,
                "kind": src_sp.kind,
                "source_raw": src_sp.raw,
                "target_raw": candidate_raw,
                "status": "mismatch",
                "reason": "missing_or_mismatched_target_entity",
            }
            mismatch.append(entry)
            items.append(entry)

    return {
        "preserved": preserved,
        "mismatch": mismatch,
        "unverifiable": unverifiable,
        "items": items,
        "counts": {
            "total": len(src_spans),
            "preserved": len(preserved),
            "mismatch": len(mismatch),
            "unverifiable": len(unverifiable),
        },
        "target_spans": [asdict(s) for s in tgt_spans],
    }
