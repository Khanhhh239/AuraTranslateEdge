# Plan 02 — Format, Spans, Locale Normalization & Chunk Trace Execution Report

**Owner:** `formatter` (`docs/workflow/next-steps/02-format/PLAN.md`)  
**Target Environment:** Python >= 3.10 (`wheelhouse_cp310` / AIBOX 8550 `linux/aarch64`, offline `pycrfsuite`-optional fallback), verified locally with `/private/tmp/s2s-original-env/bin/python`.  
**Cross-Agent API Sync:** Shared `identifier_candidates`, `extract_spans`, and `numeric_fields` from `s2s_edge/s2s_edge/hybrid_textnorm.py` with `source_guard` (and incorporated `source_guard` boundary catches #10 and #11 for sentence-final `unsupported` `.` and Vietnamese accented unit lookaheads).

---

## 1. Files Modified / Created (Strict Ownership)

- Modified: [format_locales.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/format_locales.py)
- Modified: [hybrid_textnorm.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/hybrid_textnorm.py)
- Modified: [tts_format.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/tts_format.py)
- Modified: [tts.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/tts.py)
- Created: [test_format_boundaries.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_format_boundaries.py)
- Created: [test_format_locale_boundaries.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_format_locale_boundaries.py)
- Created: [test_tts_chunk_trace.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_tts_chunk_trace.py)
- Created: [EXECUTION_REPORT.md](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/docs/workflow/next-steps/02-format/EXECUTION_REPORT.md)

---

## 2. Implemented Changes (T1 – T4)

### Baseline & Frozen Cases
- Preserved all 55 existing `EXAMPLES` in `tests/test_hybrid_format.py`.
- Froze all 17 multi-language edge cases from `reports/tts_format_s2s_eval_2026-11-13.md` / `tts_format_4langs_report.json` (`vi` x5, `en` x4, `ko` x4, `zh` x4) in `tests/test_format_boundaries.py` (`FROZEN_EDGE_CASES`), verifying all P0 fixes while marking the `PLAN.md`-deferred P1 items (`14h`/`14h45`, `37.5 độ C`, Korean `3시 30분`, VI `ZIP -> dip`).

### T1 — General Identifier Candidates & Context Boundaries
- **`format_locales.py` (`IDENTIFIER_CUES`):** Added multilingual cues (`gọi`, `hotline`, `tel`, `fax`, `code`, `số điện thoại`, `mã zip`, `mã đơn`, `mã số`, `mật khẩu`, `전화`, `휴대폰`, `연락처`, `코드`, `비밀번호`, `주문번호`, `电话`, `手机`, `热线`, `座机`, `代码`, `编号`, `密码`, `航班号`) while deliberately excluding bare `số` so house numbers like `Số nhà 734` stay cardinal.
- **`hybrid_textnorm.py`:**
  - Implemented `identifier_candidates(text: str, lang: str = 'en') -> list[dict]` (`{start, end, raw, reason}`) and `extract_spans(text: str, lang: str, *, date_order=None, learned_identifiers=False) -> list[dict]`.
  - Added general candidate grammar (`_collect_identifier_candidates`, `_has_cue_or_list_context`, `_is_phone_structure`, `_check_paren_and_hyphen`) covering `+country` phones (`+84-908-246-135`, `+1-800-555-0199`, `+82-10-1234-5678`), balanced parenthesized phones (`(202) 555-0123`), space/dash grouping (`0908 246 135`, `0908-246-135`), separator-chained secondary phones after a cue (`Gọi 0908-246-135 hoặc 0912-345-678`, `phone (202) 555-0123, +1-800-555-0199`), CJK adjacent cues without whitespace (`电话0908-246-135`, `邮编04782`, `航班号VN247`), and cue-backed spaced prefix codes (`Flight VN 123`, `Chuyến bay của tôi là VN 123`, `항공편 VN 247`, `航班 VN 247`) with `_CUE_WORDS` exclusion so `ZIP 67297` does not spell `ZIP`.
  - Protected counter-examples (`-12.5%`, `3-5 ngày`, `330 đồng`, `Số nhà 734`, `12:30`, bare numbers without cues) and hardened `_TYPED_PATTERNS` unit/currency lookaheads (`(?![A-Za-z\u00C0-\u024F\u1E00-\u1EFF])`) so Vietnamese words like `gọi` or `lần` after numbers (`123 gọi`, `2 lần`) are not misparsed as `measure` (`g`, `l`).
  - Unbalanced parentheses and malformed/double hyphens (`(202 555-0123`, `202) 555-0123`, `0908--246-135`, `0908-246-`) yield `ambiguous_*` reasons / `unresolved` spans with explicit warnings.
  - CRF fallback (`_learned_spans`): handles `(ImportError, OSError, RuntimeError, ValueError)` cleanly when `pycrfsuite` is absent on `aarch64`.

### T2 — CJK Boundaries, Unsupported Notation & Locale Separators
- Removed fullwidth CJK comma `，` from `NUM` (`r'[+\-−]?\d+(?:[.,．]\d+)*'`) so CJK clauses (`1000000000000元，1000000001000，3-5天，系数1e-3。`) separate cleanly.
- Updated `unsupported` scientific & multi-hyphen range patterns with ASCII lookarounds (`(?<![A-Za-z0-9_.\-/])...(?![A-Za-z0-9_\-/]|\.\d)`) so ranges/scientific notation adjacent to Han/Hangul (`3-5天`, `系数1e-3。`, `3-5일`, `1.5e+10개`) AND before sentence-final `.` (`1e-3.`, `3-5.`) stay whole, classify as `unresolved` (`unsupported`), and emit warnings instead of partial numeric parsing.
- Fixed `numeric_fields(raw, lang)` multi-group check (`1 <= len(groups[0]) <= 3`) so invalid repeated separators (`12.34.56`, `1,23,45`, `1234,567,890`) raise `ValueError` -> `unresolved` + warning, recorded `separator_policy` in trace, and added `_excluded_regions` for URLs, emails, and versions (`v1.2.3`, `version 1.2.3`).

### T3 — AM/PM/a.m./p.m. Sentence Boundary
- Replaced `_AMPM_GROUP` so dotted forms only swallow a trailing `.` mid-sentence before whitespace + lowercase/comma, preserving sentence-ending `.` (e.g., `12:00 PM.`, `4:00 p.m.`, `4:00 p.m. Bring notes.`) across `vi`, `en`, `ko`, and `zh`.

### T4 — TTS Chunk Trace, QCS8550 Contracts, Duplicate-Span Mapping & `oov_delta`
- **`tts_format.py`:**
  - `adaptive` / `_adaptive_run` record `planned_chunks`, `attempted_chunks`, and `synthesized_chunks` (each with `text`, `normalized_length`, `preprocessed_length`, `utf8_bytes`, `word_count`, `max_units`, `depth`, `capped`, `retried`) while preserving `planned_chunks` and `audio_cap_retries`. Retried parent chunks appear in `attempted_chunks` (`capped=True`) and `audio_cap_retries`, never in `synthesized_chunks`.
  - `_map_protected_ranges` uses position-ordered `difflib.SequenceMatcher` alignment instead of first-occurrence `str.find` guessing, emitting explicit warnings and `span_mapping` `unmapped`/`ambiguous` entries when spans cannot be safely aligned.
- **`tts.py`:**
  - `TtsVi`: Added `_transform_vi_segments` so `latin_to_vi` preserves exact `tts_input_protected_ranges` and `span_mapping` even across duplicate spoken substrings; enforces `<=26` words and `<=512` UTF-8 bytes; reports per-call `oov_delta` in `self.last_oov_delta` and `self.format_trace['oov_delta']`.
  - `TtsZh`: Enforces both `<=20` chars and `<=112` UTF-8 bytes via custom `counter`; raises `ValueError('Melo input exceeds 112 UTF-8 bytes')` in non-legacy modes if an indivisible token exceeds 112 bytes.
  - `TtsEnKo`: Computes `preprocessed_chunk_lengths` from `synthesized_chunks` (`preprocessed_length`) against Supertonic's actual `N=96` preprocessed codepoint tensor budget (`supertonic_full_N96_L150_s5.onnx`; note `PLAN.md` line 44 text referenced `<= 576 cps`, whereas the fused QCS8550/CPU ONNX graph enforces `cp[1, 96]`).

---

## 3. Verification Results

### 3.1 Unit Tests (CRF Active)
Command:
```bash
PYTHONPATH=/Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge \
  /private/tmp/s2s-original-env/bin/python -m unittest \
  /Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_hybrid_format.py \
  /Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/tests_identifiers.py \
  /Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_format_boundaries.py \
  /Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_format_locale_boundaries.py \
  /Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_tts_chunk_trace.py -v
```
Result: **`Ran 38 tests in 0.203s` — `OK`** (0 failures, 0 errors, 0 skipped).

### 3.2 Unit Tests (`pycrfsuite` Mocked Missing — AIBOX aarch64 Wheelhouse Mode)
Command:
```bash
PYTHONPATH=/Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS:/Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge \
  /private/tmp/s2s-original-env/bin/python -c "import sys, unittest; sys.modules['pycrfsuite'] = None; suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_hybrid_format', 'tests.tests_identifiers', 'tests.test_format_boundaries', 'tests.test_format_locale_boundaries', 'tests.test_tts_chunk_trace']); res = unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(0 if res.wasSuccessful() else 1)"
```
Result: **`Ran 38 tests in 0.169s` — `OK`** (0 failures, 0 errors, 0 skipped).

### 3.3 Python 3.10 AST Compatibility Check
Command (`ast.parse(..., feature_version=(3, 10))` on all 4 modified `s2s_edge` files + 3 new test files):  
Result: **`PY310 AST OK: 7`** (exit code 0).

### 3.4 Host CPU ONNX Graph Check (`models/models.cpu.json`)
Executed real `Registry(..., backend='cpu', tts_format='hybrid').tts(lang).synthesize(text)` across 9 representative `vi`/`en`/`ko`/`zh` edge cases (including duplicate spans and long Korean `ko_04`):
- All 9/9 produced non-empty finite `float32` waveforms (`dur_s`: `3.251s` – `18.75s`, exit code 0).
- All `synthesized_chunks` stayed within QCS8550/CPU static budgets (`vi`: `max_words <= 22 <= 26`, `max_utf8 <= 119 <= 512`, `oov_delta = 0`; `en`/`ko`: `preprocessed_length <= 96`; `zh`: `utf8_bytes <= 58 <= 112`), with `retries = 0` and exact match between `planned_n`, `attempted_n`, and `synthesized_n`.
