# Plan 03 Execution Report — Source Preservation & Numeric Guard

**Owner:** `source_guard`  
**Target Platform Priority:** AIBOX QCS8550 (`source_mode=0`, pure Python >=3.10 + ONNX CPU/NPU, `<5ms` host overhead, zero Apple Metal / `mlx_lm` / `torch` / `transformers` in runtime path, works with or without `pycrfsuite`).

## 1. Summary of Changes

### T1 — Pure-Python Semantic Numeric Guard (`numeric_guard.py`, `conversation.py`, `mt.py`, `neural_textnorm.py`)
- Added [numeric_guard.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/numeric_guard.py) defining `NumericSpan`, `GuardReport`, `extract_known_numeric_spans(text, lang, *, include_crf=False)`, and `guard_source_numeric_changes(before, after, lang, *, allow_outside_punct_and_case=True, include_crf=False)`.
- `extract_known_numeric_spans` reuses `@formatter`'s `HybridNormalizer.extract_spans` (pure-Python `learned_identifiers=False` by default) to share lexical boundaries and separator policies without duplicating regex parsers, plus typed context promotion for unpunctuated 12h meridiem (`330 pm` -> `3:30 PM`) and spelled temperature units (`37.5 degrees Celsius` / `37.5 độ C`).
- `GuardReport` distinguishes `"verified"`, `"invalid"`, `"unknown"`, and `"not_applicable"`, enforces exact characters on opaque spans (`signature=None`) and date component order (`21/12/2027` vs `12/21/2027`), catches uncovered written digits and multi-word/cued Vietnamese spoken numerals as `"unknown"` (`valid=False`, `issues=[]`), and verifies non-span token identity.
- **T1a (`source_mode=0`):** Integrated into [conversation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/conversation.py) (`restore_vi`, `Engine.prepare_text`, `Engine.translate_segment`, `Engine.translate_segment_stream`), recording `Engine.last_source_guard` and falling back to raw ASR text if punctuation/normalization alters protected spans or token identity.
- **T1b (`source_mode=1`, macOS dev):** Integrated into [neural_textnorm.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/neural_textnorm.py) via `_guard_canonical_identifiers`, retry-on-mutation inside Qwen `prepare_source`, span-safe fallback via `split_preserving_spans`, and trace records carrying `token_identity` + `numeric_validation` instead of unconditional `content_preserved: True`.

### T2 — Punctuation Preservation (`punct_vi.py`, `textnorm.py`, `conversation.py`)
- Updated [punct_vi.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/punct_vi.py):
  - Added `ViCapu.last_validation: GuardReport | None` reset per call.
  - Added `_protected_edit_constraints` and updated `_apply` to block `$TRANSFORM_VERB_*`, `$MERGE_*` (checked before `start == end`), opaque span case edits, and internal-span `$APPEND_*` punctuation insertions (never turning spoken `"không"` into `"0"`).
  - Replaced silent `except Exception: pass` in `ViCapu.__call__` chunk merging with token-alignment verification, source-indexed overlap splicing fallback, whole-output `guard_source_numeric_changes`, and fallback to `vi_punct` or raw input when invalid.
  - Wrapped `vocab.txt` and `verb-form-vocab.txt` reads in `with open(...)` context managers.
- Updated [textnorm.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/textnorm.py) `vi_punct` to guard `breaks` and clause boundaries against protected spans (using lazy `numeric_guard` import to avoid a circular dependency with `hybrid_textnorm.py`), preserve negation `"không"` vs cued `"số không"` and `có ... không` question marks, and expose `vi_punct.last_validation`.

### T3 — Span-Safe Source Segmentation (`source_segmentation.py`, `mt.py`, `conversation.py`)
- Added [source_segmentation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/source_segmentation.py) with `SpanBudgetError(ValueError)`, `load_nllb_encode(assets_dir=None)` (using SentencePiece `assets/nllb_spm.model`), and `split_preserving_spans(text, protected_spans, *, token_limit, encode, lang=None, merge_small=False)`.
- Never splits inside protected spans (`$15.50`, `37.5`, `3:30 PM`, `21/12/2027`, leading-zero phone/ID spans); prefers normal sentence/clause boundaries (`split_sentences` compatibility) while honoring `token_limit` (NPU `mt_enc` `S=64` -> `S - 2 = 62` tokens); raises `SpanBudgetError` explicitly if a single atomic span exceeds `token_limit`.
- Integrated into [mt.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/mt.py) (`Nllb.__call__`, `_split`, `_tr`, `MtFallback.__call__` re-raising `ValueError`/`SpanBudgetError` instead of silently truncating) and [conversation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/conversation.py) (`split_sentences` backward-compatible optional `token_limit`/`encode` and unified batch/stream preparation).

### T4 — Target Entity Audit & Provenance Ledger (`numeric_guard.py`, `conversation.py`, `tools/s2s_web.py`)
- Added `audit_target_entities(source_entities, target_text, tgt_lang, *, src_lang="en")` in [numeric_guard.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/numeric_guard.py), returning `{preserved, mismatch, unverifiable, items, counts, target_spans}` across `en`/`vi`/`ko`/`zh` without modifying target text.
- Exposed `target_audit` and `source_guard` in [conversation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/conversation.py) (`translate_segment` & `translate_segment_stream`).
- Updated [s2s_web.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tools/s2s_web.py) `Service.run` to record `audit_ledger` (`asr_raw -> prepared -> mt -> tts_text`, `source_guard`, `target_audit`), `target_audit_summary`, and `audit_ledger.jsonl` without silently mutating `mt` output.

## 2. Verification & Observed Results

### Unit & Regression Test Suites
1. **Full 10-file regression suite (with `pycrfsuite` available):**
   ```bash
   /private/tmp/s2s-original-env/bin/python -m unittest \
     OneVoice_E2E_HybridTTS/tests/test_numeric_guard.py \
     OneVoice_E2E_HybridTTS/tests/test_punctuation_preservation.py \
     OneVoice_E2E_HybridTTS/tests/test_source_segmentation.py \
     OneVoice_E2E_HybridTTS/tests/test_numeric_audit.py \
     OneVoice_E2E_HybridTTS/tests/test_asr_integration.py \
     OneVoice_E2E_HybridTTS/tests/test_asr_speed_accuracy_integration.py \
     OneVoice_E2E_HybridTTS/tests/test_runtime_profiles.py \
     OneVoice_E2E_HybridTTS/tests/test_aibox8550_and_launcher_p2.py \
     OneVoice_E2E_HybridTTS/tests/test_hybrid_format.py \
     OneVoice_E2E_HybridTTS/tests/tests_identifiers.py -v
   ```
   **Result:** `Ran 70 tests in 31.081s` — **OK** (0 failures, 0 errors, 0 skips).

2. **Plan 03 suite without CRF wheel (`sys.modules['pycrfsuite'] = None`):**
   - Executed `tests/test_numeric_guard.py`, `tests/test_punctuation_preservation.py`, `tests/test_source_segmentation.py`, `tests/test_numeric_audit.py`.
   - **Result:** `Ran 20 tests in 0.982s` — **OK**.

3. **Pure-Python CPU Latency & Static Compatibility (`source_mode=0` AIBOX 8550 checks):**
   - CPU latency benchmark on `"Please meet at 3:30 PM on 21/12/2027, flight VN 123, ticket $15.50, temp 37.5°C, call 0912345678."`: `guard_ms=0.482`, `split_ms=0.018` (`<5ms` budget).
   - Python 3.10 AST parse (`ast.parse(..., feature_version=(3, 10))`) across all 8 modified/created production files: `ast_py310_ok 8`.
   - AST import scan across `numeric_guard.py`, `source_segmentation.py`, `punct_vi.py`, `textnorm.py`, `mt.py`, `conversation.py` for `mlx_lm`, `torch`, `transformers`, `neural_textnorm`: `ast_bad_imports: []`.

### Isolated CLI E2E Verification (`source_mode=0`, `--out-dir /tmp/plan03_e2e_check`)
- **EN -> VI (`en_to_vi_special_input.wav`):**
  - Command: `/private/tmp/s2s-original-env/bin/python OneVoice_E2E_HybridTTS/tools/run_e2e.py en vi OneVoice_E2E_HybridTTS/results/neural_fix/final/audio/en_to_vi_special_input.wav --source-mode 0 --vad-profile legacy --ort-policy basic --out-dir /tmp/plan03_e2e_check/en_vi`
  - Observed segments:
    1. `"The meeting is at 3:30 pm on November 15, 2026, the ticket costs $15.50 with a 15% discount."` -> `"Cuộc họp là lúc 3:30 tối vào ngày 15 tháng 11 năm 2026, vé có giá 15,50 đô la với giảm giá 15%."` (`source_guard='verified'`, `target_audit={'total': 4, 'preserved': 4, 'mismatch': 0, 'unverifiable': 0}`, `source_normalize_s=0.00087s`)
    2. `"The temperature is 37.5 degrees Celsius."` -> `"Nhiệt độ là 37,5 độ C."` (`source_guard='verified'`, `target_audit={'total': 1, 'preserved': 1, 'mismatch': 0, 'unverifiable': 0}`, `source_normalize_s=0.00534s`)
    3. `"My flight is VN 123, and my ZIP code is 67297."` -> `"Chuyến bay của tôi là VN 123, và mã địa chỉ là 67297."` (`source_guard='verified'`, `target_audit={'total': 2, 'preserved': 2, 'mismatch': 0, 'unverifiable': 0}`, `source_normalize_s=0.00049s`)
- **VI -> EN (`vi_to_en_special_input.wav`):**
  - Command: `/private/tmp/s2s-original-env/bin/python OneVoice_E2E_HybridTTS/tools/run_e2e.py vi en OneVoice_E2E_HybridTTS/results/neural_fix/final/audio/vi_to_en_special_input.wav --source-mode 0 --vad-profile legacy --ort-policy basic --out-dir /tmp/plan03_e2e_check/vi_en`
  - Observed segment: spoken Vietnamese numbers preserved cleanly in `asr` (`source_guard='unknown'`, no `"không" -> "0"` rewrite), `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`.

## 3. Owned Files Touched
- [numeric_guard.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/numeric_guard.py) (new)
- [source_segmentation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/source_segmentation.py) (new)
- [punct_vi.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/punct_vi.py)
- [textnorm.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/textnorm.py)
- [mt.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/mt.py)
- [conversation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/conversation.py)
- [neural_textnorm.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/s2s_edge/s2s_edge/neural_textnorm.py)
- [s2s_web.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tools/s2s_web.py)
- [test_numeric_guard.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_numeric_guard.py) (new)
- [test_punctuation_preservation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_punctuation_preservation.py) (new)
- [test_source_segmentation.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_source_segmentation.py) (new)
- [test_numeric_audit.py](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/tests/test_numeric_audit.py) (new)
- [EXECUTION_REPORT.md](file:///Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS/docs/workflow/next-steps/03-source-preservation/EXECUTION_REPORT.md) (new)
