# P0/P1: bảo toàn nguồn và audit số qua MT

**Executor:** Antigravity. **Spec:** `../SPEC.md`. **Goal:** không làm đổi/mất entity nguồn đã xác định trên đường chạy **AIBOX 8550 (`source_mode=0`)** lẫn dev macOS (`source_mode=1`); không tự “sửa” điều ASR chưa nhận. **Architecture:** numeric evidence → semantic guard → span-safe segmentation (tôn trọng NPU `S=64`) → target audit. **Tech:** Python ≥3.10 + ONNX (`vibert_capu_int8`) + CRF tùy chọn cho target AIBOX 8550; `neural_textnorm` (`Qwen/ByT5/NeMo`) chỉ chạy dev macOS; chưa thêm model/weights.

Phụ thuộc: baseline/profile từ 01; dùng lexical boundary đã chốt trong 02 cho written numbers. Không tự viết parser nguồn thứ hai có policy khác formatter. Spoken-VI chưa hiểu thì giữ token và báo unknown; chưa yêu cầu ITN đầy đủ. Mọi logic bảo toàn nguồn cốt lõi (T1a, T2, T3, T4) **không được phụ thuộc `mlx_lm`, `torch`, `transformers`, Apple Metal hay GPU** để chạy mượt trong ngân sách RAM 12–16 GB của AIBOX 8550.

## T1 - Semantic guard trước/sau chuẩn bị nguồn & đường chạy AIBOX 8550 (`source_mode=0`)

**Files:** thêm `s2s_edge/s2s_edge/numeric_guard.py`, `tests/test_numeric_guard.py`; sửa `conversation.py`, `mt.py` (cho **T1a AIBOX 8550 P0 `source_mode=0`**) và `neural_textnorm.py` (cho **T1b dev macOS `source_mode=1`**).
**Interfaces:**
- `NumericSpan(start, end, kind, raw, signature: tuple | None)`.
- `extract_known_numeric_spans(text, lang) -> list[NumericSpan]`: adapter thuần Python từ cùng lexical/type policy formatter (<5 ms CPU, chạy tốt khi thiếu CRF wheel), không render lời đọc vào nguồn.
- `guard_source_numeric_changes(before, after, lang) -> GuardReport`; report có `valid: bool`, `issues: list`, `unknowns: list`, `warnings: list`, `status` (verified/invalid/unknown/not_applicable), `fallback_policy: str | None`. Unknown không coi là xác minh semantic thành công. Signature gồm sign, integer/fraction precision, entity type/unit hoặc characters của ID; locale và meridiem tường minh.

`GuardReport.valid=True` chỉ khi không có issues và không có unknown numeric candidates. Không có entity số thì ghi not_applicable riêng, không suy ra model hiểu mọi văn bản. Span signature=None là opaque: phải giữ nguyên ký tự nguồn, trừ equivalence grouping/casing được khai báo. Thay punctuation trong opaque span phải bị chặn; không được chỉ warning rồi chấp nhận. Với unknown giữ nguyên vẫn không có semantic-success: dùng nguyên nguồn và log unknown thay vì cho model tự sửa.

- [ ] **T1a (AIBOX 8550 P0 — `source_mode=0` trong `conversation.py` / `mt.py`):** bảo vệ span số/tiền/nhiệt độ/giờ/ID (`$15.50`, `37.5°C`, `3:30 PM`, `VN 123`, `0912345678`, `1e-3`, `3-5`) khỏi bước tách câu theo dấu chấm trong `mt.py` (`re.split(r"(?<=[.!?])\s+")`) và gắn guard xác nhận từ `asr_raw` → `prepared_source` → trước MT trên đường chạy không-Metal. Nếu chuẩn hóa edge (`textnorm` / `CAPU` / `identifier`) làm sai signature, fallback tức thì về nguồn thô có bảo vệ span.
- [ ] **T1b (macOS Dev-Only — `source_mode=1` trong `neural_textnorm.py`):** Failing mutation tests: mất dấu âm, thập phân→giờ, ngày→integer, mất 0 đầu, đảo/trùng/mất chữ số, đổi đơn vị; opaque `1e-3→1e3`, `3-5→35`, ngày mơ hồ `03/04→03:04` không được xác nhận hợp lệ. Kiểm cả `canonical_identifiers` trước Qwen; không chỉ guard output cuối. Guard kiểm định chạy đầy đủ trên macOS Metal, bỏ qua/báo rõ khi chạy trên Linux AIBOX 8550 không có Metal.
- [ ] Accept casing/whitespace và dấu câu **ngoài span**. Viết rõ các equivalence cho phép; không tự chấp nhận đổi 3:30 PM↔15:30 cho tới khi có parser/meridiem xác nhận. Dạng 330 pm muốn khôi phục giờ phải có typed context và test riêng, không coi dãy digits giống nhau là đủ.
- [ ] Nối guard vào `conversation.py` (cả `mode=0` và `mode=1`) và `neural_textnorm.py`. Thay metadata `content_preserved` mơ hồ bằng `token_identity` và `numeric_validation` status. Invalid: retry có giới hạn (với LLM dev) hoặc fallback nguyên nguồn + warning (`mode=0` & `mode=1`); unknown: ghi rõ, không tự bù chữ số.
- [ ] Run `python -m unittest discover -s tests -p test_numeric_guard.py -v`. Đạt 100% mutation có expected xác định trong tập đóng băng trên cả chế độ edge (`source_mode=0`) lẫn neural (trên macOS nếu có Metal), báo thêm false rejection/unknown; không tuyên bố accuracy mọi text.

## T2 - CAPU không được bỏ chunk/đổi numeric words

**Files:** `punct_vi.py`, `textnorm.py`, `numeric_guard.py`; thêm `tests/test_punctuation_preservation.py`.
**Interface:** CAPU vẫn trả string, thêm `last_validation: GuardReport` được cập nhật theo mỗi call, gồm warnings/issues/unknowns và policy fallback; không dùng lại trạng thái call trước. Alignment phải giữ nguyên thứ tự token nguồn ngoài casing/punctuation được phép. Hoạt động sạch trên AIBOX 8550 với `vibert_capu_int8.onnx` (`ORT_ENABLE_BASIC` / `enable_cpu_mem_arena=False`) **lẫn khi fallback `vi_punct` thuần Python** nếu thiếu `assets/vi_capu`.

- [ ] Mock merge fail ở câu >48 từ, có số/mã ở overlap: suffix không được mất/trùng. Tạo câu phủ định “không” và zero trong chuỗi số làm đối chứng.
- [ ] Chặn punctuation edit nằm trong numeric span đã xác định. Khi overlap merge không tin cậy: fallback raw nguồn đầy đủ hoặc giữ chunk nguồn có mapping, ghi warning; không except-pass.
- [ ] Với spoken numeral chưa xác định giá trị, chỉ bảo vệ token identity và boundary đã tag; không tự đổi “ba bảy năm” thành 37.5.
- [ ] Run targeted test (cả INT8 CAPU ONNX lẫn rule fallback); thêm 1 clip VI dài phù hợp để kiểm CAPU thật, phân biệt fixture tổng hợp và bản ngữ. Log raw→CAPU→postprocess để biết chặng làm mất số.

## T3 - Một splitter bảo toàn span cho batch/stream/MT (tương thích NPU `S=64`)

**Files:** thêm `s2s_edge/s2s_edge/source_segmentation.py`, `tests/test_source_segmentation.py`; sửa `mt.py`, `conversation.py`.
**Interface:** `split_preserving_spans(text, protected_spans, *, token_limit, encode) -> list[str]`. Chunks là các vùng nguồn nguyên thứ tự, không vượt graph budget (mặc định đọc theo SentencePiece `assets/nllb_spm.model` và tham số `S` của NPU `mt_enc`, ví dụ `S=64` trên `models_qcs8550`!). Dùng ở `Nllb.__call__`, `_split` và `conversation.split_sentences`; giữ public API cũ qua wrapper nếu cần.

- [ ] Failing cases: `$18.25`, `36.8`, `3:30`, `21/12/2027`, phone 0 đầu, VN247 sát câu/chunk limit (gồm giới hạn chặt `S=64` của QCS8550 NPU); thêm câu thường để không làm mất ranh giới hợp lệ.
- [ ] Assert nối lại chunks giữ toàn bộ token/canonical source và không cắt nội bộ span (dù có `.` trong `$15.50` hay `37.5` khi `source_mode=0`). Một span vượt budget (`>S` hoặc `>Tm`): explicit unsupported/error, không silently truncate.
- [ ] Dùng translate_atomic chỉ sau khi chunk đã được đo token; không bypass toàn bộ sentence splitting rồi chấp nhận graph NPU `S=64` cắt nguồn.
- [ ] Run targeted test, rồi test `source_mode=0` (bắt buộc qua sạch ở đây cho AIBOX 8550!) và `source_mode=1` (trên macOS dev), batch và stream trên cùng text; sau đó S2S thật trên các chiều bị ảnh hưởng (và xác minh trên hộp QCS8550 ở cổng A0–A6 nếu có thiết bị). Không regex hậu sửa mọi dấu chấm trong output MT.

## T4 - Audit target entities, chưa repair (P1)

**Files:** `numeric_guard.py`, `conversation.py`, `tools/s2s_web.py`; thêm `tests/test_numeric_audit.py`.
**Interface:** `audit_target_entities(source_entities, target_text, tgt_lang) -> {preserved,mismatch,unverifiable}`. Lưu ledger ASR raw→prepared→MT→tts_text với provenance (chạy nhẹ ngay trên `source_mode=0` của AIBOX 8550 `s2s_web.py --backend npu`). Target viết số bằng chữ mà parser chưa hiểu phải unverifiable, không tự pass/fail.

- [ ] Expected tests: target sai giá trị, đơn vị hoặc meridiem xác định được phải mismatch; dạng tương đương được khai báo phải preserved; ambiguous/spoken chưa hỗ trợ phải unverifiable. Kiểm lặp nhiều số giống nhau để không alignment nhầm occurrence.
- [ ] Mặc định chỉ audit/warning. Không thay câu dịch, không chèn placeholder vào NLLB hoặc khôi phục số theo gold transcript.
- [ ] Báo entity accuracy với mẫu số/correct/wrong/unknown theo language/type, tách lỗi ASR so audio gold với lỗi introduced-after-ASR. Audio chưa có người nghe không phải MOS.

**Output:** `EXECUTION_REPORT.md` và raw evidence theo 01. 32→2 hoặc thiếu “phẩy” trong ASR vẫn là issue decoder/audio riêng; ghi nhận bằng chứng để lập kế hoạch ASR tiếp theo, không mở scope huấn luyện trong task này.
