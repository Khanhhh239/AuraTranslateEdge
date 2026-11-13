# OneVoice / AuraTranslateEdge — Báo Cáo Hệ Thống End-to-End (Sau Nâng Cấp Plan 01–03 & Kiểm Thử 5 Chiều)

- **Nhánh:** `feat/plan01-measurement`
- **Ngôn ngữ hỗ trợ:** 4 ngôn ngữ (`vi`, `en`, `ko`, `zh`) — 6 chiều E2E, kiểm thử đầy đủ 5 chiều BTC (`en-vi`, `ko-vi`, `vi-ko`, `vi-zh`, `zh-vi`) + CLI `vi->en`
- **Mục tiêu triển khai:** Tích hợp ASR tốc độ/chính xác mới, hoàn thiện chuẩn hóa biên TTS đa ngôn ngữ (Plan 02), bảo toàn số/thực thể/dấu câu từ nguồn sang đích (Plan 03), và đảm bảo khả năng chạy offline trên phần cứng **HSPTEK AIBOX 8550** (`QCS8550`, `linux/aarch64`, Python 3.10, không phụ thuộc bắt buộc `pycrfsuite` / `torch` / Metal khi chạy `source_mode=0`).

---

## 1. Kiến Trúc Pipeline End-to-End Hiện Tại

```
[Audio Input]
   │
   ▼
[Audio Segmenter & VAD]         ── Profile `legacy` (mặc định) / `upstream` (opt-in)
   │
   ▼
[ASR Đa Ngôn Ngữ]
   ├── vi: Zipformer-Small (CTC)
   └── en / ko / zh: SenseVoice-Small (bucketed ONNX + sensevoice_am.mvn)
   │
   ▼
[Source Preservation & Guard (Plan 03)]
   ├── `source_segmentation.py`: tách câu/đoạn bảo toàn ngữ cảnh trước MT
   ├── `punct_vi.py`: khôi phục dấu câu & viết hoa (`viBERT-CAPU`, ONNX)
   └── `numeric_guard.py`: gắn thẻ/ledger số, thực thể & đối chiếu đầu ra (`target_audit`)
   │
   ▼
[MT — NLLB-600M ONNX]          ── Encoder (`S=64`) + Unrolled Decoder
   │
   ▼
[TTS Format Boundary (Plan 02)]
   ├── `hybrid_textnorm.py` + `format_locales.py`: chuẩn hóa đa ngôn ngữ theo quy tắc + optional CRF
   ├── `tts_format.py`: adaptive chunking theo giới hạn word/byte/NFKD Jamo cho từng backend
   └── `tts.py`: thực thi synth + chunk trace (`planned == attempted == synthesized`)
   │
   ▼
[TTS Backend]
   ├── vi: Piper VITS ONNX
   ├── en / ko: Supertonic 3 ONNX
   └── zh: MeloTTS-ZH ONNX
```

---

## 2. Các Hạng Mục Đã Nâng Cấp

| Hạng mục | Nội dung chính | Bằng chứng kiểm thử |
|---|---|---|
| **ASR Speed & Accuracy** | Tích hợp `SenseVoice` bucketed ONNX, `sensevoice_am.mvn`, cấu hình `models.dev_asr_buckets.json`, công cụ `eval_asr.py`, `test_asr_host.py`, `evaluate_asr_integration.py` | Xem `s2s_edge/docs/asr_optimization.md` & `tests/test_asr_speed_accuracy_integration.py` |
| **Plan 01 + AIBOX 8550 Portability** | Gắn `tts_format` / `source_mode` / `vad_profile` / `ort_policy` vào instance; `/api/health` & launcher kiểm tra code fingerprint; sửa tương thích Python 3.10 AST, fallback khi thiếu `pycrfsuite`, đóng gói `build_package.py` (`linux_aarch64`) | [01-measurement/EXECUTION_REPORT.md](../docs/workflow/next-steps/01-measurement/EXECUTION_REPORT.md) |
| **Plan 02 — Format Boundary** | Hoàn thiện biên format cho số thập phân/âm/khoảng/khoa học cuối câu, tránh nhầm hậu tố chữ với đơn vị đo, giữ giới hạn byte/word/Jamo QCS8550 & trace chunk TTS | **38/38** unit tests pass (cả khi có & khi tắt `pycrfsuite`); **9/9** synth ONNX thật trong ngưỡng (`oov_delta = 0`). [02-format/EXECUTION_REPORT.md](../docs/workflow/next-steps/02-format/EXECUTION_REPORT.md) |
| **Plan 03 — Source Preservation** | Thêm `numeric_guard.py`, `source_segmentation.py`; giữ nguyên dấu câu/viết hoa sau MT; `target_audit` đối soát số liệu nguồn-đích qua ledger Web/CLI; tách sạch `torch`/Metal khỏi `source_mode=0` | **70/70** tests hồi quy pass; **20/20** tests không `pycrfsuite` pass; overhead CPU `guard_ms=0.482`, `split_ms=0.018`. [03-source-preservation/EXECUTION_REPORT.md](../docs/workflow/next-steps/03-source-preservation/EXECUTION_REPORT.md) |

---

## 3. Kiểm Định Tổng Hợp & Benchmark BTC 5 Chiều

### 3.1. Kết quả kiểm thử tự động & tương thích AIBOX 8550
- **Full Unittest (15 file):** **110/110** pass (`52.997s`).
- **Offline fallback không có `pycrfsuite` (Plan 02 + Plan 03):** **58/58** pass (`1.280s`).
- **Python 3.10 AST:** **55/55** file `.py` trong `s2s_edge/` hợp lệ (`bad_ast: []`).
- **Dependency isolation (`source_mode=0`):** không import `torch`/Metal.
- **CLI E2E cô lập (`tools/run_e2e.py --source-mode 0`):** pass cả `en->vi` (3/3 segment `source_guard="verified"`, `target_audit.mismatch=0`) và `vi->en`.

### 3.2. Kết quả Benchmark BTC 5 Chiều (`results/btc_5dirs_eval/`)
- **Script:** `tools/eval_btc_5dirs.py` — cấu hình `source_mode=0 (S2S_NEURAL_TN=0)`, `tts_format=hybrid`, `vad_profile=legacy`, `ort_policy=auto`, `backend=cpu`, `mode=sampled (seed=42, wall_total=426.067s)`: lấy **100% clip chứa số/thực thể** + mẫu cân bằng 30 clip/chiều (**150/410** clip).
- **Độ ổn định:** **150/150 OK (0 lỗi)**; TTS chunks `planned == attempted == synthesized` đạt **161/161/161**; 0 ca rớt tại `source_guard`.

| Chiều | Số clip | Entity clips | OK / Err | Mean Total (P50 / P95) | Mean ASR / MT / TTS | TargetAudit (pres/mis/unv/tot) | TTS Chunks |
|---|---:|---:|---:|---|---|---|---:|
| `en-vi` | 30 / 50 | 8 / 8 | 30 / 0 | 3.1668s (2.8789s / 5.5059s) | 0.6013 / 1.4625 / 1.1019s | 10/1/0/11 | 36/36/36 |
| `ko-vi` | 30 / 90 | 3 / 3 | 30 / 0 | 2.0989s (2.063s / 2.5495s) | 0.5122 / 0.6564 / 0.93s | 1/1/0/2 | 30/30/30 |
| `vi-ko` | 30 / 90 | 3 / 3 | 30 / 0 | 2.8669s (2.6708s / 3.7791s) | 0.1769 / 0.6163 / 2.0538s | 0/0/0/0 | 30/30/30 |
| `vi-zh` | 30 / 90 | 7 / 7 | 30 / 0 | 3.5641s (3.1242s / 6.4066s) | 0.1794 / 0.9147 / 2.4491s | 0/0/0/0 | 34/34/34 |
| `zh-vi` | 30 / 90 | 6 / 6 | 30 / 0 | 2.2344s (2.1161s / 2.9991s) | 0.5187 / 0.7746 / 0.9409s | 4/2/0/6 | 31/31/31 |

> Chi tiết mẫu câu & các ca `target_audit` phát hiện lỗi MT/ASR xem tại [results/btc_5dirs_eval/REPORT.md](../results/btc_5dirs_eval/REPORT.md).

---

## 4. Điểm Nghẽn Hiện Tại & Bước Tiếp Theo Trên Phần Cứng Thật
1. **Model ASR/MT là nguồn sai số chính còn lại:** `target_audit` và đối chiếu ref trên BTC chỉ ra lỗi dịch số/đơn vị của NLLB (`en-vi/EV-5301`: `800` → `08:00`, `ko-vi/VK-1348`: `15000원` → `15.000 người`, `zh-vi/VM-0484`: `101块钱` → `101 đô la`, `zh-vi/VM-4870`: `7点半` → `7h30`, `zh-vi/VM-5101`: `6000多个` → `6.000`) và lỗi đồng âm của ASR tiếng Việt (`vi-ko/VK-1939`: `80 USD` → `tám USD`, `vi-ko/VK-1348`: `15.000 won` → `mười ngàn quân`), không phải lỗi rơi/sai định dạng ở tầng TTS Hybrid.
2. **Triển khai HSPTEK AIBOX 8550 (`QCS8550`):** Build gói với wheelhouse `linux_aarch64` qua `s2s_edge/tools/build_package.py` và chạy runbook A0–A6 trực tiếp trên thiết bị (đo QNN HTP NPU + nghe thẩm định toàn bộ 410 clip BTC nếu cần).
