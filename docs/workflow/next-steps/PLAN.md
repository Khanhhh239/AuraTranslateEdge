# Roadmap sau review ASR và format

**Executor:** Antigravity theo phân công của người dùng; Codex chỉ plan/review. Không yêu cầu tự chuyển sang Codex native/subagent execution.

**Goal:** giảm lỗi số/mã có bằng chứng trên **cả host phát triển lẫn thiết bị đích HSPTEK AIBOX 8550**, không đánh đổi bằng thay đổi mặc định chưa kiểm chứng.
**Architecture:** profile có thể đối chứng (`source_mode=0` mặc định trên Linux/NPU, `1` chỉ cho dev macOS), formatter có span/giá trị, guard và splitter bảo toàn trước MT (hoạt động trực tiếp ở `source_mode=0`), audit giá trị qua từng chặng.
**Tech stack:**
- **Target thiết bị (HSPTEK AIBOX 8550 — chính):** `linux/aarch64`, QCS8550 Hexagon v73 NPU (`QAIRT 2.51` / `qai_appbuilder 2.51.40`, 9 context binaries `.bin` ~4 GB đĩa / ~2.2 GB RAM mỗi chiều, 12–16 GB RAM tổng), Python 3.10 (`wheelhouse_cp310`) hoặc Python 3.12 (`wheelhouse`) / `docker_run.sh`, `S2S_NEURAL_TN=0`, `S2S_TTS_FORMAT=hybrid`, `vibert_capu_int8` (ONNX CPU `enable_cpu_mem_arena=False`), `identifier.crfsuite` tùy chọn (`python-crfsuite` optional wheel). **Không có Apple Metal / MLX / PyTorch MPS (`Qwen3-4B MLX4`, `ByT5`, `NeMo` không chạy trên 8550).**
- **Host dev / CPU reference (macOS / Linux):** Python 3.11+, ONNX CPU (`models/models.cpu.json`), CRF và `source_mode=1` (chỉ trên macOS để đối chứng); chưa đổi weights hay đòi hỏi compile NPU `.bin` mới cho P0.
**Spec:** `SPEC.md`. **Review:** `REVIEW.md`. **Hardware baseline:** `s2s_edge/docs/aibox8550_report.md`, `s2s_edge/docs/device_plan.md`, `s2s_edge/docs/chip_results.md`. **Baseline:** `baseline_manifest.json`; HEAD chỉ là nền Git, working tree còn thay đổi chưa commit.

## Thứ tự triển khai

1. **P0 - An toàn, đo đúng & khả năng đóng gói AIBOX 8550:** `01-measurement/PLAN.md`. Đóng băng baseline, VAD mới opt-in (`legacy` mặc định), profile web/CLI rõ ràng (`--backend {cpu,npu}`, `--models`, `--keep-loaded` cho NPU), mặc định `S2S_NEURAL_TN=0` trên Linux/NPU (fail tường minh nếu ép `1` không có Metal), kiểm tra wheelhouse offline `linux_aarch64` (`install.sh` dung nạp optional wheels) và sửa spawn race P2. Hoàn tất trước khi đổi default hoặc so kết quả các bước sau.
2. **P0 - Formatter (tương thích Python ≥3.10, không phụ thuộc Metal/CRF bắt buộc):** `02-format/PLAN.md`. Sửa bốn nhóm lỗi đã tái lập trong `hybrid_textnorm.py` / `format_locales.py` / `tts_format.py`; đóng băng 17 edge cases và bộ đối chứng. Test **cả khi có lẫn khi thiếu `identifier.crfsuite` / `pycrfsuite`** để bảo đảm chạy sạch trên AIBOX 8550. Độc lập với source guard, không trộn sửa ASR/MT vào commit parser.
3. **P0 - Nguồn trước MT (ưu tiên `source_mode=0` cho AIBOX 8550):** `03-source-preservation/PLAN.md`. Tách **T1a (AIBOX 8550 P0, `source_mode=0`, pure Python <5 ms / 0 MB nặng):** guard ngữ nghĩa số + `source_segmentation.py` bảo toàn `$15.50`/`37.5°C`/mã 0 đầu trước `mt.py` (tôn trọng ngân sách token NPU `mt_enc S=64`), và **T2 CAPU không mất từ** cho cả `vibert_capu_int8` lẫn fallback `vi_punct`. `T1b` nối guard cho `neural_textnorm.py` (`source_mode=1`) chỉ giữ ở nhánh dev macOS. Không khôi phục điều ASR chưa nhận ra.
4. **P1 - Audit, đóng gói NPU & đánh giá trên thiết bị AIBOX 8550:** ghi ledger trước/sau MT, test pipeline/web với cùng profile (`--backend cpu` trên host + `build_package.py` & `tools/runbook.sh` A0–A6 trên hộp 8550 thật, kiểm tra ALSA mic/speaker và soak nhiệt), và người dùng nghe xác nhận. Nhánh tối ưu NPU mới từ `asr-speed-accuracy` (compile/relink `.bin` trên AI Hub cho `_sv_frontend`, `asr_vi_decoder/joiner`, bucket `@w*`) tách thành pha P2 riêng sau khi P0 chạy ổn trên 9 QNN binaries đã profile.

Antigravity làm từng kế hoạch và lưu `EXECUTION_REPORT.md` cùng thư mục. Đưa báo cáo + diff cho Codex review trước khi mở rộng bước tiếp theo. Không tự push/merge hoặc restart server đang có job.

## Review focus xuyên suốt

- ZIP/phone nằm đúng cạnh khoảng dừng: không cắt 67297 thành 6 và 729,7 (01).
- Dấu câu CJK, dấu âm và phone thứ hai trong danh sách: không ghép/đổi giá trị (02).
- Model chỉ đổi dấu nhưng vẫn giữ chữ số: guard phải nhận ra ngữ nghĩa khác (03).
- Câu dài qua CAPU overlap và numeric span sát graph budget: không mất/trùng token (03).
- Cùng một server/profile nhưng reused cache hoặc audio cap retry: trace phải khớp code/config/chunk thực (01,02).

## Bàn giao kết quả

- Log lệnh đầy đủ và exit status; phân biệt unit, TTS graph, S2S, HTTP và mic thật.
- Mỗi ca lưu input/ASR raw/prepared/MT/tts_text/actual chunks/output + hash/metadata.
- Báo riêng correct/wrong/unknown entity theo language/type; phát âm cần người nghe, không quy ASR nghe lại thành gold.
- Không ghi đè `results/asr_integration/before`; rollback theo patch/hash, không reset cả repo.

## Gate kết thúc vòng

Chỉ xem vòng hoàn tất khi từng tiêu chí trong SPEC được đối chiếu, regression P0 không còn, cấu hình web đang dùng có test phù hợp, và hạn chế/chưa kiểm chứng được ghi rõ. Nếu thiếu audio holdout bản ngữ thì vẫn có thể hoàn tất implementation nhưng **chưa kết luận nâng chất lượng ASR hoặc promote toàn cục**.
