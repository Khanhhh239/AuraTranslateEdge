# Spec cho vòng cải thiện kế tiếp

Executor là Antigravity. Codex lập kế hoạch/review; người dùng nghe xác nhận audio. Workspace duy nhất: OneVoice_E2E_HybridTTS. Không tạo bản sao pipeline mới hoặc ghi đè thay đổi chưa commit.

## Mục tiêu

1. Chặn regression ranh giới mã và có bằng chứng tái lập được.
2. Format theo lớp dữ liệu/ngữ cảnh, không thay riêng một chuỗi hoặc thêm số bị mất.
3. Bảo toàn những entity nguồn xác định được; tách rõ unknown, mismatch và đã xác minh.

## Ràng buộc chung

- Giữ graph/weights và dataset cố định trong một so sánh; đổi một biến mỗi ablation.
- VI/EN/KO/ZH; code trong `s2s_edge/` **phải tương thích Python ≥3.10**: Host dev hiện dùng Python 3.11 (macOS arm64), còn thiết bị **HSPTEK AIBOX 8550** (`linux/aarch64`, QCS8550 Hexagon v73, offline, 12–16 GB RAM, 128 GB UFS) dùng Python 3.10 (`wheelhouse_cp310`) hoặc Python 3.12 (`wheelhouse`) với `QAIRT 2.51` (`qai_appbuilder 2.51.40`) hoặc container `docker_run.sh`.
- **Tương thích offline & ngân sách phần cứng AIBOX 8550:**
  - `s2s_edge/requirements.txt` chỉ giữ các dependency cốt lõi có wheel `linux_aarch64` (`numpy`, `scipy`, `soundfile`, `sentencepiece`, `kaldi-native-fbank`, `onnxruntime`, `psutil`); mọi gói tùy chọn (`python-crfsuite`, `openpyxl`, `sounddevice`, `jiwer`, `sacrebleu`) phải có đường chạy fallback trong `install.sh` và runtime.
  - **`source_mode=0` (`S2S_NEURAL_TN=0`) + `tts_format=hybrid` là đường chạy mặc định và bắt buộc khả thi trên AIBOX 8550** (không dùng `Qwen3-4B MLX4`, `ByT5`, `NeMo`, Apple Metal hay PyTorch MPS trong runtime thiết bị). Nếu caller yêu cầu tường minh `S2S_NEURAL_TN=1` trên Linux/NPU thiếu Metal, hệ thống phải fail rõ lỗi, không âm thầm fallback sang `0`.
  - Mọi cải thiện Plan 02/03 trên đường chạy sản phẩm phải dùng Python/stdlib + ONNX nhỏ gọn hiện có (`vibert_capu_int8` + tùy chọn `identifier.crfsuite`), giữ tổng RAM thiết bị trong ngưỡng <6 GB và không thêm dependency/model lớn ngoài kế hoạch riêng.
  - Khi chạy web/server với `--backend npu` (`QCS8550`), giữ `Registry`/QNN binaries resident (`keep_loaded=True`) theo hàng đợi tuần tự 1 worker để tránh cold reload ~2.2 GB/lượt, nhưng vẫn gắn fingerprint và `effective_profile`.
- Không lấy transcript tham chiếu thay đầu vào ASR; không tự suy ra 32 khi ASR trả 2.
- Mã giữ chữ số, thứ tự, số 0 đầu và dấu theo policy đã công bố. Canonical bỏ grouping phải có mapping với raw, không gọi đó là byte-identical.
- Dấu âm, thập phân, loại dữ liệu, đơn vị và meridiem là ngữ nghĩa; chỉ giữ dãy chữ số không đủ.
- Scientific/range chưa hỗ trợ có thể bảo toàn ký tự kèm warning; không tách thành số khác rồi báo thành công.
- Unknown/unverifiable không tính pass và không được lặng lẽ sửa output MT.
- Ngày mơ hồ không tự chọn DMY/MDY. Không xem mọi chuỗi dài là phone hoặc mọi dấu gạch nối là âm.
- Không đưa placeholder tùy ý qua NLLB pruned vocab rồi giả định model giữ nó.
- Nếu span vượt graph budget, trả lỗi/unsupported rõ, không cắt âm thầm hoặc nuốt exception.
- Log theo run ID, input hash, source/model/config fingerprint, seed/locale, phiên bản thư viện, thiết bị; không publish dữ liệu bộ đề/thu âm cá nhân.

## Tiêu chí đề xuất cho vòng này

- Regression đã xác định có expected rõ phải qua toàn bộ; không phát sinh silent content loss trên tập được đánh giá.
- Bộ mutation có nhãn phải bắt 100% mutation xác định được; đây là mục tiêu test, không tuyên bố accuracy thế giới thực.
- Luôn báo số mẫu và unknown coverage; không có golden transcript người thật thì không công bố WER/MOS.
- Default mới phải giữ đúng ZIP ở ca regression và không làm giảm metric numeric/identifier so baseline trên holdout. Nếu kết quả hòa, ưu tiên cấu hình nhẹ hơn.
- Latency chỉ chặn promotion khi tăng >10% warm median trên phép so paired cùng thiết bị mà không có cải thiện numeric được xác nhận. Đây là ngưỡng đề xuất của kế hoạch, không phải yêu cầu sản phẩm đã được đo. Báo cả cold, stage time, số đoạn; p95 tập nhỏ chỉ mang tính mô tả.

## Chưa thuộc phạm vi

Huấn luyện/đổi weights ASR, ITN VI đầy đủ, bảo đảm mọi định dạng trên thế giới, tự động sửa bản dịch không có alignment đáng tin, chứng nhận audio bằng back-ASR, hoặc compile/relink Context Binary `.bin` mới lên AI Hub cho các node NPU chưa biên dịch của nhánh `asr-speed-accuracy` (`_sv_frontend`, `asr_vi_decoder/joiner`, bucket `@w*`). Các việc đó tách thành kế hoạch P2 riêng sau khi P0/P1 ổn định trên 9 QNN binaries `models_qcs8550` hiện có.
