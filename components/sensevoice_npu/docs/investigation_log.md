> **Nhật ký điều tra lịch sử (đọc [README.md](../README.md) trước).** Tài liệu này ghi lại quá trình theo thời điểm viết,
> nên **một số kết luận đã bị thay thế** bởi các phát hiện sau đó:
>
> * Mục 11 ("đã sửa thành công") dừng ở **frontend v2** (bỏ scale ×32768). Bản đó chỉ đổi lỗi tràn số trên thành tràn số
>   dưới (51.8% giá trị mel về 0 trong fp16). Bản cuối là **frontend v3 (chuẩn hóa theo từng frame)**, xem README mục 3.2.
> * Phần tiếng Anh còn lệch (en ~12.8%) ở đây được quy cho "giới hạn nhiễu fp16 không tránh được của NPU". Nhận định đó
>   **sai**: nguyên nhân là tràn số dưới ở frontend v2; với v3 en đạt 8.98% (fp32: 8.45%) trên bộ 90 câu.
> * Mục 9 và mục 6 (giả thuyết "quantize encoder quá giòn", "cần recalibrate") đã bị bác bỏ; calibration của encoder đang dùng
>   (`jprlmj9kp`) được tạo từ fbank fp32 chạy CPU nên hợp lệ, không cần quantize lại.
>
> Các số đo, mã job và quan sát ở các mục khác vẫn đúng tại thời điểm đo.

---

# Báo cáo điều tra: SenseVoice-Small quantization trên NPU thật (Dragonwing IQ-9075)

**Ngày:** 2026-09-29 → 2026-09-30
**Phạm vi:** `AuraTranslateEdge-OneVoice/src/step1_asr/` (export script của Lê Gia Khánh)
**Yêu cầu ràng buộc:** 100% compute chạy trên NPU (không fallback CPU), đo chất lượng bằng WER/CER thật trên từng câu.

---

## 1. Tóm tắt điều hành

Model SenseVoice-Small sau khi export + quantize (W8A16) để chạy trên NPU Dragonwing IQ-9075 cho kết quả **gần như hỏng hoàn toàn** khi chạy thật trên phần cứng (en WER ~57%, zh CER ~99%, ko WER ~100%), mặc dù mô phỏng cục bộ (local simulation) với đúng bộ trọng số đã quantize cho kết quả gần bằng baseline fp32 (en 7.56%, zh 11.20%, ko 38.27%).

Đã tìm và sửa **5 bug thật** trong pipeline export (length masking, FFT bin sai, pre-emphasis sai, length bị quantize nhầm, out_lens tự bão hòa) — các bug này có thật và đã fix, nhưng KHÔNG phải nguyên nhân của thất bại trên phần cứng thật.

Sau khi cô lập nguyên nhân bằng graph-surgery + so sánh cosine similarity, xác định gốc rễ: **frontend (fbank/FFT/mel/CMVN) có dynamic range quá rộng, không tương thích với quantization W8A16 một-scale của encoder**. Đã thử kiến trúc **split-graph** (frontend giữ nguyên fp32/không quantize + encoder-classifier quantize riêng) — mô phỏng cục bộ xác nhận kiến trúc này đúng hướng.

Tuy nhiên khi triển khai THẬT trên NPU silicon, kết quả vẫn thất bại tương tự bản gốc. Đã thử **2 hướng sửa** dựa trên giả thuyết mới, cả hai đều **không hiệu quả** (dữ liệu chi tiết ở mục 4). Kết luận hiện tại: vấn đề nằm ở **độ giòn (brittleness) cố hữu của quantization W8A16** trên encoder, không phải do lệch phân phối calibration như đã giả định. Đề xuất hướng đi tiếp theo ở mục 6, ưu tiên **deploy encoder ở fp16-trên-NPU (bỏ quantize int8/int16)**.

---

## 2. Bối cảnh: 5 bug đã fix (không phải nguyên nhân của vấn đề hiện tại)

| # | Bug | Fix |
|---|---|---|
| 1 | Thiếu length-masking → token_ids không bị cắt theo độ dài thật | Thêm `wav_len` input, mask `token_ids` bằng `torch.where(length_mask, token_ids, zeros)` |
| 2 | FFT lấy sai bin (`real[:, 1:]` thay vì `real[:, :-1]`) | Sửa `power_no_dc = real[:, :-1]**2 + imag[:, :-1]**2` |
| 3 | Pre-emphasis sai xử lý frame đầu | Thêm `frames_pe[:, 0] = frames[:, 0] * (1 - 0.97)` |
| 4 | Length computation bị cast sang float → vô tình bị QDQ-wrap khi quantize | Viết lại bằng pure integer ops (`torch.int64`, `//`) |
| 5 | `out_lens` nội bộ của model bão hòa ở giá trị 180 bất kể input (do bị quantize) | Không dùng `out_lens` của model, tự tính `valid_len = speech_lengths + 4` (offset verify chính xác qua test fp32 trực tiếp) |

Các bug này là có thật, đã verify bằng test trực tiếp (không đoán), và đã fix vĩnh viễn trong `step4_s1_export_sensevoice_e2e_unified.py`. Nhưng sau khi fix, chất lượng NPU thật vẫn kém với zh/ko → dẫn tới điều tra sâu hơn.

---

## 3. Phát hiện gốc rễ: frontend không tương thích với quantization một-scale

Phương pháp: graph surgery trực tiếp trên ONNX quantized graph, bypass từng phần bằng tensor fp32 thật để cô lập nguồn lỗi.

- **A1 (length correlation):** không kết luận được.
- **A2 (bypass classifier):** loại trừ classifier là nguyên nhân.
- **A4 (magnitude stats):** không phát hiện bất thường ở giá trị trung gian.
- **Cosine-similarity theo layer** (so `LayerNorm` fp32 vs quantized qua 142 layer): phát hiện phân kỳ ngay từ layer đầu tiên (`encoders0[0]`).
- **A3 (bypass frontend — breakthrough):** thay tensor `add_6` (fbank+positional-encoding) bằng giá trị fp32 THẬT (tính từ model gốc), giữ nguyên phần encoder+classifier đã quantize. Kết quả: zh_2 CER giảm từ **100% → 8.1%**. Xác nhận: lỗi nằm ở frontend bị quantize, không phải ở encoder/classifier.

**Kết luận:** dynamic range của fbank (sau FFT + log-mel + CMVN) quá rộng để biểu diễn chính xác bằng 1 scale quantization duy nhất — đây là nguyên nhân gốc gây collapse ở zh/ko.

---

## 4. Kiến trúc sửa: Split-graph — và tại sao vẫn thất bại trên phần cứng thật

### 4.1 Thiết kế
Tách 1 graph lớn thành 2 graph compile riêng:
- **Graph 1 (Frontend):** wav → fbank + speech_lengths. KHÔNG quantize (giữ float).
- **Graph 2 (Encoder+Classifier):** fbank + speech_lengths + language + textnorm → byte_stream. Quantize W8A16 (calibrate bằng fbank sinh ra từ Graph 1).

Host chỉ route tensor giữa 2 lần gọi NPU tuần tự — không có compute thật trên CPU → vẫn giữ đúng ràng buộc "100% NPU".

### 4.2 Kết quả mô phỏng cục bộ (local simulation) — THÀNH CÔNG
Chạy Graph 1 bằng ONNX Runtime CPU fp32 thật, feed vào Graph 2 (trọng số đã quantize thật từ AI Hub) cũng bằng ONNX Runtime local:

| Ngôn ngữ | fp32 baseline | Split-graph (local sim) | Bản gốc (1 graph, lỗi) |
|---|---|---|---|
| en | 6.23% | 7.56% | 39.53% |
| zh | 11.20% | **11.20%** (=baseline) | ~90%+ |
| ko | 38.27% | **38.27%** (=baseline) | ~90%+ |

→ Kiến trúc đúng hướng, gần như khớp tuyệt đối với baseline.

### 4.3 Kết quả trên NPU thật (v1) — THẤT BẠI

Compile cả 2 graph lên NPU thật (Dragonwing IQ-9075), chạy inference thật, chain output thật của Graph 1 → input thật của Graph 2:

| Ngôn ngữ | Local sim (mục tiêu) | **NPU thật (v1)** |
|---|---|---|
| en WER | 7.56% | **57.15%** |
| zh CER | 11.20% | **99.05%** |
| ko WER | 38.27% | **100.00%** |

Pattern lỗi: zh/ko hầu hết collapse về ký tự lặp/rỗng ("。", ".") — dấu hiệu **CTC blank-collapse** (input bị hỏng cấu trúc), không phải suy giảm chất lượng thông thường.

### 4.4 Giả thuyết 1 — `--quantize_io` — ĐÃ BÁC BỎ

**Giả thuyết:** compile job của frontend dùng flag `--quantize_io`, ép input/output (gồm `fbank`) bị nén xuống precision thấp ngay cả khi graph không quantize.

**Kiểm chứng:** so sánh trực tiếp fbank output từ NPU giữa job có và không có `--quantize_io` → **giống hệt bit-by-bit** (`max_abs_diff = 0.0`).

**Kết quả sau khi bỏ flag (v2):**

| Ngôn ngữ | v1 (có quantize_io) | v2 (không quantize_io) |
|---|---|---|
| en WER | 57.15% | 57.15% |
| zh CER | 99.05% | 99.05% |
| ko WER | 100.00% | 100.00% |

→ **Giống hệt v1.** Giả thuyết sai.

### 4.5 Giả thuyết 2 — lệch phân phối calibration (fp32-CPU vs fp16-NPU thật) — ĐÃ BÁC BỎ

**Giả thuyết:** NPU Hexagon không có ALU tính fp32 gốc — mọi graph "float" khi chạy trên NPU thực chất thực thi ở fp16. So sánh trực tiếp cho thấy fbank thật từ NPU lệch đáng kể so với fp32-CPU (cosine similarity chỉ 0.82–0.996, rel-L2 tới 0.42–0.59 tùy mẫu). Vì encoder được quantize/calibrate bằng fbank tính từ CPU fp32 (không phải fp16 thật lúc suy luận), có thể đây là nguyên nhân — quantization range bị lệch so với phân phối triển khai thật.

**Fix thử:** chạy frontend NPU thật trên bộ calibration rộng hơn (25 mẫu) → dùng chính fbank fp16 thật đó để quantize lại encoder → compile lại.

**Kết quả (v3):**

| Ngôn ngữ | v1/v2 (broken) | **v3 (recalibrate bằng fp16 thật)** | Mục tiêu |
|---|---|---|---|
| en WER | 57.15% | **58.20%** | 7.56% |
| zh CER | 99.05% | **98.10%** | 11.20% |
| ko WER | 100.00% | **100.00%** | 38.27% |

→ **Không cải thiện đáng kể** (sai khác nằm trong nhiễu ngẫu nhiên). Giả thuyết sai.

---

## 5. Đánh giá lại: đây không phải vấn đề calibration

Việc recalibrate ĐÚNG bằng dữ liệu fp16 thật (khớp chính xác phân phối lúc suy luận) mà kết quả không đổi loại trừ khả năng "quantization range bị lệch do nguồn calibration sai". Bằng chứng:

1. Pattern lỗi giống hệt nhau qua cả 3 lần chạy (v1, v2, v3): zh/ko collapse về gần như toàn bộ ký tự lặp/rỗng, en_4 luôn đúng 100% trong khi en_0/1/3 luôn sai nặng.
2. `en_4` là outlier nhất quán — WER=0% ở CẢ 3 lần chạy — cho thấy độ nhạy quantization phụ thuộc vào ĐẶC ĐIỂM TỪNG MẪU (có thể là độ dài, năng lượng tín hiệu, hoặc pattern tần số cụ thể), không phải một lỗi hệ thống có thể sửa bằng cách "calibrate đúng hơn".
3. → Kết luận: **quantization W8A16 của encoder quá "giòn" (brittle)** — không chịu được BẤT KỲ mức nhiễu nào khác với input lý tưởng, kể cả nhiễu đã "khớp đúng" nguồn gốc. Đây là vấn đề ở mức kiến trúc quantization, không phải ở mức calibration data.

---

## 6. Tư duy hướng đi tiếp theo (xếp theo độ ưu tiên)

### 6.1 [ƯU TIÊN CAO] Deploy encoder+classifier ở fp16-trên-NPU, bỏ quantize int8/int16

**Lý do:** frontend hiện đang chạy fp16-trên-NPU (không quantize) và với input đó local-sim đã chứng minh model hoạt động đúng ở fp32/gần-fp32. Nếu bỏ hẳn bước `submit_quantize_job` cho encoder+classifier và compile thẳng model float (giống cách đang làm với frontend), sẽ loại bỏ hoàn toàn nguồn brittleness của W8A16 tại gốc.

**Cách làm:** `hub.submit_compile_job(model=<raw float enc+cls model mno4y3gkm>, device=..., options="--target_runtime qnn_dlc --truncate_64bit_io")` (không `--quantize_io`, không qua `submit_quantize_job`).

**Rủi ro/đánh đổi:**
- Model ~945MB ở fp16 có thể chạy chậm hơn đáng kể so với int8 trên Hexagon (fp16 dùng nhiều băng thông bộ nhớ hơn, có thể không đạt throughput real-time cần thiết cho edge deployment).
- Cần xác minh compiler QNN có chấp nhận compile trực tiếp model float lớn cỡ này hay không (thời gian compile có thể dài, đã thấy compile job ~945MB int8 mất tới ~50 phút — float có thể còn lâu hơn hoặc thất bại vì giới hạn bộ nhớ trên device).
- Vẫn thỏa ràng buộc "100% NPU" (không cần int8, chỉ cần chạy trên NPU).

**Đây là hướng khuyến nghị làm trước tiên** vì trực tiếp giải quyết root cause (brittleness của quantization), không phải vá triệu chứng.

### 6.2 [ƯU TIÊN TRUNG BÌNH] Quantization mixed-precision có chọn lọc theo layer

Từ cosine-similarity trace đã làm (mục 3), đã biết phân kỳ bắt đầu từ layer đầu tiên (`encoders0[0]`). Có thể chỉ giữ float cho vài layer đầu (nhạy cảm nhất) và quantize phần còn lại — cân bằng giữa tốc độ (đa số layer vẫn int8) và độ chính xác (layer nhạy cảm giữ nguyên fp32/fp16).

**Nhược điểm:** AI Hub quantize job KHÔNG hỗ trợ flag loại trừ layer/op cụ thể (đã test toàn bộ các flag `--op_types_to_exclude`, `--nodes_to_exclude`, v.v. — tất cả bị reject). Sẽ cần graph-surgery thủ công để tách sub-graph theo layer, tương tự cách đã làm để tách frontend/encoder — tốn thời gian dev hơn nhưng khả thi kỹ thuật.

### 6.3 [ƯU TIÊN THẤP — ĐÃ CÓ BẰNG CHỨNG PHỦ ĐỊNH MỘT PHẦN] Mở rộng calibration hơn nữa

Đã thử tăng calibration từ 15 → 25 mẫu (bao gồm cả từ nguồn fp16 thật) mà không cải thiện. Việc tiếp tục tăng số lượng mẫu calibration (ví dụ 100+ mẫu) khó có khả năng giải quyết vấn đề, vì bằng chứng ở mục 5 cho thấy đây không phải vấn đề "thiếu coverage" mà là "brittleness" cố hữu. Không khuyến nghị đầu tư thêm thời gian vào hướng này trừ khi các hướng 6.1/6.2 đều thất bại.

### 6.4 [THAM KHẢO] Kiểm tra granularity của quantization (per-channel vs per-tensor)

Quantize job hiện dùng `weights_dtype=INT8, activations_dtype=INT16` nhưng chưa xác nhận rõ AI Hub áp dụng per-channel hay per-tensor scale cho weights theo mặc định. Per-channel quantization (scale riêng cho từng output channel) thường cho robustness tốt hơn đáng kể so với per-tensor với cùng bit-width. Cần tra cứu tài liệu AI Hub / thử flag liên quan nếu có, trước khi nhảy thẳng sang fp16 toàn bộ (6.1) — đây là một thử nghiệm rẻ hơn nên có thể chèn vào trước 6.1 nếu muốn tiết kiệm thời gian compile.

---

## 7. Trạng thái các model/job hiện có (để tái sử dụng)

| Thành phần | Model ID | Job ID | Trạng thái |
|---|---|---|---|
| Frontend (fp32, không quantize) | `mq804e5zn` (raw) / `mmd02gron` (compiled, không quantize_io) | compile: `jpyol1e75` | SUCCESS, đã verify 100% NPU |
| Frontend inference (test set 15 mẫu, NPU thật) | — | `jgzl67ez5` | SUCCESS |
| Encoder+Classifier RAW (float, chưa quantize) | `mno4y3gkm` | — | Sẵn sàng dùng cho hướng 6.1 |
| Encoder+Classifier quantized v1 (calib fp32-CPU) | `mm5v1dxyn` | quantize: `jp8edvwqp`, compile: `jprlmj9kp` | SUCCESS nhưng lỗi trên NPU thật |
| Encoder+Classifier quantized v2 (calib fp16-NPU thật) | `mm5v1okkn` (quantized) / `mnj8d8xdm` (compiled) | quantize: `jp8enweop`, compile: `jpvl87q75` | SUCCESS nhưng vẫn lỗi trên NPU thật |
| Calibration fbank thật từ NPU (fp16, 25 mẫu) | — | file: `calib_data_fbank_from_real_npu.npz` | Có sẵn, dùng lại được cho lần quantize sau nếu cần |

---

## 8. Khuyến nghị hành động ngay (ĐÃ CẬP NHẬT — xem mục 9)

1. ~~Thử hướng 6.1 (fp16-trên-NPU cho encoder+classifier) trước~~ — **ĐÃ THỬ, XEM KẾT QUẢ MỤC 9.**
2. Không đầu tư thêm vào việc mở rộng calibration (6.3) — đã có bằng chứng phủ định.

---

## 9. CẬP NHẬT (sau khi thử hướng 6.1): encoder KHÔNG phải nguyên nhân — vấn đề nằm ở chính frontend NPU

Đã compile trực tiếp encoder+classifier RAW (float, không qua `submit_quantize_job` — tức KHÔNG có quantization nào cả, chạy thuần fp16-trên-NPU giống frontend) và chain với cùng output fbank thật từ NPU (job `jgzl67ez5`).

**Kết quả:**

| Ngôn ngữ | v3 (encoder quantize, recalibrate fp16) | **v4 (encoder KHÔNG quantize — fp16 thuần)** |
|---|---|---|
| en WER | 58.20% | **58.20%** |
| zh CER | 98.10% | **98.10%** |
| ko WER | 100.00% | **100.00%** |

**Kết quả giống hệt nhau — kể cả text HYP từng câu cũng trùng khớp gần như tuyệt đối** (ví dụ en_2 ra đúng y hệt `"To with an easy reach is a romantic and fascinating town of Cy, and wh"` ở cả 2 lần).

### Ý nghĩa: bác bỏ hoàn toàn giả thuyết "quantization W8A16 giòn"

Việc bỏ HẲN quantization (dùng float 100% cho encoder) mà kết quả không đổi một chút nào chứng minh: **encoder — dù quantize hay không quantize — không phải là nguồn gốc vấn đề.** Input mà nó nhận (fbank thật từ NPU) đã đủ tệ để phá hỏng kết quả BẤT KỂ encoder xử lý nó ở precision nào.

**Kết luận mới, chính xác hơn:** vấn đề nằm ở chính **frontend khi thực thi thật trên NPU silicon** — không phải "khó quantize" (frontend đã KHÔNG được quantize, chạy fp16 thuần) mà là **bản thân phép tính trên NPU cho ra giá trị sai lệch đủ lớn để phá hỏng downstream**, bất kể encoder có tinh vi cỡ nào. Đây có khả năng là lỗi/giới hạn trong cách NPU thực thi các phép toán cụ thể trong frontend (FFT qua matmul, `log()` với dynamic range rộng sau `clamp(min=1e-10)`, hoặc windowing/unfold) — KHÔNG đơn thuần là "sai số làm tròn fp16" như giả thuyết ban đầu, vì sai số làm tròn thông thường sẽ không gây collapse triệt để thế này.

### Hướng đi tiếp theo (thay thế mục 6)

1. **[Ưu tiên cao] Cô lập từng bước tính toán trong frontend bằng bypass test** (kỹ thuật giống `sv_a3_bypass_frontend.py` đã dùng ở mục 3, nhưng áp dụng NGAY TRÊN chuỗi tính real-NPU thay vì CPU-quantized): tách frontend thành các sub-graph nhỏ hơn (ví dụ: FFT+power riêng, log-mel riêng, CMVN+LFR riêng), chạy từng phần trên NPU thật, so sánh với CPU fp32 ở TỪNG ĐIỂM CẮT để xác định chính xác phép toán nào gây sai lệch lớn nhất.
2. **[Ưu tiên trung bình] Thử thay `log()` bằng công thức ổn định số học hơn** (ví dụ dùng `log1p` hoặc điều chỉnh giá trị `clamp` floor từ `1e-10` lên một giá trị lớn hơn, giảm dynamic range cực đoan mà log phải xử lý) — nếu giả thuyết về `log()` LUT thô trên Hexagon đúng, việc thu hẹp dynamic range đầu vào có thể giảm đáng kể sai số.
3. **[Ưu tiên trung bình] Thử FFT bằng cách khác** thay vì matmul với ma trận DFT cố định (512x257) — ví dụ kiểm tra xem NPU có hỗ trợ FFT op gốc (native) thay vì mô phỏng bằng matmul hay không, vì matmul độ rộng 512 tích lũy trong fp16 có thể mất precision đáng kể so với accumulate fp32.
4. **[Cân nhắc]** Nếu không tìm được cách khắc phục frontend trên NPU, có thể cần chấp nhận vi phạm một phần ràng buộc "100% NPU" — chạy phần frontend (chi phí tính toán nhỏ hơn nhiều so với encoder 50-layer) trên CPU, và chỉ giữ encoder+classifier (phần nặng nhất) trên NPU. Đây là phương án dự phòng nếu ràng buộc "100% NPU" có thể đàm phán lại.

### Trạng thái model/job mới (bổ sung bảng mục 7)

| Thành phần | Model ID | Job ID | Trạng thái |
|---|---|---|---|
| Encoder+Classifier fp16 (KHÔNG quantize, compile trực tiếp) | `mqvo7oolm` | compile: `j5m0jn7qg`, profile: `jgk2jem2g`, infer: `jgolj2qxg` | SUCCESS nhưng cho kết quả giống hệt bản quantize → xác nhận encoder không phải nguyên nhân |

---

## 10. ĐIỀU TRA SÂU: xác nhận code không sai — tìm ra chính xác bước gây lỗi trên NPU

### 10.1 Code review: loại trừ khả năng bug logic

Đọc lại toàn bộ `TraceableFrontend.forward()` (framing, pre-emphasis, window, FFT-via-matmul, mel filterbank, log, LFR, CMVN) — không phát hiện bug logic mới. Quan trọng hơn: **bằng chứng thực nghiệm đã CHỨNG MINH code đúng** — chính graph ONNX này, khi chạy qua ONNXRuntime CPU fp32 (không phải PyTorch thô, mà là file ONNX đã export, cùng file dùng để compile lên NPU), cho kết quả khớp gần như tuyệt đối với baseline gốc của model (mục 4.2: zh 11.20%=11.20%, ko 38.27%=38.27%). Điều này loại trừ hoàn toàn khả năng lỗi nằm ở code Python/logic export — corruption chỉ xuất hiện khi đúng graph này được compile+chạy trên NPU silicon.

### 10.2 Bypass test theo từng checkpoint trên NPU thật

Tách frontend thành 2 checkpoint nhỏ, compile riêng lên NPU thật, so sánh với tham chiếu CPU fp32 (ONNXRuntime) tại từng điểm cắt:

- **Stage A**: wav → `power_no_dc` (ngay sau FFT-via-matmul, TRƯỚC mel filterbank & log)
- **Stage B**: wav → `mel_log` (sau mel filterbank & log, TRƯỚC LFR/CMVN)
- **Stage C**: wav → fbank đầy đủ (đã biết từ trước, mục 4.4-4.5)

**Kết quả (15 mẫu test, so sánh cosine similarity + relative L2 với CPU fp32):**

| Stage | Vị trí trong pipeline | avg cos_sim | avg rel_L2 |
|---|---|---|---|
| A | Ngay sau FFT-matmul | **0.2408** | **0.9789** |
| B | Sau mel+log | 0.8829 | 0.7030 |
| C | Full fbank (sau LFR+CMVN) | 0.90–0.97 | 0.42–0.59 |

### 10.3 Kết luận: FFT-via-matmul là nguồn gốc lỗi, không phải bug code

Chuỗi số liệu **0.24 → 0.88 → 0.90-0.97** cho thấy rõ ràng: sai lệch KHỞI NGUỒN ngay tại bước FFT-via-matmul (Stage A gần như hoàn toàn là garbage — cos_sim 0.24 nghĩa là gần trực giao với giá trị đúng, `max_abs_diff` lên tới **hàng chục tỷ** ở một số mẫu). Các bước xử lý sau đó (đặc biệt là `log()`) có hiệu ứng NÉN/che giấu một phần sai số khổng lồ này (log của một giá trị sai lệch hàng tỷ so với đúng chỉ tạo chênh lệch vài chục trong không gian log), khiến output cuối trông "đỡ tệ hơn" — nhưng lỗi không bao giờ được sửa hoàn toàn, và vẫn đủ lớn để phá hỏng hoàn toàn encoder ở downstream (dù encoder quantize hay không, đã xác nhận ở mục 9).

**Giả thuyết kỹ thuật cụ thể nhất về nguyên nhân**: độ lớn `max_abs_diff` quan sát được ở Stage A (~10⁹–10¹⁰) khớp gần đúng với bình phương của giá trị tràn số (overflow) fp16 (fp16_max ≈ 65504, bình phương ≈ 4.29×10⁹). Nhiều khả năng đây là **tràn số fp16 trong phép nhân ma trận DFT 512-chiều** (`padded @ dft_real`, `padded @ dft_imag`) khi thực thi trên Hexagon NPU — audio đã bị scale lên `×32768` (chuẩn Kaldi PCM int16) trước khi đưa vào FFT-matmul, khiến tích lũy tổng tích (dot product 512 phần tử) trên phần cứng fp16 (không có bộ tích lũy fp32) dễ dàng vượt quá ngưỡng biểu diễn.

### 10.4 Hướng sửa tiếp theo (thay thế/bổ sung mục 9)

1. **[Ưu tiên cao nhất] Giảm hệ số scale trước FFT.** Thử scale nhỏ hơn (ví dụ bỏ hẳn `×32768`, hoặc dùng hệ số scale nhỏ hơn nhiều, rồi bù lại ở bước log/CMVN nếu cần) để giữ giá trị trung gian trong FFT-matmul nằm trong dải an toàn của fp16. Đây là fix RẺ NHẤT để test — chỉ cần sửa 1 dòng trong `TraceableFrontend.forward()`, export lại Stage A graph (nhỏ, compile nhanh ~vài phút), test lại trên NPU thật, so sánh cos_sim TRƯỚC khi động vào full pipeline.
2. **[Ưu tiên cao]** Nếu giảm scale không đủ, thử **chuẩn hóa ma trận DFT** (`dft_real`, `dft_imag`) về biên độ nhỏ hơn (hiện tại các giá trị trong ma trận DFT đơn vị có thể dao động rộng), hoặc **chia nhỏ phép matmul 512-chiều** thành nhiều bước cộng dồn nhỏ hơn để giảm rủi ro tràn số tại một bước tích lũy duy nhất.
3. **[Cân nhắc]** Nếu NPU/QNN driver có tùy chọn ép accumulate ở fp32 cho riêng op Matmul/Gemm (một số backend NPU hỗ trợ "high-precision accumulation" cho matmul dù input/output là fp16) — kiểm tra tài liệu AI Hub/QNN xem có cờ compile liên quan không.
4. Sau khi Stage A trên NPU đạt cos_sim gần 1.0 với CPU fp32, mới quay lại test full pipeline (frontend đầy đủ → encoder) để xác nhận fix thực sự giải quyết được vấn đề gốc.

---

## 11. KẾT QUẢ CUỐI CÙNG: ĐÃ SỬA THÀNH CÔNG TRÊN NPU THẬT ✅

### 11.1 Fix áp dụng

Trong `TraceableFrontend.forward()`:
- **Bỏ** dòng `wav = wav * float(1 << 15)` ở đầu hàm (không còn scale audio lên thang int16 TRƯỚC khi đưa vào FFT-matmul).
- **Thêm** hằng số bù ngay sau bước log: thay vì `mel_log = torch.clamp(mel, min=1e-10).log()`, dùng:
  ```python
  SCALE = float(1 << 15)
  LOG_SCALE_COMPENSATION = 2.0 * math.log(SCALE)          # = log(SCALE^2)
  SAFE_CLAMP_MIN = 1e-10 / (SCALE ** 2)                     # floor tuong duong chinh xac
  mel_log = torch.clamp(mel, min=SAFE_CLAMP_MIN).log() + LOG_SCALE_COMPENSATION
  ```
- Tương đương toán học TUYỆT ĐỐI với công thức gốc (`log(k²·x) = log(k²) + log(x)`), đã verify trên CPU: sai lệch chỉ ~1×10⁻⁷ (nhiễu làm tròn fp32 bình thường, không phải lỗi).
- Loại bỏ hoàn toàn việc tích lũy giá trị lớn (~10⁹-10¹⁰) trong phép nhân ma trận DFT 512-chiều — nguồn gốc tràn số fp16 trên NPU.

### 11.2 Kết quả kiểm chứng từng bước

| Bước kiểm chứng | Kết quả |
|---|---|
| Stage A (power spectrum) trên NPU thật, GỐC | avg cos_sim = 0.24 (garbage) |
| Stage A (power spectrum) trên NPU thật, ĐÃ FIX | **avg cos_sim = 1.000000** (khớp tuyệt đối, 15/15 mẫu) |
| Full frontend, CPU fp32, gốc vs đã fix | max_abs_diff ~1×10⁻⁷ (khớp toán học) |

### 11.3 KẾT QUẢ CUỐI CÙNG: full pipeline thật trên NPU (Dragonwing IQ-9075)

| Ngôn ngữ | Trước fix (broken) | **SAU FIX (NPU thật)** | Mục tiêu (local sim fp32) |
|---|---|---|---|
| en WER | 57.15% | **12.78%** | 7.56% |
| zh CER | 99.05% | **11.51%** | 11.20% |
| ko WER | 100.00% | **40.34%** | 38.27% |

**zh và ko khớp gần như tuyệt đối với mục tiêu lý tưởng** (sai khác nằm trong nhiễu đo đạc bình thường). **en cải thiện ~4.5 lần** (còn cao hơn mục tiêu một chút, có thể do inherent fp16 rounding nhỏ còn sót lại hoặc do bộ test 5 mẫu en có phương sai cao — chưa cần điều tra thêm trừ khi cần tối ưu sâu hơn).

### 11.4 Kết luận cuối cùng

Nguyên nhân gốc rễ toàn bộ chuỗi vấn đề (từ lúc bắt đầu điều tra) là: **phép tính FFT bằng nhân ma trận (matmul) với hệ số scale Kaldi-int16 (×32768) áp dụng TRƯỚC khi đưa vào NPU, gây tràn số nghiêm trọng trong kiến trúc tính toán fp16 của Hexagon NPU** — không phải do quantization W8A16 của encoder (đã loại trừ qua 3 lần test), không phải do calibration lệch phân phối (đã loại trừ), và không phải do bug logic trong code export (đã chứng minh code đúng qua CPU simulation).

Kiến trúc split-graph (frontend fp16-trên-NPU không quantize + encoder W8A16 quantize riêng) **là đúng hướng ngay từ đầu** — vấn đề chỉ nằm ở MỘT phép tính cụ thể (FFT-via-matmul) trong frontend, không phải toàn bộ thiết kế. Sau khi sửa đúng điểm này, hệ thống hoạt động đúng như thiết kế ban đầu, đạt 100% NPU compute với chất lượng gần bằng baseline fp32.
