# SenseVoice-Small chạy 100% trên NPU Qualcomm Dragonwing IQ-9075

ASR đa ngôn ngữ (en / zh / ko) chạy toàn bộ phần tính toán trên NPU Hexagon của **Dragonwing IQ-9075 EVK**
(thiết bị thật qua Qualcomm AI Hub). Host chỉ chuyển tensor giữa hai lần gọi NPU và giải mã UTF-8.

Thư mục này gồm code export / compile / đánh giá / demo, kết quả đo, và toàn bộ hành trình điều tra
(vì sao bản đầu hỏng, nguyên nhân gốc, cách sửa). Phần "Bài học" ở cuối đáng đọc trước khi làm việc tiếp với AI Hub.

---

## 1. Kết quả

### 1.1 Bộ test 90 câu (30 en / 30 zh / 30 ko, FLEURS `test[5:]`, không trùng bộ 5 câu cũ)

Cùng 90 câu, cùng cách chấm. So NPU thật với tham chiếu fp32 chạy CPU (ONNX Runtime):

| Chỉ số | **NPU thật** (frontend v3 + encoder W8A16) | fp32 CPU (tham chiếu) | Chênh |
|---|---|---|---|
| en WER | 8.98% | 8.45% | +0.53 |
| zh CER | 8.33% | 8.65% | −0.32 |
| ko CER | 7.33% | 7.65% | −0.32 |
| ko WER | 30.68% | 26.56% | +4.12 |

Kết luận: chất lượng trên chip thật gần như trùng fp32. Với tiếng Hàn nên dùng **CER**: WER nhạy với lỗi
khoảng trắng ("1 만년 전 에" so với "1만년 전에"), nên chênh +4 điểm ở WER nhiều khả năng chủ yếu là lỗi tách từ (CER ngang fp32 ủng hộ điều này); chưa phân tích từng câu để khẳng định.
Kết quả từng câu: [`results/eval30_npu_results.json`](results/eval30_npu_results.json),
[`results/eval30_cpu_ref_results.json`](results/eval30_cpu_ref_results.json).

Giới hạn của phép đo: 30 câu mỗi ngôn ngữ vẫn nhỏ (sai số thống kê cỡ ±1–2 điểm); toàn bộ là giọng đọc FLEURS
sạch, chưa đánh giá trên thu âm micro / nhiễu thật.

### 1.2 Tốc độ chip (đo bằng AI Hub Profiler, 100 lần chạy liên tiếp, model đã nằm sẵn trong bộ nhớ thiết bị)

| Thành phần | Thời gian NPU / clip | Profile job | Layer trên NPU |
|---|---|---|---|
| Frontend v3 | **64.2 ms** (59.7–65.1) | `j57ez489p` | 82/82 |
| Encoder + Classifier W8A16 | **269.0 ms** | `jp0mwzo2g` | 3409/3409 |
| Tổng tham chiếu | **≈ 333 ms / clip tối đa 29 s** | | 100% NPU |

Đây là tốc độ khi model đã nằm sẵn trên thiết bị (như khi deploy lên một board cố định). **Không phải** độ trễ khi gọi
qua AI Hub, xem mục 6.

### 1.3 Quá trình cải thiện (bộ 15 câu: 5 en / 5 zh / 5 ko, tất cả chạy trên NPU thật)

| # | Thay đổi | en WER | zh CER | ko WER |
|---|---|---|---|---|
| 1 | Chain 2 graph, frontend v1 (scale ×32768) | 57.15 | 99.05 | 100.00 |
| 2 | + bỏ `--quantize_io` khi compile frontend | 57.15 | 99.05 | 100.00 |
| 3 | + recalibrate encoder bằng fbank NPU (của frontend lỗi) | 58.20 | 98.10 | 100.00 |
| 4 | + encoder fp16 không quantize | 58.20 | 98.10 | 100.00 |
| 5 | Frontend v2: bỏ scale ×32768, bù trong log | 12.78 | 11.51 | 40.34 |
| 6 | Gộp frontend v2 + encoder fp16 thành 1 graph | 12.49 | 11.51 | 48.27 |
| 7 | **Frontend v3: chuẩn hóa theo từng frame** | **7.56** | **10.56** | **36.48** |
| | tham chiếu fp32 CPU | 6.23 | 11.20 | 38.27 |

Bộ 15 câu quá nhỏ để so sánh tinh giữa các dòng 5–7; số liệu đáng tin hơn là bảng 90 câu ở mục 1.1.
Các dòng 1–4 cho cùng một kết quả hỏng vì lỗi nằm ở frontend, không đổi theo thay đổi ở encoder.

---

## 2. Kiến trúc cuối: 2 graph NPU tách rời

```
wav [1,464000] f32 ─┐
wav_len [1] i32    ─┴─► [Graph 1: Frontend v3, fp16 trên NPU] ─► fbank [1,500,560], speech_lengths [1]
                                                                   │  (host chỉ chuyển tensor)
language [1], textnorm [1] ─────────────────────────────────────────┤
                                                                   ▼
                          [Graph 2: Encoder 50 lớp + CTC head + collapse + detokenize, W8A16] ─► byte_stream [1,12096] i32
                                                                   │
                                           host: bỏ byte 0, decode UTF-8 ─► text
```

* **Graph 1 – Frontend v3** (`export/sv_export_frontend_v3_perframe.py`): framing → pre-emphasis → Hamming → DFT bằng matmul →
  power → mel → log → LFR (stack 7, hop 6) → CMVN. Không quantize. Compile:
  `--target_runtime qnn_dlc --truncate_64bit_io`.
* **Graph 2 – Encoder + Classifier** (`export/step4_s1_export_sensevoice_split_npu.py`): SenseVoice-Small encoder, CTC head
  25055 lớp, CTC collapse tĩnh (cumsum + scatter vào thùng rác) và tra bảng byte UTF-8, tất cả nằm trong graph.
  Quantize `weights=INT8, activations=INT16` (AI Hub `submit_quantize_job`), calibration 25 mẫu ở miền fbank. Compile:
  `--target_runtime qnn_dlc --quantize_io --truncate_64bit_io`.
* **Vì sao tách đôi**: frontend (dải giá trị rộng) và encoder (chịu quantize tốt) cần độ chính xác khác nhau,
  mà AI Hub quantize cả graph một thể và không có cờ loại trừ layer (mục 6).
* **Shape tĩnh**: NPU cần kích thước cố định ⇒ mỗi clip tối đa **29 s** (464000 mẫu @ 16 kHz), ngắn hơn thì đệm 0
  (`wav_len` cho biết độ dài thật), dài hơn thì **bị cắt** (chưa có chia đoạn).
* Ngôn ngữ: `language` zh=3, en=4, ko=12 (auto=0); `textnorm` 14 = bật ITN.

---

## 3. Nguyên nhân gốc và cách sửa

### 3.1 Năm lỗi trong script export (đã sửa, nhưng không phải nguyên nhân chính của sự cố trên chip)

Script gốc của Lê Gia Khánh (`AuraTranslateEdge-OneVoice`), sửa trong
`export/step4_s1_export_sensevoice_e2e_unified.py`:

| # | Lỗi | Sửa |
|---|---|---|
| 1 | Không che token ngoài độ dài thật ⇒ lặp ký tự vô hạn ở cuối câu | thêm input `wav_len`, mask `token_ids` theo độ dài thật |
| 2 | FFT lấy sai bin (`real[:,1:]` bỏ DC giữ Nyquist) ⇒ lệch tần số mel | `real[:, :-1]` |
| 3 | Pre-emphasis mẫu đầu frame không nhân `(1−0.97)` | `frames_pe[:,0] = frames[:,0]*(1−0.97)` |
| 4 | Tính độ dài qua float ⇒ AI Hub quantize nhầm cả chuỗi tính độ dài (QDQ) | viết lại thuần số nguyên int64 |
| 5 | `out_lens` do chính model trả về bị quantize và bão hòa ở 180 | tự tính `valid_len = speech_lengths + 4` (đã kiểm chứng offset đúng) |

### 3.2 Lỗi thật: frontend không vừa dải số của fp16

Hexagon HTP **không có phép tính fp32**: graph "float" biên dịch lên NPU thực chất chạy fp16 (max 65504,
min bình thường 6.1e-5). Frontend nhân audio ×32768 rồi bình phương, nên có hai lỗi liên tiếp:

1. **Tràn số trên (bản v1).** `wav×32768` đưa `real`/`imag` của DFT lên cỡ 2·10⁵ > 65504; bình phương ra 10⁹–10¹⁰.
   Đo Stage A (power spectrum) trên NPU: cos_sim 0.24 so với CPU, `max_abs_diff` ~10⁹–10¹⁰.
2. **Tràn số dưới (bản v2, "sửa" lỗi 1 bằng cách bỏ scale).** Stage A khớp cos_sim 1.0 nên tưởng đã xong, nhưng giá trị
   mel nhỏ đi 10⁹ lần: mô phỏng fp16 cho thấy **51.8% giá trị mel bị làm tròn về 0**, sai số log trung bình 1.75.
   Không có hệ số scale toàn cục nào vừa: dải mel ~11 bậc (1e-9 … 95), `k ≥ 2⁵` lại tràn trên; tốt nhất `k = 2⁴` vẫn
   sai số log ~0.17–0.30. Hệ quả: fbank đầy đủ trên NPU lệch nặng (cos_sim thấp nhất 0.50, trung bình ~0.80);
   zh/ko ít bị vì nhiều năng lượng ở bin giữa, en chịu nhiều nhất.
3. **Sửa (v3): chuẩn hóa theo từng frame.** Mỗi frame chia cho biên độ đỉnh của chính nó:
   `g_t = P / max(peak_t, 1e-4)` với `P = 2`, tính toàn bộ trong miền đã chuẩn hóa, rồi bù chính xác bằng hằng số cộng sau log:
   `mel_log = log(clamp(mel, 6.2e-5)) − 2·log(g_t) + 2·log(32768)` (tương đương toán học).
   Frame im lặng tuyệt đối (peak = 0) gán `log(1e-10)` như công thức gốc. Mô phỏng fp16: sai số log trung bình **0.0034**;
   trên NPU thật fbank đạt **cos_sim trung bình 0.99999** (thấp nhất 0.99996) so với bản fp32 gốc.
   Một lần lặp đáng nhớ: ngưỡng "frame im lặng" đặt `≤ 1e-4` làm hỏng các mẫu có nhiễu nền 1 LSB (~3e-5);
   chỉ được coi là im lặng khi frame là số 0 tuyệt đối (`< 3e-8`).

### 3.3 Những giả thuyết đã thử và bác bỏ (tránh lặp lại)

| Giả thuyết | Kiểm chứng | Kết luận |
|---|---|---|
| `--quantize_io` làm hỏng fbank | fbank NPU (mẫu kiểm tra) có/không flag **giống hệt từng bit**, và WER/CER cả 15 câu trùng nhau | sai |
| Encoder calibrate lệch phân phối | recalibrate bằng fbank NPU: kết quả không đổi | sai |
| Quantize W8A16 của encoder quá giòn | encoder fp16 không quantize: WER/CER trung bình **trùng khớp** (58.20 / 98.10 / 100.00) | sai |
| Lỗi logic trong code export | cùng graph ONNX chạy ONNX Runtime CPU fp32 khớp baseline | sai |
| Lỗi nằm ở độ dài / mask | `speech_lengths` từ NPU khớp công thức chính xác 15/15 | sai |

Bài học phương pháp: lần "xác nhận" đầu chỉ kiểm tra một điểm trung gian (Stage A) nên bỏ sót lỗi tràn dưới ở
các bước sau. **Luôn so tensor cuối cùng (fbank) NPU với fp32 trên toàn bộ mẫu**, không chỉ một bước giữa.

Về calibration của encoder đang dùng (`jprlmj9kp`): fbank calibration được tạo bằng cách chạy frontend gốc
`model_sv_frontend.onnx` trên **CPU fp32** (`hub/sv_prepare_calib_fbank.py`), tức đúng phân phối fp32 chuẩn và
frontend v3 bám sát nó (cos 0.99999). Không cần quantize lại. Lần recalibrate "không hợp lệ" ở bảng 3.3 là bản dùng fbank
chạy trên NPU của frontend lỗi (job `jpvl87q75`), đã bỏ.

---

## 4. Cách tái lập

Biến môi trường: `SV_ROOT` = thư mục làm việc chứa `data/asr/` (xem bước 0) và nơi ghi `outputs/sensevoice-e2e-onnx/`
(mặc định là đường dẫn trên máy tác giả). Cần: `torch`, `funasr`, `onnx`, `onnxruntime`, `qai-hub` (đã `qai-hub configure`
với API token), `jiwer`, `soundfile`; thêm `datasets`, `librosa` cho bước tải dữ liệu.
Các **job/model ID trong script thuộc tài khoản AI Hub của tác giả**; ở tài khoản khác phải chạy lại compile và thay ID
(mục 7).

```bash
export SV_ROOT=/duong/dan/lam/viec

# 0. dữ liệu: bộ 5 câu/ngôn ngữ cũ do src/step1_asr/fetch_asr_data.py của repo AuraTranslateEdge-OneVoice tạo (data/asr);
#    bộ 90 câu mới:
python eval/fetch_asr_eval30.py                 # -> data/asr_eval30/, data/manifest_eval30.json

# 1. export Graph 2 (encoder+classifier) và frontend gốc fp32 (chỉ dùng làm tham chiếu + calibration)
python export/step4_s1_export_sensevoice_split_npu.py
python export/inline_onnx.py $SV_ROOT/outputs/sensevoice-e2e-onnx/model_sv_enc_cls_patched.onnx \
                              $SV_ROOT/outputs/sensevoice-e2e-onnx/model_sv_enc_cls_inline.onnx   # AI Hub không nhận external-data

# 2. export Graph 1 (frontend v3), ra thẳng file inline
python export/sv_export_frontend_v3_perframe.py

# 3. calibration cho encoder (miền fbank, chạy frontend fp32 trên CPU)
python hub/sv_prepare_calib.py && python hub/sv_prepare_calib_v2.py && python hub/sv_prepare_calib_fbank.py
```

Quantize + compile encoder trên AI Hub (tham số đúng như bản đã dùng; job `jp8edvwqp` → `jprlmj9kp`):

```python
import qai_hub as hub, numpy as np
dev = hub.Device("Dragonwing IQ-9075 EVK")
m = hub.upload_model("model_sv_enc_cls_inline.onnx")
c = np.load("calib_data_fbank.npz")
n = len(c["fbank"])
calib = hub.upload_dataset({
    "fbank": [c["fbank"][i].reshape(1, 500, 560) for i in range(n)],   # phải có batch dim, nếu không AI Hub báo sai shape
    "speech_lengths": [c["speech_lengths"][i].reshape(1) for i in range(n)],
    "language": [c["language"][i].reshape(1) for i in range(n)],
    "textnorm": [c["textnorm"][i].reshape(1) for i in range(n)],
})
q = hub.submit_quantize_job(model=m, calibration_data=calib,
                            weights_dtype=hub.QuantizeDtype.INT8,
                            activations_dtype=hub.QuantizeDtype.INT16)
q.wait()
cj = hub.submit_compile_job(model=q.get_target_model(), device=dev,
                            options="--target_runtime qnn_dlc --quantize_io --truncate_64bit_io")
```

> Đoạn trên được dựng lại từ tham số của job thật (`QuantizeJob jp8edvwqp`, `CompileJob jprlmj9kp`); chưa chạy lại
> nguyên văn như một script riêng. `calib_data_fbank.npz` lưu `fbank` dạng `(N,500,560)`, nên cần thêm batch dim như trên.

Compile frontend v3, chạy NPU và chain vào encoder cho 15 câu: `python hub/sv_chain_v3_frontend.py`
(sửa `jprlmj9kp` thành compile job encoder của bạn). Đánh giá 90 câu:

```bash
python eval/sv_eval30_cpu_ref.py   # tham chiếu fp32 CPU (cần model_sv_combined_single_inline.onnx, xem experiments/)
python eval/sv_eval30_npu.py       # NPU thật: 2 batch job (frontend, encoder), 1 lần "tải model" mỗi stage
```

Demo web (ghi âm nhiều đoạn → gửi 1 batch → xem text + timeline):

```bash
cd demo && uvicorn asr_demo_server:app --host 127.0.0.1 --port 8420
# mở ra ngoài bằng: cloudflared tunnel --url http://127.0.0.1:8420
```

Server dùng pipeline 2 graph (frontend v3 + encoder), tối đa 30 đoạn × 29 s mỗi lượt, có log ở `demo/asr_batch_logs/`.
Con số "NPU" hiển thị là **hằng số tham chiếu từ Profiler** (mục 1.2), không phải đo từng lần gọi. Luồng
này dùng cùng định dạng dữ liệu với `eval/sv_eval30_npu.py` đã chạy thật, nhưng chưa được kiểm thử bằng một lượt ghi âm
qua trình duyệt trong commit này.

---

## 5. Cấu trúc thư mục

```
sensevoice_npu/
├── export/        step4_s1_export_sensevoice_e2e_unified.py   script gốc (Lê Gia Khánh) + 5 bản sửa
│                  step4_s1_export_sensevoice_split_npu.py     export 2 graph (encoder+classifier là Graph 2)
│                  sv_export_frontend_v3_perframe.py           Graph 1 bản cuối (chuẩn hóa theo frame)
│                  inline_onnx.py                              đổi external-data thành 1 file (bắt buộc cho AI Hub)
├── hub/           sv_prepare_calib*.py, sv_chain_v3_frontend.py  calibration; compile+chạy frontend v3, chain encoder
├── eval/          fetch_asr_eval30.py, sv_eval30_*.py          bộ 90 câu, chấm WER/CER, NPU và CPU fp32
├── demo/          asr_demo_server.py                           web ghi âm batch + timeline + log
├── diagnostics/   các script điều tra (xem 3.2): so cosine fbank NPU/CPU, bypass frontend, quét scale fp16,
│                  mô phỏng fp16 chuẩn hóa theo frame, đo từng stage trên NPU
├── experiments/   graph gộp 1 graph (frontend v2 + encoder fp16): không dùng ở bản cuối
├── results/       eval30_*.json, ket_qua_15mau_cu_frontend_v2.csv (kết quả bộ 15 câu ở dòng 5 bảng 1.3)
└── docs/          investigation_log.md   nhật ký điều tra đầy đủ (bản lịch sử, có thể lỗi thời một phần)
```

---

## 6. Bài học về Qualcomm AI Hub (quan trọng khi làm tiếp)

* **Không có phép tính fp32 trên NPU.** Mọi graph float chạy fp16; kiểm tra dải giá trị trung gian (overflow lẫn underflow),
  đừng tin "đã khớp ở một bước giữa".
* **Độ trễ gọi qua AI Hub ≠ tốc độ chip.** Mỗi `submit_inference_job` chờ cấp thiết bị dùng chung rồi **tải model lên thiết bị
  (cold load)**; Profiler đo `first_load_time` encoder tới ~637 s, còn inference thuần chỉ 269 ms. Trạng thái
  `RUNNING_INFERENCE` gộp cả tải model lẫn tính toán, API không tách được, cũng không cho thời gian riêng từng mẫu trong batch.
  Thời gian chip thuần chỉ lấy được từ `submit_profile_job` (đọc `execution_summary.all_inference_times`).
* **Gộp nhiều mẫu vào 1 dataset / 1 job** để chỉ trả phí tải model một lần (bộ 90 câu chạy 2 job tổng cộng).
  Với ngữ cảnh demo/serving thật, dùng board Dragonwing IQ-9075 EVK (~$999) giữ model "ấm" sẽ bỏ hẳn phần này.
* `qai_hub.upload_model` **từ chối ONNX external-data** (".onnx.data ... is not regular file"): luôn lưu inline
  (`export/inline_onnx.py`), được khi model < 2 GB.
* `submit_quantize_job` chỉ nhận `weights_dtype`/`activations_dtype` (enum `QuantizeDtype`) và `calibration_data=`; option
  chỉ có `--range_scheme min_max`, **không có cờ loại trừ op/node** ⇒ không làm được mixed-precision trong một graph, phải tách graph.
* `job.get_status()` đôi khi trễ hàng giờ so với thực tế; dùng `job.wait()` và các lời gọi tải kết quả làm nguồn đúng.
* Windows: `job.wait()` in ký tự đồng hồ cát làm sập console cp1252 ⇒ đặt `PYTHONIOENCODING=utf-8`.
* Tunnel miễn phí của cloudflared cắt request ở ~100 s ⇒ demo dùng mô hình POST trả `job_id` rồi trình duyệt polling.
* Compile model ~946 MB mất từ vài phút tới ~50 phút tùy tải server.

---

## 7. Job / model ID (tài khoản AI Hub của tác giả, chỉ để tra cứu)

| Thành phần | ID |
|---|---|
| Encoder: model gốc → quantize → compile | `mno4y3gkm` → `jp8edvwqp` → `jprlmj9kp` |
| Frontend v3: compile → target → profile | `j57ez99vp` → `mq26z3d0n` → `j57ez489p` |
| Encoder profile | `jp0mwzo2g` |
| Eval 90 câu: frontend batch / encoder batch | `jpyoexm85` / `jglynxxj5` |
| Frontend v3, chain 15 câu: inference / encoder inference | `jp4yq338p` / `jp8eo11kp` |

---

## 8. Giới hạn đã biết và việc còn lại

* Tối đa 29 s mỗi clip, dài hơn bị cắt; chưa có chia đoạn / VAD.
* Chưa đánh giá trên tiếng nói thu qua micro / môi trường ồn; bộ test 30 câu/ngôn ngữ vẫn nhỏ.
* en vẫn cao hơn fp32 một chút (8.98% so với 8.45%): chưa rõ có còn nhiễu fp16 dư ở encoder hay chỉ là biến thiên mẫu.
* Calibration encoder chỉ 25 mẫu (5 thật + 20 tổng hợp độ dài khác nhau); mở rộng có thể cải thiện thêm.
* Graph gộp 1 graph (giảm một nửa số lần cold load) chưa được cập nhật sang frontend v3; đo ở bộ 15 câu cho thấy bản
  gộp dùng frontend v2 + encoder fp16 kém hơn ở tiếng Hàn (48.27% so với 40.34%), nhưng đó là mẫu nhỏ và khác frontend.
* Độ trễ khi gọi qua AI Hub không dùng được cho real-time; cần board thật để đạt ~0.33 s / clip.

---

## 9. Ghi công

Script export gốc: Lê Gia Khánh (`AuraTranslateEdge-OneVoice`). Mô hình: SenseVoice-Small (FunAudioLLM / FunASR).
Dữ liệu đánh giá: Google FLEURS. Hạ tầng: Qualcomm AI Hub.
