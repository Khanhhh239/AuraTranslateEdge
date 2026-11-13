# OneVoice — web demo S2S

Server chỉ dùng thư viện chuẩn của Python, phục vụ chuỗi ASR → MT → TTS và trả từng bước dạng NDJSON để giao diện hiện tiến trình.

- **Engine thật** (mặc định khi tìm thấy package `s2s_edge`, cạnh `web_demo/` trong gói hoặc ở `../s2s_edge` trong repo): `S2SEngine` trong `engine.py`. Nó chạy đúng pipeline của hộp: VAD → ASR → lọc rác → dấu câu tiếng Việt → MT (route 3.3B/600M, tách câu, kính ngữ tiếng Hàn) → TTS. Chạy NPU trên AIBOX 8550; trên laptop dùng `S2S_BACKEND=cpu` (cần các file ONNX export).
- **Engine giả** để sửa giao diện khi không có model: `S2S_ENGINE=mock python web_demo/server.py`.

Trên hộp: `bash start.sh demo` (xem `HUONG_DAN_CHAY_BOX.md`). Laptop: `python tools/laptop_web.py --demo` rồi mở http://localhost:8000.

| API | Việc |
|---|---|
| `GET /api/health` | tên engine, các chiều dịch, thông tin model / MT (3.3B có đang chạy không) |
| `POST /api/translate?src=vi&tgt=en` | body = WAV; trả `asr` → `mt` → `tts` (audio base64) → `done` |
| `POST /api/text?src=vi&tgt=en` | body = `{"text": "..."}`; chữ đã sửa → `mt` → `tts` → `done` (nút ✎ "sửa chữ rồi dịch lại") |

Giao diện: mic (bấm, hoặc giữ Space để nói), tải file / kéo thả, nghe lại âm thanh gốc và bản dịch, ✎ sửa chữ nhận dạng rồi dịch lại, sao chép bản dịch, giao diện sáng/tối, bản điện thoại.

Test (không cần model, engine thật với đồ thị giả): `python web_demo/test_engine.py`.
