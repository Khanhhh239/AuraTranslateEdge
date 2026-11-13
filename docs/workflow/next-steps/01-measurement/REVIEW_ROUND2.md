# Review vòng 2 — Plan 01

Ngày: 2026-11-13. Commit được review: `ecf4e94`.

Phạm vi: đọc source, test, báo cáo và evidence đã lưu; hai agent đối chiếu runtime và hợp đồng đánh giá. Không chạy lại test/inference, không sửa runtime. Kết quả test dưới đây là kết quả trong log của executor.

## Finding còn lại

### P2 — Launcher có thể nhận nhầm server sau khi spawn

`tools/start_s2s_web.py:224–228` chỉ cần một health response khác `None` để trả thành công, trước khi kiểm `proc.poll()`. Không xác minh PID, effective profile hay fingerprint tại bước này, dù bước reuse đã xác minh chúng.

Nếu hai launcher đồng thời khởi chạy cùng một cổng, launcher không bind được vẫn có thể thấy health của launcher kia và báo `started` với PID/profile của process mình. Đây là một cạnh tranh chấp lúc khởi động; không phải lỗi reuse thông thường của R2 còn nguyên.

Đề nghị executor: kiểm child còn sống và health thuộc đúng `proc.pid`, khớp profile/fingerprint trước khi trả thành công; bổ sung test mô phỏng health của process khác xuất hiện sau lần probe đầu. Chưa có test tái hiện được chạy trong vòng review này.

## R1–R5 sau sửa

- R1: đóng trong source. Registry truyền TTS mode vào từng backend; TTS/source mode được giữ theo instance thay vì đọc lại env trong lúc xử lý.
- R2: đóng lỗi cũ ở luồng reuse. Validate env trước health; so sánh profile và fingerprint; từ chối mismatch; reload yêu cầu server rảnh. Còn P2 khởi động ở trên.
- R3: đóng lỗi cũ. Run directory được bảo vệ khỏi ghi đè; manifest, baseline, cấu hình, thứ tự chạy và drift được ghi/kiểm.
- R4: đóng lỗi cũ. Baseline không hợp lệ làm comparative run exit 2 trước khi tạo engine. Standalone được ghi rõ là không so sánh.
- R5: đóng lỗi substring cũ. Scorer match theo loại, giá trị và số lần xuất hiện; có unknown/coverage. Lexical proxy được ghi đúng vai trò. Chưa xem scorer là bộ hiểu ngữ nghĩa đầy đủ cho mọi cách viết/đọc số.

## Bằng chứng đã đọc

- `results/next_steps/logs/r5_final_unit_tests.log`: 49 tests, OK, 29.681 giây.
- `results/next_steps/plan01_r3_smoke_vad/summary.json`: comparative, baseline 4/4, không drift, failure_count=0. Smoke này chỉ có một repeat; không dùng riêng nó để kết luận latency.
- Typed smoke trong báo cáo: legacy 9 đúng / 0 sai / 6 unknown; upstream 7 đúng / 2 sai / 6 unknown. Không tính unknown là đạt.
- Báo cáo ghi nhận HTTP hai chiều với source mode 0/1 và batch sáu chiều cho hai profile được chọn. Review này không chạy lại hay nghe xác nhận WAV.

## Đánh giá độ hoàn thiện

**Plan 01: đã có nền tảng đo và profile đủ để chuyển sang Plan 02, kèm một P2 cần xử lý trước khi coi launcher hoàn tất.** Không còn thấy blocker P1 cũ trong các đường code đã review. Chưa gọi toàn bộ hệ thống hoàn thiện.

**Chất lượng S2S: vẫn chưa nghiệm thu.** Báo cáo còn lỗi giờ `330 pm`, tiền `$15.50`, nhiệt độ `37.5`, điện thoại bị ASR chèn dấu phẩy và ASR mất `32`. Đây là việc của Plan 02/03; không quy tất cả cho TTS vì lỗi có thể xuất hiện ở ASR, source normalization hoặc MT trước đó.

**Độ bao phủ chất lượng: còn thiếu.** Chưa có native holdout VI/KO/ZH, chưa nghe/MOS hoặc đo ASR roundtrip đầu ra; batch sáu chiều chưa chạy source mode 1. Không suy từ 49 unit tests sang chất lượng giọng nói đã đạt.

## Thứ tự tiếp theo

1. Sửa P2 launcher và test tranh chấp khởi động; cập nhật commit tham chiếu trong execution report (hiện phần đầu còn ghi HEAD cũ `43ea359`).
2. Chuyển Plan 02: chuẩn hóa theo loại thực thể, kiểm chuỗi ASR → source → MT → TTS để xác định nơi mất thông tin; giữ nguyên quy tắc đọc từng ký tự cho điện thoại/ZIP/mã.
3. Plan 03: bảo toàn số/ngữ nghĩa qua source normalization và MT. Mở rộng nhãn và dữ liệu người thật; sau đó kiểm nghe riêng, không thay bằng unit tests hoặc lexical proxy.

Không đề nghị đổi ORT default toàn cục chỉ dựa trên phép đo của một máy. Giữ VAD legacy là quyết định containment phù hợp với evidence hiện có.
