# Những gì đã đổi (13/11, bổ sung 14/11) và cách chạy

Ba thay đổi vào luồng chính: **(1) MT mặc định là NLLB-3.3B theo route**, **(2) lọc rác ASR/MT**, **(3) sửa lỗi tách câu cắt số + kính ngữ tiếng Hàn rộng hơn** (mục 6), **(4) câu dài** (mục 7). Chưa chạy trên chip nên mọi số liệu dưới đây đo trên laptop.

## 1. Đã đổi gì

| Hạng mục | Trước | Bây giờ |
|---|---|---|
| `models_qcs8550/models.json` (mặc định mọi script) | NLLB-600M mọi chiều | **NLLB-3.3B** cho vi↔zh, vi↔ko; 600M cho en→vi, vi→en (khóa `mt_route`) |
| Bộ cũ | `models.json` | giữ nguyên nội dung ở `models_qcs8550/models_600m.json` (lưới an toàn thủ công) |
| `models_nllb33.json` | bản thử | giống hệt `models.json`, giữ để các lệnh cũ không gãy |
| Chữ ASR trước MT | chỉ bỏ câu < 2 ký tự | `clean_asr`: bỏ nhãn lạc "Q. ", "H. ", "问：", "Um. " đầu câu, thán từ đứng thành câu riêng ("嗯。", "그.", cuối câu "没。") khi nhiễu, chuỗi lặp ≥ 4 lần |
| Chữ MT trước TTS | đọc nguyên | `clean_mt`: bỏ "❚ ‖", "问:" chép từ nguồn, "- "/dấu câu đầu câu, "。" trong câu Việt/Anh; chỉ toàn dấu câu thì không đọc |
| Web / `results.csv` | | thêm cột `asr_raw`, `asr_cleaned`, thẻ vấn đề "ASR đã lọc rác" |

`session0.sh`, `run_btc.sh`, `run_adv.sh`, `box_serve.sh`, `run_demo.sh`, `start_box.sh` đều đọc `models.json` (hoặc `logs/models_ok.txt`) nên tự dùng 3.3B, không cần `MODELS=...` nữa. Kiểm golden của `session0.sh` chỉ xét các đồ thị 600M/ASR/TTS; bốn đồ thị 3.3B được kiểm riêng ở bước 3 bên dưới.

Nếu 3.3B không nạp được (thiếu bộ nhớ, lỗi NPU): pipeline lùi về 600M **và báo to** (khung `!! MT DEGRADED`, dải đỏ trên web, cột `mt_model` ≠ `mt_wanted`). Chi tiết: `docs/mt33_routing.md`.

## 2. Công tắc

| Mục đích | Cách |
|---|---|
| Quay lại 600M toàn bộ | `S2S_MT=600m` hoặc `MODELS=models_qcs8550/models_600m.json` |
| Cấm lùi âm thầm (đang muốn đo 3.3B) | `S2S_MT_STRICT=1` |
| Tắt lọc rác | `S2S_ASR_CLEAN=0` |
| Tắt tách câu ở tầng pipeline (lớp MT vẫn tách) | `S2S_MT_SPLIT=0` |
| Đổi route một chiều | sửa một dòng trong `mt_route` của `models.json` |

## 3. Cách chạy trên hộp (theo thứ tự)

```bash
# laptop: đẩy gói lên hộp
python tools/laptop_push.py

# hộp, tài khoản system: kiểm 3.3B có chạy trên NPU không và chất lượng
runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt33_check.py 2>&1 | grep -a "^##" | tee /tmp/mt33_check.txt'
runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt_rubric.py 2>&1 | grep -a "^##" | tee /tmp/mt_rubric.txt'

# chạy web (mặc định đã là 3.3B)
SKIP_CHECK=1 bash tools/start_box.sh
```

Hai lệnh kiểm thoát mã 2 và in `FAIL: NLLB-3.3B IS NOT IN USE` nếu 3.3B không chạy: khi đó dừng và đọc lỗi, đừng chạy BTC/ADV.

Sau đó chạy BTC và ADV như thường (`tools/run_btc.sh`, `tools/run_adv.sh`, hoặc trên web), rồi kiểm bộ lọc:

```bash
python tools/junk_report.py <results.csv trong zip> --out junk_report.md
```

Báo cáo có 3 phần: mọi chỗ đã cắt (xem có cắt nhầm lời thật không), mảnh ngắn đầu/cuối câu không có trong transcript gốc (ứng viên rác mới), rác còn sót trong MT. Chỉ thêm luật khi có mẫu thật.

## 4. Đã kiểm gì, chưa kiểm gì

Đã kiểm (laptop):
- Test: `python tests/test_asr_clean.py`, `test_mt_clean.py`, `test_asr_clean_pipeline.py` đều qua (ca giữ/ca đổi, 20.000 chuỗi ngẫu nhiên mỗi lớp, cả đường thường lẫn stream, công tắc tắt).
- Lọc rác trên 1.256 câu ASR thật: 22 câu bị đổi, đọc từng câu thấy đều là rác. Transcript gốc không bị cắt (trừ "um" đầu câu).
- Ảnh hưởng tới bản dịch trên 21 câu đó: chrF 68,5 → 72,7, nhưng mẫu nhỏ, không có ý nghĩa thống kê (p ≈ 0,33). Phần chắc chắn là âm thanh không còn đọc "问:", "❚".
- Route 3.3B: BTC 90 câu vi→zh chrF +7,2; rubric tổng 3.3B/route hơn 600M (số trong `docs/mt33_routing.md`). 3.3B đo là bản int8 laptop.

**Chưa kiểm:**
- 3.3B w8a16 trên NPU: chất lượng, thời gian mỗi câu, bộ nhớ khi nạp cùng ASR/TTS.
- ASR w8a16 trên chip có thể sinh rác khác bản CPU, nên luật lọc có thể thiếu hoặc thừa.
- Ca cố ý đánh đổi: tự hỏi rồi tự đáp "没。" ở cuối câu bị cắt; "Hmm, ..." đầu câu bị bỏ "Hmm". Ghi ở `KNOWN_TRADEOFFS` trong `tests/test_asr_clean.py`.

## 5. Bằng chứng

`Downloads\outputOnevoice2\asr_clean\` (README, `mt_effect.csv`, `junk_report.md`) và `Downloads\outputOnevoice2\mt_rubric\`.

## 6. Tách câu, số, kính ngữ tiếng Hàn (13/11, ĐÃ ĐÍNH CHÍNH)

**Đính chính:** bản trước của mục này nói đường thường trên chip dịch cả đoạn một lần và việc tách câu cho chrF +5,3. **Điều đó sai.** Lớp MT trên chip (`Nllb.__call__` trong `mt.py`) vốn đã tách câu từ trước: chip 600M đạt chrF 58,3 trên các đoạn en→vi nhiều câu, ngang mức "tách câu" đo trên laptop. Mức +5,3 chỉ là so với cách laptop dịch cả đoạn, không phải cải thiện trên chip.

**Lỗi thật đã sửa:** bộ tách câu cũ của lớp MT cắt sau mọi dấu "." kể cả khi không có khoảng trắng phía sau:
- "Giá là 15.000 đồng." thành "15." và "000 đồng.";
- "36.5 độ" thành "36." và "5 độ";
- "7 a.m." thành "a." và "m.".
Bộ tách mệnh đề dùng khi câu quá dài cũng cắt "1,000" và "7:30".

Bây giờ:
- Dấu `. ! ? ;` ASCII chỉ là hết câu khi theo sau là khoảng trắng. Dấu `, :` chỉ là ranh giới mệnh đề khi theo sau là khoảng trắng.
- Không cắt sau "Mr.", "Dr.", "a.m.", "TP.".
- Pipeline và lớp MT dùng chung **một** hàm `split_sentences` (nằm trong `mt.py`), nên tách lại lần nữa không đổi gì. Câu cụt như "Vâng." ghép vào câu bên cạnh.

Trên BTC chỉ 1 câu có số kiểu này nên không có số đo chrF. Bằng chứng là test `tests/test_mt_long.py::test_numbers_not_split`: chạy với code cũ thì test thất bại, đúng như mô tả ở trên. `Engine.translate_text` vẫn là đường MT chung cho mọi chỗ gọi (đường thường, stream, gõ chữ trên web, `eval_btc.py`, `eval_adv.py`).

**Kính ngữ tiếng Hàn** (`ko_polite.py`): thêm "-ㄹ까 → -ㄹ까요", "어때 → 어때요", "멈춰 → 멈춰요", "(안) 돼 → 돼요", "(하지) 마 → 마세요". Đuôi "-니까/-습니까" giữ nguyên. Trên 180 câu vi→ko của BTC có 5 câu bị đổi, cả 5 đều gần đáp án hơn (chrF tổng 25,20 → 25,37). Trước đó 167/180 câu đã đúng kính ngữ.

**Đã thử, không làm:** đổi dấu câu ASCII sang toàn khổ trong tiếng Trung (",." → "，。"). chrF không đổi (41,98 → 41,98) vì phép đo đã chuẩn hóa dấu câu, nên không có bằng chứng là có lợi.

**Thời gian:** không đổi so với trước, vì lớp MT trên chip vốn đã tách câu.

**Lỗi MT còn lại** (rubric 54/66 theo route, đều là lỗi từ vựng, chưa sửa vì chưa có cách sửa chung mà không học vẹt bộ test):
- Địa danh Việt: "Đà Nẵng" → "大南" (đúng là 岘港).
- Từ lóng/thành ngữ: "nước ngọt" → "甜水", "buôn dưa lê" → "卖梨", "mới tới" (= 才到) → "新巴士".
- en→vi (600M): "boil" → "nấu" (đúng là "luộc"), "alarm" → "báo động" (đúng là "báo thức"), "minutes" (biên bản) → "hồ sơ".
- Nhiều câu điểm thấp là do ASR nghe sai ("taxi" → "tax", "bắp rang" → "bắ rang"), không phải lỗi MT.

Test: `tests/test_mt_text.py`: tách câu ở cả hai đường, zh nối không khoảng trắng, câu đơn chỉ gọi MT một lần, công tắc tắt, từ viết tắt, kính ngữ (có kiểm lọc hai lần ra cùng kết quả).

## 7. Một câu quá dài thì sao

Đồ thị MT trên chip có giới hạn cứng: đầu vào tối đa 62 token. Đầu ra dừng ở 95 token hoặc ở 2,2 × số token nguồn + 14 (chặn trường hợp lặp vô tận). Code xử lý có sẵn trong `Nllb._tr`, nay đã có test (`tests/test_mt_long.py`):

1. Câu vừa giới hạn và dịch xong (có token kết thúc): dùng luôn.
2. **Câu quá 62 token, hoặc bộ giải mã hết bước mà chưa kết thúc**: cắt theo câu, rồi theo mệnh đề ("," ":" kèm khoảng trắng), rồi chia đôi theo từ, đệ quy đến 6 tầng. Dịch từng mảnh rồi nối lại. Test kiểm không mất chữ nào, đúng thứ tự, không cắt giữa số.
3. Không cắt được nữa (một từ duy nhất): chấp nhận bản cắt cụt và **đếm** vào `mt.truncations` (có trong `bench.py`, `eval_mt.py`), không giấu.

Trên BTC, câu dài nhất sau khi tách là 38 token. Không câu nào chạm giới hạn 62, nên đường cắt ở bước 2 chỉ được kiểm bằng test giả lập, chưa gặp trên dữ liệu thật. Người nói một câu rất dài không ngắt nghỉ (ASR tiếng Việt cần vibert-capu thêm dấu câu) có thể chạm giới hạn.

**Trường hợp chưa xử lý: model tự dừng sớm, bỏ ý dù câu vừa giới hạn.** Ví dụ "Tôi xin được giới thiệu cho các bạn trước chút, vị này là giáo sư…" bị bỏ vế đầu. Đã thử hai cách và **không đưa vào luồng chính**:
- Luôn cắt câu dài (≥ 20 token) theo dấu phẩy: chrF **48,2 → 44,0**, tốt hơn 16 / kém hơn 36 trên 56 câu. Mảnh rời mất ngữ cảnh và sinh ra chữ bịa, ví dụ zh→vi thêm "tôi nhìn thấy một bức ảnh của một người đàn ông".
- Chỉ cắt khi bản dịch nguyên câu ngắn bất thường (< 0,8 lần bản cắt): chỉ bắt được 7/56 câu (5 tốt hơn / 2 kém hơn, +0,5 chrF trung bình). Quá ít để kết luận, và phải dịch hai lần.

Chi tiết: `Downloads\outputOnevoice2\mt_long\long_whole_vs_clause.csv`.

**Đo lại route cho công bằng.** Cột 3.3B trong `mt_compare` được dịch cả đoạn, còn 600M trên chip thì tách câu. Đo lại 3.3B có tách câu như pipeline, so với 600M chip (BTC, khoảng tin cậy 95 % bootstrap):

| Chiều | 600M chip | 3.3B (cũ, cả đoạn) | 3.3B (tách câu) | chênh | Route |
|---|---|---|---|---|---|
| en→vi | 56,18 | 54,87 | 56,77 | +0,6 (−1,2..+2,3) | giữ 600M: không khác biệt, 600M nhanh hơn |
| vi→en | 51,68 | 54,03 | 54,15 | +2,5 (−0,9..+6,1) | giữ 600M: chưa chắc chắn |
| vi→zh | 34,76 | 41,98 | 41,24 | **+6,5 (+2,5..+10,6)** | 3.3B |
| zh→vi | 50,57 | 53,90 | 53,98 | **+3,4 (+0,7..+6,1)** | 3.3B |
| ko→vi | 37,11 | 40,25 | 40,39 | **+3,3 (+0,4..+6,1)** | 3.3B |
| vi→ko | 26,74 | 24,85 | 24,20 | −2,5 (−6,8..+1,7) | 3.3B, vì sửa lỗi đồng âm và rubric 10/10 (xem `mt33_routing.md`) |

Route không đổi. Riêng en→vi trước đây ghi 3.3B "−1,3" là do cách đo; đo đúng thì hai model ngang nhau. Nếu 3.3B trên chip đủ nhanh, có thể thử route vi→en sang 3.3B khi có số thời gian thật. Chi tiết: `Downloads\outputOnevoice2\mt_split\route_fair.csv`.


## 8. Sửa lỗi theo REPORT DATA CHECK (14/11)

Nguồn lỗi: báo cáo kiểm dữ liệu và việc soát lại 1.256 câu BTC + ADV của `outputDataa3`. Mỗi sửa có ca **phải đổi** và ca **phải giữ nguyên** trong `tests/test_fixes_20261114.py` (25 test). Ca "phải giữ" là những câu trông giống nhưng không được đổi, nên sửa nào bắt nhầm sẽ bị test bắt ra.

| Lỗi | Ví dụ trước → sau | Sửa ở đâu | Không đổi (đã kiểm) |
|---|---|---|---|
| Đơn vị ăn chữ cái đầu của từ tiếng Việt sau số | "5 lần" → "năm lítần" → **"năm lần"**; "15 mét" → "mười lăm métét" → **"mười lăm mét"**; cũng sửa "3 gói", "2 mẫu", "4 lượt", "10 lớp", "7 mặt" | `hybrid_textnorm.py`: ranh giới sau đơn vị/tiền tệ nhận cả chữ có dấu | "20 m.", "3 kg", "1 l nước", "5 g", "3.5kg입니다" (vẫn đọc 킬로그램), "5kg的" |
| Đơn vị một chữ cái viết hoa | "5G" → "năm gam" → **"năm G"**; "5M users" → "five meters" → **"five M"** | như trên: đơn vị một chữ in hoa (trừ L) bỏ qua | "2 L" (lít), "5 g", "20 m" |
| Định dạng chưa hỗ trợ | "24/7" → **"hai mươi bốn trên bảy / twenty-four seven"**; "15-11-2026", "15.11.2026" → ngày; "tháng 3/2017" → "tháng ba năm…"; "21st" → "twenty first" (chỉ tiếng Anh); "10-20 phút" → "mười đến hai mươi phút" | các kiểu span mới `idiom`, `date`, `monthyear`, `ordinal`, `range` | "tỉ số 6-6", "3-1", "thua 3-5", "lost 3-5", số điện thoại "555-0123", "3-5" đứng một mình (vẫn báo là mơ hồ, như luật gốc của repo) |
| Khoảng số chỉ đọc "X đến Y" khi có bằng chứng | phải có từ chỉ lượng đứng sau (phút, 분, 分钟) hoặc từ gợi ý đứng trước (từ, from, pages, khoảng), và không có từ tỉ số trong 25 ký tự quanh đó | `_range_ok` | xem cột trên |
| Tiếng Hàn: số thuần Hàn trước lượng từ | "8시간" → "팔시간" → **"여덟 시간"**; "2개" → **"두 개"**; "3살" → **"세 살"**; "3시" → **"세 시"**; "3번째" → **"세 번째"** | `_KO_COUNTER`, `ko_native` (1 đến 99) | "13시" (십삼 시), "100개", "2개월", "3번 출구", "5달러", "1만 명", "6월" không thêm dấu cách |
| Mã dính chữ | "s403" → "sbốn trăm lẻ ba" → **"s bốn trăm lẻ ba"**; "bạnwork2002" không còn bị đọc như một mã | khoảng trắng quanh lời đọc; ranh giới mã nhận chữ có dấu, còn chữ Hán/Hàn được phép đứng sát mã (航班号是VN247) | |
| Tách câu sau chữ viết tắt tên | "my name is M. James" bị tách làm hai câu → **giữ một câu**; "Bamela F. Fagj" cũng vậy | `mt.py`: `_INITIAL`, trừ khi từ tiếp theo là từ mở câu thông dụng | "Plan B. We go now.", "So did I. Then…" vẫn tách |
| Dấu hai chấm do mô hình dấu câu tiếng Việt chèn vào | Chỉ sửa hai dạng gây hại. Một là nhãn ngắn kiểu người nói (≤ 2 từ đầu câu): "Thư: Có lý do gì không ạ?" → **"Thư, có lý do gì không ạ?"**, MT không còn dịch như một lời thoại. Hai là dấu hai chấm treo ở cuối: "Nó vừa vặn với tôi:." → "Nó vừa vặn với tôi." (BTC VK-8934: chrF 0 → 23,6) | `conversation.py`: `fix_vi_colons` | "3:30", "2:1"; dấu hai chấm mở danh sách ("gồm: tiếng Tây Ban Nha, tiếng Anh, …"); dấu hai chấm bình thường ("Phòng đang dọn: Khoảng 2 phút…", "Tôi hỏi thêm một câu nữa: …"), vì đổi các câu này làm MT kém đi (VE-6066 −5,3, VK-4594 −1,7, ADV VK_ADV_006); tên riêng giữ chữ hoa |
| NLLB lặp tên trong ngoặc | "도니 대학교 (도니 대학교)" → **"도니 대학교"** (TTS không còn đọc hai lần) | `asr_clean.drop_dup_parens`: chỉ bỏ ngoặc khi một từ trong ngoặc lặp lại trong 30 ký tự ngay trước. Gọi trong `clean_mt` **và sau bước phiên âm tên** (`conversation.py`), vì NLLB thường viết "도니 대학교 (Dony University)", chỉ sau khi phiên âm mới thành lặp. Lần kiểm đầu áp lên đầu ra cuối nên không thấy điều này; chạy đúng thứ tự pipeline mới phát hiện bản sửa chưa có tác dụng | "(SWAPO)", "(se)", "(đã gồm thuế)", và khi câu nguồn có ngoặc |
| 3.3B dừng giữa câu ở dấu phẩy (vi→zh, 14/1.256 câu) | dịch lại bằng 3.3B với length penalty 1.6, sau đó mới thử 600M. Chỉ nhận bản dịch nếu **đủ câu và dài hơn rõ rệt** (≥ 1,15 lần), vì một bản đủ câu "mà không dài hơn" là đã bỏ sót ý ở chỗ khác. Nếu không bản nào đạt thì giữ bản 3.3B dài hơn, không bao giờ dùng 600M | `mt.py`: `_comma_cut`, `_retry_cut` (tag `mt33_lp1.6` / `mt600_cut_fallback`) | câu đã đủ thì không dịch lại (test); router chỉ kích hoạt ở đúng 14 câu đó, không câu nào khác |

**Kiểm trên CPU, không dương tính giả** (`Documents\onevoice_cpu_tts\run1\fix_check\`):
- Chụp đầu ra của bộ chuẩn hoá, bộ tách câu, `clean_asr`, `clean_mt` trên toàn bộ 1.256 câu thật, trước và sau khi sửa (`before.json`, `after3.json`). Đọc **từng** câu bị đổi:
  - 12 câu chuẩn hoá đổi, cả 12 đều đúng hơn;
  - 3 câu tách câu đổi (đều là chữ viết tắt tên);
  - 10 câu `clean_mt` đổi (đều là tên bị lặp trong ngoặc);
  - `clean_asr` 0 câu đổi.
- Vòng soát đầu tiên tìm ra 6 chỗ bắt nhầm hoặc làm hỏng của chính bản sửa. Tất cả đã sửa và đã có test giữ lại:
  - "3.5kg입니다" đọc mất 킬로그램;
  - "tháng tháng ba";
  - "Plan B." bị gộp câu;
  - "(SWAPO)" bị xoá;
  - tiếng Hàn bị thêm dấu cách thừa;
  - mã đứng sát chữ Hán bị mất.
- Test gốc của repo cho bộ chuẩn hoá chạy với code mới: `tests/test_hybrid_format` 10/10, `s2s_edge/test_hybrid_format` + `tests_identifiers` 15/15, `tests_numbers` OK. Vòng đầu có 5 test hỏng, đã sửa code cho khớp luật gốc của repo:
  - kiểu viết "twenty first";
  - "3-5" đứng một mình vẫn báo là mơ hồ;
  - mã đứng sát chữ Hán.
- Test của gói: 7 bộ test dạng script (32 ca), 25 test mới (có test chạy qua `Engine` theo đúng thứ tự MT → phiên âm tên → lọc ngoặc) và `web_demo/test_engine` đều qua.
- Router dịch lại (code thật, mô phỏng trên laptop bằng 3.3B CTranslate2 và 600M HF) trên 14 câu bị cắt (`router_real_test.py`, `router_real_out.txt`):
  - 10 câu thành đủ câu, đọc từng câu so với câu nguồn: đều đúng ý hơn bản bị cắt (câu BTC VM-0383: chrF 43,0 → 44,0);
  - 4 câu giữ bản 3.3B. Trong đó có VM_ADV_008: bản 600M đủ câu nhưng bỏ một vế và bịa ra ngày tháng, nên bị luật "phải dài hơn" chặn lại.
- Ngưỡng 1,15 được chọn sau khi đã thấy 14 câu này (mẫu nhỏ), nên cần theo dõi tag `mt600_cut_fallback` trên hộp.

**Chưa làm:**
- (Đã làm) `outputDataa3.zip` đã chạy lại theo đúng thứ tự pipeline:
  - MT 34 câu, 27 câu đổi;
  - TTS 43 câu, 0 lỗi;
  - BTC: 4 câu tốt lên, 1 câu kém đi (EV-7818: tách câu đúng nhưng MT chọn từ khác), xem `README.md` và `rerun_changes.csv` trong zip.
- Chưa chạy trên hộp.
- Phân số (3/4, 1/2), "1990s", "COVID-19" vẫn đọc như cũ.
