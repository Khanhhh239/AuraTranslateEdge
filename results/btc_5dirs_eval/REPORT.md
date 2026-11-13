# BTC 5-Direction End-to-End Evaluation Report (`source_mode=0`)

- **Mode:** `sampled` (seed=`42`, wall_total=`426.067s`)
- **Coverage / Timing Note:** 5-clip smoke took 17.98s incl. model load (~3.60s/clip -> projected 410 clips ~15-24 min > 8 min threshold); running 100% of digit/entity clips + balanced seed=42 sample of 30 clips/dir (150/410 total).
- **Runtime profile:** `source_mode=0` (`S2S_NEURAL_TN=0`), `tts_format=hybrid`, `vad_profile=legacy`, `ort_policy=auto`, `backend=cpu`
- **Code fingerprint:** `git_head=ecf4e945a95e16d553edbe7913cd49245980931a`, `s2s_edge_sha256=d9d35ec66ccc1964634984f4c216d67925defb2177f5e1ddfebd96314e636457`

## 1. Per-Direction Coverage, Latency, and Entity Preservation

| Dir | Selected / Total | Entity Clips | OK / Err | Mean Total (s) | P50 / P95 (s) | Mean ASR / MT / TTS (s) | TargetAudit (pres/mis/unv/tot) | ASR vs Gold (c/w/u) | MT vs Ref (c/w/u) | TTS Chunks (plan/att/syn) |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `en-vi` | 30 / 50 | 8 / 8 | 30 / 0 | 3.1668 | 2.8789 / 5.5059 | 0.6013 / 1.4625 / 1.1019 | 10/1/0/11 | 0/0/0 | 0/0/0 | 36/36/36 |
| `ko-vi` | 30 / 90 | 3 / 3 | 30 / 0 | 2.0989 | 2.063 / 2.5495 | 0.5122 / 0.6564 / 0.93 | 1/1/0/2 | 1/0/0 | 2/2/0 | 30/30/30 |
| `vi-ko` | 30 / 90 | 3 / 3 | 30 / 0 | 2.8669 | 2.6708 / 3.7791 | 0.1769 / 0.6163 / 2.0538 | 0/0/0/0 | 0/0/4 | 0/1/0 | 30/30/30 |
| `vi-zh` | 30 / 90 | 7 / 7 | 30 / 0 | 3.5641 | 3.1242 / 6.4066 | 0.1794 / 0.9147 / 2.4491 | 0/0/0/0 | 0/0/4 | 1/1/4 | 34/34/34 |
| `zh-vi` | 30 / 90 | 6 / 6 | 30 / 0 | 2.2344 | 2.1161 / 2.9991 | 0.5187 / 0.7746 / 0.9409 | 4/2/0/6 | 6/0/0 | 1/3/0 | 31/31/31 |

## 2. Sample Examples per Direction (ASR / MT / tts_text / WAV)

### `en-vi`
- **`EV-0157`** (`total=1.5261s`, `in=9.216s`, `out=4.0287s`, `target_audit={'total': 2, 'preserved': 2, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: play only all music released between the year one thousand nine hundred and ninety and two thousand
  - `asr_raw`: Play only all music released between the year 1990 and 2000.
  - `asr`: Play only all music released between the year 1990 and 2000.
  - `mt`: Chỉ chơi tất cả nhạc phát hành từ năm 1990 đến năm 2000.
  - `tts_text`: chỉ chơi tất cả nhạc phát hành từ năm một nghìn chín trăm chín mươi đến năm hai nghìn.
  - `wav`: `results/btc_5dirs_eval/wavs/en-vi/EV-0157_out.wav`
- **`EV-0358`** (`total=1.5439s`, `in=5.72s`, `out=3.9126s`, `target_audit={'total': 2, 'preserved': 2, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: seven hundred thirty four main street harper valley oregon six seven two nine seven
  - `asr_raw`: 734 Main Street, Harper Valley Oregon,67297.
  - `asr`: 734 Main Street, Harper Valley Oregon,67297.
  - `mt`: 734 Main Street, Harper Valley Oregon, 67297.
  - `tts_text`: bảy trăm ba mươi bốn main street, harper valley oregon, sáu bảy hai chín bảy.
  - `wav`: `results/btc_5dirs_eval/wavs/en-vi/EV-0358_out.wav`
- **`EV-0491`** (`total=3.3593s`, `in=9.2563s`, `out=6.252s`, `target_audit={'total': 1, 'preserved': 1, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: you'll just need roughly ten dollars for food and gas money my friend is driving so we usually all chip in a few dollars for gas money
  - `asr_raw`: You'll just need roughly $10 for food and gas money. my friend is driving so we usually all chip in a few dollars for gas money.
  - `asr`: You'll just need roughly $10 for food and gas money. my friend is driving so we usually all chip in a few dollars for gas money.
  - `mt`: Bạn chỉ cần khoảng 10 đô la cho tiền lương thực và xăng. Bạn tôi đang lái xe nên chúng tôi thường trả vài đô la cho tiền xăng.
  - `tts_text`: bạn chỉ cần khoảng mười đô la cho tiền lương thực và xăng. bạn tôi đang lái xe nên chúng tôi thường trả vài đô la cho tiền xăng.
  - `wav`: `results/btc_5dirs_eval/wavs/en-vi/EV-0491_out.wav`
- **`EV-1508`** (`total=6.256s`, `in=8.72s`, `out=5.3638s`, `target_audit={'total': 2, 'preserved': 2, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: i'll uh look at it like when my usual working hours are like from twelve o clock till three o clock or something
  - `asr_raw`: I'll look at it like when my usual working hours are like from 12 o'clock to 3 o'clock or something.
  - `asr`: I'll look at it like when my usual working hours are like from 12 o'clock to 3 o'clock or something.
  - `mt`: Tôi sẽ nhìn vào nó như khi giờ làm việc bình thường của tôi là từ 12 giờ đến 3 giờ hoặc cái gì đó.
  - `tts_text`: tôi sẽ nhìn vào nó như khi giờ làm việc bình thường của tôi là từ mười hai giờ đến ba giờ hoặc cái gì đó.
  - `wav`: `results/btc_5dirs_eval/wavs/en-vi/EV-1508_out.wav`

### `ko-vi`
- **`VK-0419`** (`total=1.7686s`, `in=1.92s`, `out=0.418s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: 골라 주세요.
  - `asr_raw`: 골라 주세요.
  - `asr`: 골라 주세요.
  - `mt`: Hãy chọn.
  - `tts_text`: hãy chọn.
  - `wav`: `results/btc_5dirs_eval/wavs/ko-vi/VK-0419_out.wav`
- **`VK-1348`** (`total=1.9076s`, `in=2.304s`, `out=1.1378s`, `target_audit={'total': 1, 'preserved': 0, 'mismatch': 1, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 1, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 2, 'all_labeled_correct': False}`)
  - `gold_text`: 15,000원이에요.
  - `asr_raw`: 15000원이에요.
  - `asr`: 15000원이에요.
  - `mt`: Đó là 15.000 người.
  - `tts_text`: đó là mười lăm nghìn người.
  - `wav`: `results/btc_5dirs_eval/wavs/ko-vi/VK-1348_out.wav`
- **`VK-1939`** (`total=2.063s`, `in=2.88s`, `out=1.1378s`, `target_audit={'total': 1, 'preserved': 1, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`, `mt_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`)
  - `gold_text`: 네, 80달러입니다.
  - `asr_raw`: 네, 80 달러 입니다.
  - `asr`: 네, 80 달러 입니다.
  - `mt`: Có, 80 đô la.
  - `tts_text`: có, tám mươi đô la.
  - `wav`: `results/btc_5dirs_eval/wavs/ko-vi/VK-1939_out.wav`
- **`VK-0143`** (`total=2.0453s`, `in=2.28s`, `out=1.5906s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: 남향 방 없나요?
  - `asr_raw`: 남향방 없나요?
  - `asr`: 남향방 없나요?
  - `mt`: Không có đường nào về phía Nam không?
  - `tts_text`: không có đường nào về phía nam không?
  - `wav`: `results/btc_5dirs_eval/wavs/ko-vi/VK-0143_out.wav`

### `vi-ko`
- **`VK-0419`** (`total=4.4076s`, `in=2.46s`, `out=1.8808s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: Chọn giúp tôi, IEC 0.
  - `asr_raw`: chọn giúp tôi ierc không
  - `asr`: Chọn giúp tôi IERC không?
  - `mt`: 내 IERC를 선택해?
  - `tts_text`: 내 IERC를 선택해?
  - `wav`: `results/btc_5dirs_eval/wavs/vi-ko/VK-0419_out.wav`
- **`VK-1348`** (`total=2.6708s`, `in=2.736s`, `out=1.8808s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 2, 'coverage': 1.0, 'labeled_expected': 2, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: 15.000 won 15.000
  - `asr_raw`: mười ngàn quân mười ngàn
  - `asr`: Mười ngàn quân, Mười ngàn.
  - `mt`: 10천명, 10만명.
  - `tts_text`: 십천명, 십만명.
  - `wav`: `results/btc_5dirs_eval/wavs/vi-ko/VK-1348_out.wav`
- **`VK-1939`** (`total=2.8963s`, `in=4.3947s`, `out=3.3437s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: Vâng, tiền trọ và tiền thế cộng lại là 80 USD ạ.
  - `asr_raw`: vâng tiền trọ và tiền thế cộng lại là tám usd ạ
  - `asr`: Vâng, tiền trọ và tiền thế cộng lại là tám USD ạ.
  - `mt`: 예, 예금과 예금 합하면 8달러입니다.
  - `tts_text`: 예, 예금과 예금 합하면 팔 달러입니다.
  - `wav`: `results/btc_5dirs_eval/wavs/vi-ko/VK-1939_out.wav`
- **`VK-0143`** (`total=3.6293s`, `in=3.776s`, `out=2.0201s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: Không có phòng khách hướng Nam à?
  - `asr_raw`: không có phòng khách hướng nam à
  - `asr`: Không có phòng khách hướng Nam à?
  - `mt`: 남쪽으로 방이 없나요?
  - `tts_text`: 남쪽으로 방이 없나요?
  - `wav`: `results/btc_5dirs_eval/wavs/vi-ko/VK-0143_out.wav`

### `vi-zh`
- **`VM-0484`** (`total=3.9604s`, `in=2.8561s`, `out=2.4845s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: Vậy 101 nhân dân tệ nhé, được không?
  - `asr_raw`: vậy một trăm lẻ một nhân dân tệ nhé được không
  - `asr`: Vậy một trăm lẻ một nhân dân tệ nhé. Được không?
  - `mt`: 那么,一百个人民币.你好吗?
  - `tts_text`: 那么,一百个人民币.你好吗?
  - `wav`: `results/btc_5dirs_eval/wavs/vi-zh/VM-0484_out.wav`
- **`VM-0925`** (`total=3.1006s`, `in=3.3494s`, `out=2.3917s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`, `mt_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`)
  - `gold_text`: Tối qua 12:00 tôi mới đi ngủ.
  - `asr_raw`: tối qua mười hai giờ tôi mới đi ngủ
  - `asr`: Tối qua mười hai giờ tôi mới đi ngủ.
  - `mt`: 晚上12点,我才上床了.
  - `tts_text`: 晚上十二点,我才上床了.
  - `wav`: `results/btc_5dirs_eval/wavs/vi-zh/VM-0925_out.wav`
- **`VM-1043`** (`total=3.5222s`, `in=3.1463s`, `out=2.9257s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 1, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: Rất đơn giản, bạn ngồi taxi khoảng 5 phút là tới.
  - `asr_raw`: rất đơn giản bạn ngồi taxi khoảng năm phút là tới
  - `asr`: Rất đơn giản, bạn ngồi taxi khoảng năm phút là tới.
  - `mt`: 简单地说,你坐出租车五分钟左右.
  - `tts_text`: 简单地说,你坐出租车五分钟左右.
  - `wav`: `results/btc_5dirs_eval/wavs/vi-zh/VM-1043_out.wav`
- **`VM-1475`** (`total=6.4066s`, `in=8.7894s`, `out=6.8325s`, `target_audit={'total': 0, 'preserved': 0, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`, `mt_ent={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: Buổi sáng sớm hàng ngày bẩy giờ hơn tôi mới thức dậy, vì vậy thường xuyên không có thời gian ăn sáng, uống một cốc sữa là đi học luôn.
  - `asr_raw`: buổi sáng sớm hàng ngày bảy giờ hơn tôi mới thức dậy vì vậy thường xuyên không có thời gian ăn sáng uống một cốc sữa là đi học luôn
  - `asr`: Buổi sáng sớm hàng ngày, bảy giờ hơn tôi mới thức dậy, vì vậy thường xuyên không có thời gian ăn sáng, uống một cốc sữa là đi học luôn.
  - `mt`: 每天早上,比我早起7个小时,所以经常没有早餐,喝一杯牛奶,就像去上学一样.
  - `tts_text`: 每天早上,比我早起七个小时,所以经常没有早餐,喝一杯牛奶,就像去上学一样.
  - `wav`: `results/btc_5dirs_eval/wavs/vi-zh/VM-1475_out.wav`

### `zh-vi`
- **`VM-0484`** (`total=1.9782s`, `in=2.76s`, `out=2.1595s`, `target_audit={'total': 1, 'preserved': 0, 'mismatch': 1, 'unverifiable': 0}`, `asr_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`, `mt_ent={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: 那101块钱吧，行吗？
  - `asr_raw`: 那101块钱吧，行吗？
  - `asr`: 那101块钱吧，行吗？
  - `mt`: Đó là 101 đô la, phải không?
  - `tts_text`: đó là một trăm lẻ một đô la, phải không?
  - `wav`: `results/btc_5dirs_eval/wavs/zh-vi/VM-0484_out.wav`
- **`VM-0925`** (`total=2.0919s`, `in=2.688s`, `out=2.2756s`, `target_audit={'total': 1, 'preserved': 1, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`, `mt_ent={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}`)
  - `gold_text`: 昨晚12点我才睡觉。
  - `asr_raw`: 昨晚12点我才睡觉。
  - `asr`: 昨晚12点我才睡觉。
  - `mt`: Tôi chỉ ngủ vào lúc 12 giờ tối hôm qua.
  - `tts_text`: tôi chỉ ngủ vào lúc mười hai giờ tối hôm qua.
  - `wav`: `results/btc_5dirs_eval/wavs/zh-vi/VM-0925_out.wav`
- **`VM-1043`** (`total=2.2725s`, `in=4.368s`, `out=2.8444s`, `target_audit={'total': 1, 'preserved': 1, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`, `mt_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`)
  - `gold_text`: 很简单，你坐出租车大概5分钟就到了。
  - `asr_raw`: 很简单，你坐出租车大概5分钟就到了。
  - `asr`: 很简单，你坐出租车大概5分钟就到了。
  - `mt`: Thật đơn giản, bạn có thể đi xe taxi trong khoảng 5 phút.
  - `tts_text`: thật đơn giản, bạn có thể đi xe taxi trong khoảng năm phút.
  - `wav`: `results/btc_5dirs_eval/wavs/zh-vi/VM-1043_out.wav`
- **`VM-1475`** (`total=4.1152s`, `in=7.536s`, `out=6.3333s`, `target_audit={'total': 1, 'preserved': 1, 'mismatch': 0, 'unverifiable': 0}`, `asr_ent={'correct': 1, 'wrong': 0, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': True}`, `mt_ent={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}`)
  - `gold_text`: 每天早晨，7点多我才起床，所以经常没时间吃早饭，喝一杯牛奶就去上课了。
  - `asr_raw`: 每天早晨7点多我才起床，所以经常没时间吃早饭，喝一杯牛奶就去上课了。
  - `asr`: 每天早晨7点多我才起床，所以经常没时间吃早饭，喝一杯牛奶就去上课了。
  - `mt`: Mỗi ngày tôi thức dậy sau 7 giờ sáng, vì vậy tôi thường không có thời gian ăn sáng và uống một ly sữa để đi học.
  - `tts_text`: mỗi ngày tôi thức dậy sau bảy giờ sáng, vì vậy tôi thường không có thời gian ăn sáng và uống một ly sữa để đi học.
  - `wav`: `results/btc_5dirs_eval/wavs/zh-vi/VM-1475_out.wav`

## 3. Entity Mismatches / Warnings Observed

- `en-vi` / `EV-5301`: `target_audit.mismatch=[{'index': 0, 'kind': 'number', 'source_raw': '800', 'target_raw': '08:00', 'status': 'mismatch', 'reason': 'missing_or_mismatched_target_entity'}]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}` | `asr=Just to confirm, you set the alarm to go off at 800.` -> `mt=Chỉ để xác nhận, anh đã đặt báo động lên 08:00.`
- `ko-vi` / `VK-0419`: `target_audit.mismatch=[]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}` | `asr=골라 주세요.` -> `mt=Hãy chọn.`
- `ko-vi` / `VK-1348`: `target_audit.mismatch=[{'index': 0, 'kind': 'money', 'source_raw': '15000원', 'target_raw': '15.000', 'status': 'mismatch', 'reason': 'missing_or_mismatched_target_entity'}]`, `mt_vs_gold_ref={'correct': 1, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 2, 'all_labeled_correct': False}` | `asr=15000원이에요.` -> `mt=Đó là 15.000 người.`
- `vi-ko` / `VK-1939`: `target_audit.mismatch=[]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}` | `asr=Vâng, tiền trọ và tiền thế cộng lại là tám USD ạ.` -> `mt=예, 예금과 예금 합하면 8달러입니다.`
- `vi-zh` / `VM-1475`: `target_audit.mismatch=[]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}` | `asr=Buổi sáng sớm hàng ngày, bảy giờ hơn tôi mới thức dậy, vì vậy thường xuyên không có thời gian ăn sáng, uống một cốc sữa là đi học luôn.` -> `mt=每天早上,比我早起7个小时,所以经常没有早餐,喝一杯牛奶,就像去上学一样.`
- `zh-vi` / `VM-0484`: `target_audit.mismatch=[{'index': 0, 'kind': 'number', 'source_raw': '101', 'target_raw': '101 đô la', 'status': 'mismatch', 'reason': 'missing_or_mismatched_target_entity'}]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}` | `asr=那101块钱吧，行吗？` -> `mt=Đó là 101 đô la, phải không?`
- `zh-vi` / `VM-0925`: `target_audit.mismatch=[]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}` | `asr=昨晚12点我才睡觉。` -> `mt=Tôi chỉ ngủ vào lúc 12 giờ tối hôm qua.`
- `zh-vi` / `VM-4870`: `target_audit.mismatch=[{'index': 0, 'kind': 'number', 'source_raw': '7', 'target_raw': '7h30', 'status': 'mismatch', 'reason': 'missing_or_mismatched_target_entity'}]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 0, 'unknown': 0, 'coverage': 0.0, 'labeled_expected': 0, 'all_labeled_correct': False}` | `asr=明天是星期天，坐车的人肯定很多，我们早点走吧，7点半怎么样？` -> `mt=Ngày mai là Chủ nhật, chắc chắn sẽ có rất nhiều người đi xe, chúng ta đi sớm hơn, 7h30 thì sao?`
- `zh-vi` / `VM-5101`: `target_audit.mismatch=[]`, `mt_vs_gold_ref={'correct': 0, 'wrong': 1, 'unknown': 0, 'coverage': 1.0, 'labeled_expected': 1, 'all_labeled_correct': False}` | `asr=我们现在学过的词汇大概有6000多个。` -> `mt=Có khoảng 6.000 từ ngữ mà chúng ta đã học được.`

Generated WAV directory: `results/btc_5dirs_eval/wavs`
