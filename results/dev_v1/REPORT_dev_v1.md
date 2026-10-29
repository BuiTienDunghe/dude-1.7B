# Phần 5a: benchmark lõi mở (test 1.0, đã đóng băng) và kết quả trên dev_v1

Ngày 2026-10-13.

- Bảng đầy đủ (mọi lát, Brier, mức tụt) nằm ở `results/dev_v1/tables_dev_v1.md`; số liệu thô ở `results/dev_v1/metrics_dev_v1.json`.
- Dữ liệu và manifest ở `data/bench_v1/`.
- **Không model nào được chạy trên tập test.** Lệnh "chưa train lại gì" vẫn giữ; lần này không train gì.

## 1. Tập test bench_v1, bản 1.0 (đã đóng băng)

- **Dấu đóng băng:** SHA-256 của `test_v1.0.jsonl` là `d92d31f696a37cdf7b5d6f95bd1ab899c878e0b4a042a8c7ae691ae8ab4b0d47`. Mã này được ghi trong `manifest_test_v1.0.json` và `FROZEN_test_v1.0.sha256` (đóng băng lúc 2026-10-12T16:31Z). File test và manifest đã được đặt chỉ đọc. Trong lúc sửa lỗi script dựng, file test được ghi lại vài lần, tất cả **trước** thời điểm đóng băng, và chưa model nào đọc nó.
- **Ghim:** seed 20261012. Revision và license của từng nguồn nằm trong manifest; ViHSD ghim ở `88e81b36…`.

| Nguồn | Câu (en / vi / vi không dấu) | Loại câu hỏi | License |
|---|---|---|---|
| Belebele (244 đoạn để dành) | 272 / 272 / 272 | choice 4 | cc-by-sa-4.0 |
| SIB-200 (split test) | **120** / 120 / 120 | choice 7 | cc-by-sa-4.0 |
| MASSIVE (split test) | 300 / 300 / 300 | choice 10 | cc-by-4.0 |
| XCOPA test (vi) + COPA test (en, ghép theo idx, nhãn lấy từ XCOPA) | 300 / 300 / 300 | choice 2 | cc-by-4.0; COPA: BSD 2-Clause |
| ViFactCheck (test) | — / 300 / 300 | khoảng một nửa choice 3, một nửa noul | MIT |
| ViWikiFC (test) | — / 300 / 300 | khoảng một nửa choice 3, một nửa noul | CC BY 4.0 |
| ViHSD (test) | — / 300 / 300 | khoảng một nửa score 3 mức, một nửa noul | MIT (bộ có cổng truy cập; người làm dự án đã tự xin quyền) |

**Thiếu so với quy mô trong brief.** Tất cả đều là hệ quả của quy tắc "trùng test–dev bằng 0", với phần song song của dev_v1 phải giữ nguyên dev_v0:
- **SIB-200 test chỉ còn 120/204 câu.** 84 câu SIB test (văn bản FLORES) nằm nguyên văn trong các đoạn Belebele của dev_v0.
- **Belebele test chỉ còn 272/300 câu.** 183 câu ứng viên bị loại vì đoạn văn của chúng chứa các câu SIB (cũng là FLORES) đã có trong dev_v0.
- MASSIVE loại 73 câu ứng viên có câu lệnh trùng dev_v0, vẫn đủ 300. XCOPA loại 6 câu, ViHSD loại 3 câu.

Muốn đủ quy mô thì phải chấp nhận trùng với dev_v0, hoặc dựng lại phần song song của dev; việc đó sẽ thành bản 1.1.

## 2. dev_v1

- **Khối song song:** đúng dev_v0, 2.000 câu, không dựng lại. Id và mọi trường giữ nguyên (đã assert). Thêm 1.000 dòng không dấu cho các câu tiếng Việt.
- **Khối bản địa:** lấy từ split dev, cân theo nhãn, mỗi dòng nguồn sinh một câu hỏi:

| Nguồn | Câu | Ghi chú |
|---|---|---|
| ViFactCheck | **163** | 560/723 câu dev dùng chung bài báo với split test nên bị loại |
| ViWikiFC | 300 | 1.785/2.090 câu dev dùng chung đoạn Wikipedia với split test; phần còn lại vẫn đủ |
| ViHSD | 300 | loại 157 câu trùng hoặc chứa bình luận của split test |

- Mỗi câu bản địa có thêm bản không dấu. Tổng dev_v1: 4.526 dòng.
- **Cách chuyển thành câu hỏi:** đúng như brief. Câu score giữ nguyên thứ tự mức; câu choice xáo theo dòng; câu noul dùng hai lựa chọn Có/Không theo thứ tự cố định.
- **Câu hỏi mới** nằm trong `config/bench_v1_questions.json` (**chưa được người duyệt**).
- **Nhãn ViHSD:** README không ghi nhãn số tương ứng với mức nào. Tôi dùng 0/1/2 = sạch/xúc phạm/thù ghét, suy ra từ bài báo gốc và phân bố nhãn.
- **Bản không dấu:** bỏ dấu kết hợp, và đổi đ/Đ thành d/D, trên state, câu hỏi và lựa chọn. Đã assert mọi chuỗi số và id không đổi. Các từ khung prompt của dự án ("Văn bản:", "Câu hỏi:"…) vẫn giữ dấu.

## 3. Chống rò rỉ

- **Test và dev_v1:** **0** trùng ở cả hai chiều. Kiểm bằng ba phép: trùng nguyên văn, gần trùng MinHash ≥ 0,8, và chứa đoạn. Phép chứa đoạn xét cả câu ngắn của test nằm trong văn bản dev lẫn ngược lại.
- **train_v0 (cùng val_v0) và test: 3 câu.**
  - 2 câu ViNLI (`pholaya/vinli`) nằm nguyên văn trong bài báo ViFactCheck test.
  - 1 bình luận VSFC (`pholaya/vsfc/topic`) trùng hệt một bình luận ViHSD test.
- **train_v0 và dev_v1: 1 câu.** Một bình luận ViSFD (`pholaya/visfd/aspect`) chứa nguyên một bình luận ViHSD dev.
- **Cách xử lý:** theo brief, không sửa gì. **Adapter của dự án được đánh dấu là đã thấy văn bản của ViFactCheck (2 câu) và ViHSD (2 câu).** Mức này rất nhỏ so với 300 câu mỗi nguồn.
- **Danh sách nguồn cấm dùng để train** đã được bổ sung ViFactCheck, ViWikiFC và ViHSD trong `PLAN_decisions.md`.

## 4. Cổng kiểm tra trước khi tin số trên dev_v1

- [x] **Đường chấm của dự án tái lập dev_v0:** trên 2.000 dòng dev_v0 nằm trong dev_v1, cả Qwen3-1.7B gốc lẫn adapter full_vocab cho probs, pred, letter_mass và top_token trùng hoàn toàn với kết quả cũ. Đường chấm cho noul và score **không cần code mới**: cả hai được trình bày thành lựa chọn chữ cái (Có/Không; ba mức), chấm bằng nguyên `eval_base.py`.
- [x] **decider-2b tái lập bước baseline:** trên 2.000 dòng song song có dấu, xác suất trùng hoàn toàn với `results/baselines` ở cả cách ánh xạ "text" lẫn "keyed" (lệch tối đa 0; 0 quyết định đổi).
  - Lượt chạy đầu có lỗi: các dòng dev_v0 mang `kind` cũ (`reading_mc`/`topic`/`intent`/`causal`), nên adapter mới đưa chúng nhầm sang đường score. Lỗi đã được phát hiện bằng chính cổng này và sửa trước khi lập bảng.
- [x] **Adapter decider cho noul và score:** đi đúng đường `system_one` của gói. So với `Decider.system_one()` (đã làm tròn 4 chữ số), lệch tối đa 4,9e-5. Temperature theo loại giữ nguyên bản phát hành: noul 1,624; score 1,124 (mỗi mức một hàng yes/no, chuẩn hóa P(yes)).
- [x] **Câu hiển nhiên noul/score** (`results/dev_v1/obvious_typed.jsonl`; thang trung tính, xuôi và ngược, Việt và Anh):

| Model | vi: noul / score (cả hai chiều thang) | en: noul / score (cả hai chiều thang) |
|---|---|---|
| decider-2b | 10/10 · 5/5 | 10/10 · 5/5 |
| Qwen3-1.7B gốc | 10/10 · 4/5 (một câu đúng ở thang xuôi, sai ở thang ngược) | 9/10 · 5/5 |
| Dự án (full_vocab) | 9/10 · 5/5 | 10/10 · 5/5 |

  Không có dấu hiệu gán nhầm có hệ thống. Xác suất luôn cộng bằng 1 và đúng số phần tử.

- [x] **Cắt cụt:** đường chấm của dự án đưa nguyên prompt vào model, không cắt. Với decider-2b, tôi đặt giới hạn state 32.768 token; 14 câu (bài báo ViFactCheck) vượt mức cắt mặc định 1.536 token của `decide()` và đáng lẽ đã bị cắt nếu để mặc định. Không câu nào bị cắt hay bị loại.

## 5. Kết quả trên dev_v1

decider-2b được báo với cách ánh xạ "text" cho câu choice. Cách này cho nó điểm cao hơn: 71,0% so với 69,3% của "keyed" trên mọi câu choice. Accuracy (%) dưới đây tính trên mọi câu trong lát, kèm [CI 95% bootstrap theo state].

| Lát | n | Đoán đều / nhãn phổ biến | 1.7B gốc | Dự án (full_vocab) | decider-2b | decider − dự án (p) |
|---|---|---|---|---|---|---|
| song song · en | 1.000 | 19,8 / 20,3 | 82,9 | 85,5 [83,2; 87,6] | **88,6** [86,6; 90,5] | +3,1 (0,005) |
| song song · vi · có dấu | 1.000 | 19,8 / 20,3 | 71,7 | 79,9 [77,4; 82,2] | **86,0** [83,7; 88,1] | +6,1 (9e-7) |
| song song · vi · **không dấu** | 1.000 | 19,8 / 20,3 | 47,4 | **59,4** [56,4; 62,4] | 48,4 [45,5; 51,5] | **−11,0** (5e-9) |
| bản địa · có dấu | 763 | 41,3 / 42,5 | 53,7 | 65,3 [62,0; 68,4] | 63,4 [59,7; 67,0] | −1,8 (0,39) |
| bản địa · **không dấu** | 763 | 41,3 / 42,5 | 46,9 | **52,3** [48,8; 55,8] | 41,2 [37,6; 44,9] | **−11,1** (3e-8) |

**Theo nguồn, tiếng Việt** (có dấu → không dấu):

| Nguồn | Dự án | decider-2b | decider − dự án, có dấu (p) | decider − dự án, không dấu (p) |
|---|---|---|---|---|
| Belebele | 77,3 → 58,3 | 83,0 → 49,0 | +5,7 (0,04) | −9,3 (0,009) |
| SIB-200 | 81,7 → 58,3 | 82,3 → **36,0** | +0,7 (0,87) | **−22,3** (1e-9) |
| MASSIVE | 81,0 → 62,7 | 92,7 → 58,7 | +11,7 (6e-10) | −4,0 (0,24) |
| XCOPA | 79,0 → 56,0 | 86,0 → 53,0 | +7,0 (0,17) | −3,0 (0,74) |
| ViFactCheck | 71,2 → 62,6 | 81,0 → 52,8 | +9,8 (0,02) | −9,8 (0,07) |
| ViWikiFC | 76,7 → 64,7 | 76,7 → **42,0** | 0,0 (1) | **−22,7** (9e-10) |
| ViHSD | 50,7 → 34,3 | **40,7** → 34,0 | **−10,0** (0,01) | −0,3 (1) |

**Theo loại câu hỏi, khối bản địa có dấu:**
- choice: dự án 66,0, decider 72,3 (+6,2; p=0,07).
- noul: dự án **73,8**, decider 65,6 (−8,3; p=0,002).
- score: cả hai 42,4.
- **Noul theo nguồn** (ngưỡng 0,5):
  - ViFactCheck: cả hai 88,5% (Brier nhị phân: dự án 0,092, decider 0,080).
  - ViWikiFC: cả hai 83,7% (0,127 so với 0,117).
  - **ViHSD: dự án 58,3%, decider 39,1%.** Tỉ lệ đúng của tập là 64,1%, nên decider thấp hơn cả mốc luôn đoán "không".
- **Score ViHSD:** sai số tuyệt đối của mức kỳ vọng là 0,632 (dự án) và 0,617 (decider), so với mốc 0,674 khi luôn đoán mức trung bình. Accuracy của cả hai đều 42,4%.

**Mức tụt khi bỏ dấu** (có dấu − không dấu, ghép theo pair_id):

| | Dự án | decider-2b |
|---|---|---|
| Khối song song | 20,5 [17,4; 23,6] | **37,6** [34,5; 40,6] |
| Khối bản địa | 13,0 [9,3; 16,7] | **22,3** [18,1; 26,4] |

Mọi chênh lệch trong bảng này đều có p < 0,005, trừ ViFactCheck và ViHSD của model gốc.

**Mức tụt Anh → Việt** (khối song song, có dấu): 1.7B gốc 11,2; dự án 5,6 [3,1; 7,9]; decider 2,6 [0,6; 4,6].

**Theo độ dài state** (tiếng Việt, có dấu):

| Nhóm | n | Dự án | decider-2b | decider − dự án (p) |
|---|---|---|---|---|
| ≤ 256 token | 1.582 | 73,9 | 75,9 | +2,0 (0,09) |
| 257–1.024 token | 140 | 72,1 | 79,3 | +7,1 (0,11) |
| > 1.024 token | 41 (31 bài) | 65,9 | 78,0 | +12,2 (0,23) |

Ở bản không dấu, các nhóm độ dài được đếm trên token của chính văn bản không dấu nên số câu mỗi nhóm hơi khác. decider kém dự án ở cả ba nhóm (−10,9, −12,1, −11,9).

## 6. Trả lời ba câu hỏi (chỉ số, không đề xuất hướng train)

Mốc so sánh là mức của chính decider-2b trên khối song song tiếng Việt có dấu: **86,0%**.

**1. decider-2b tụt ở lát nào, và tụt bao nhiêu so với mốc đó?**
- **Văn bản không dấu.** Khối song song chỉ còn 48,4% (−37,6 điểm), khối bản địa 41,2%.
  - Nặng nhất là SIB-200 (82,3 → 36,0, tức −46,3 điểm) và ViWikiFC (76,7 → 42,0).
  - Khi đã bỏ dấu, decider chỉ ngang Qwen3-1.7B gốc ở khối song song (48,4 so với 47,4) và thấp hơn model gốc ở khối bản địa (41,2 so với 46,9).
- **ViHSD, bình luận mạng xã hội:** 40,7%, tức −45,3 điểm so với mốc và thấp hơn mốc nhãn phổ biến (48,0). Riêng câu noul ViHSD chỉ đạt 39,1%.
- **Câu noul của khối bản địa:** 65,6% (−20,4). Phần lớn do ViHSD; ở hai bộ factcheck, noul của decider ngang dự án.
- **Khối bản địa có dấu nói chung:** 63,4% (−22,6). Một phần là do bài bản địa vốn khó hơn với mọi model: dự án đạt 65,3, model gốc 53,7.
- **ViWikiFC có dấu:** 76,7% (−9,3).

**2. Ở lát nào adapter của dự án ngang hoặc hơn decider-2b?**
- **Hơn, có ý nghĩa thống kê:**
  - Mọi lát không dấu: song song +11,0, bản địa +11,1, SIB-200 +22,3, ViWikiFC +22,7, Belebele +9,3.
  - ViHSD có dấu: +10,0 (p=0,01).
  - Noul bản địa có dấu: +8,3 (p=0,002).
- **Ngang, không phân biệt được:**
  - Khối bản địa có dấu nói chung (−1,8; p=0,39).
  - ViWikiFC có dấu (0,0), SIB-200 có dấu (+0,7 cho decider).
  - Score ViHSD (0,0), XCOPA có dấu (+7,0 cho decider; p=0,17).
  - Không dấu của MASSIVE (−4,0; p=0,24) và XCOPA (−3,0; p=0,74).
- **decider hơn rõ:**
  - Song song tiếng Việt có dấu (+6,1), tiếng Anh (+3,1).
  - MASSIVE có dấu (+11,7; nguồn mà decider có thể đã thấy khi train).
  - ViFactCheck có dấu (+9,8; p=0,02), Belebele có dấu (+5,7; p=0,04).

**3. Lát nào còn quá ít mẫu để kết luận?**
- **State dài hơn 1.024 token:** 41 câu có dấu, chỉ 31 bài; CI của chênh lệch là [−2,4; +28,2].
- **ViFactCheck dev:** 163 câu trên 115 bài, vì dev và test dùng chung bài báo. Trong đó noul có 78 câu, choice 85 câu. Vì không đủ mẫu, nhãn cũng không cân: 47 được hỗ trợ, 62 bị bác bỏ, 54 không đủ thông tin.
- **XCOPA:** 100 câu.
- **Score:** chỉ có ở ViHSD, 144 câu.
- **Từng tổ hợp loại × nguồn của khối bản địa:** mỗi ô khoảng 80–160 câu; CI rộng khoảng ±10 điểm.
- **ViHSD:** dev được lấy cân theo nhãn (1/3 sạch), khác hẳn phân bố tự nhiên (khoảng 82% sạch). Accuracy và mốc nhãn phổ biến ở đây chỉ phản ánh mẫu cân này.

## Những gì tôi không chắc

- **Câu hỏi và mô tả lựa chọn của khối bản địa chưa được duyệt.** Chúng do agent viết. Cách diễn đạt của câu noul ViHSD có thể khiến decider (một model tiếng Anh, được hỏi bằng tiếng Việt) hiểu sai chiều câu hỏi. Mức 39% (dưới mốc luôn đoán "không") gợi ý điều đó, nhưng tôi chưa kiểm chứng.
- **Bản không dấu vẫn giữ dấu ở các từ khung prompt của dự án.** decider dùng khung tiếng Anh cố định, nên không bị ảnh hưởng.
- **Ánh xạ nhãn số của ViHSD** được suy ra từ bài báo, không có trong README.
- **decider-2b có thể đã thấy MASSIVE khi train** (theo REPORT_baselines). Adapter của dự án đã thấy 4 câu văn bản của ViFactCheck và ViHSD (mục 3).
- **Chỉ có 1 seed** cho adapter của dự án.
- **Nhóm độ dài của bản không dấu** được đếm trên văn bản không dấu, nên số câu mỗi nhóm khác với bản có dấu.

## Việc cần người duyệt

1. `config/bench_v1_questions.json`: câu hỏi và mô tả lựa chọn cho ba nguồn bản địa (chưa được người duyệt).
2. `results/dev_v1/review_native_dev20.md`: 20 dòng dev mỗi nguồn bản địa, gồm cả bản không dấu. Nguồn ViHSD có thể chứa ngôn từ xúc phạm hoặc thù ghét.
3. Phần thiếu của test 1.0 so với quy mô trong brief (SIB-200 120/204, Belebele 272/300) và của dev bản địa (ViFactCheck 163/300): chấp nhận như hiện tại, hay sửa ở bản 1.1.
4. Ánh xạ nhãn số của ViHSD (0 sạch, 1 xúc phạm, 2 thù ghét).
5. Bốn câu train_v0 trùng với ViFactCheck và ViHSD: theo brief, không sửa, chỉ đánh dấu.

## File

| File | Nội dung |
|---|---|
| `data/bench_v1/test_v1.0.jsonl`, `manifest_test_v1.0.json`, `FROZEN_test_v1.0.sha256` | tập test 1.0 đã đóng băng (chỉ đọc) |
| `data/bench_v1/dev_v1.jsonl`, `manifest_dev_v1.json`, `leakage_v1.json` | dev_v1 và kết quả kiểm rò rỉ |
| `config/bench_v1_questions.json` | câu hỏi mới (chưa được người duyệt) |
| `results/dev_v1/eval/<model>/preds.jsonl` | dự đoán thô (base, full_vocab, decider-2b text và keyed) |
| `results/dev_v1/tables_dev_v1.md`, `metrics_dev_v1.json` | bảng đầy đủ |
| `results/dev_v1/obvious_typed.jsonl`, `obvious_typed_eval*/`, `eval/checks_decider_v1.json` | các cổng kiểm adapter |
| `scripts/bench/*.py`, `setup/eval_dev_v1*.sh` | code |
