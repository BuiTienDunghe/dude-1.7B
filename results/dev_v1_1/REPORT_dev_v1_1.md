# Bench bản 1.1 (bỏ ViHSD, sửa câu chữ) và chấm lại trên dev_v1.1

Ngày 2026-10-13. Không train. Không model nào chạy trên test (1.0 hay 1.1). Bảng đầy đủ ở `tables_v1_1.md`, số máy đọc được ở `metrics_v1_1.json`.

## Tóm tắt

- **Cổng A đạt.** Đáp án đúng của dev_v1 khớp 100% trường nhãn của nguồn: Belebele 600/600, MASSIVE 600/600; kiểm thêm SIB-200 600/600 và XCOPA 100/100. Hai câu nghi lỗi đều khớp nguồn, nên là lỗi nguồn hoặc nhãn mơ hồ, không phải lỗi dựng. Hai câu giữ nguyên trong dữ liệu (mục 1).
- **Bản 1.1:**
  - Bỏ ViHSD.
  - Sửa câu chữ của câu kiểm chứng ("hỗ trợ" thành "xác nhận") và 8 mô tả ý định MASSIVE.
  - Giữ nguyên mọi câu đã lấy mẫu, id, thứ tự lựa chọn và nhãn nhiễu.
  - **Test 1.1 đã đóng băng:** SHA-256 `fbad818b11b7ca8e23767a7287e9db7dae9a6a767042c04eb79c053a8b811b0c`, 4.176 dòng.
  - **dev_v1.1:** 3.926 dòng (`18ba27b4…`).
- **Kết quả chính trên dev_v1.1:**
  - decider-2b vẫn dẫn ở bản có dấu: VI song song 86,2, bản địa 81,2.
  - Các adapter Qwen3 dẫn ở bản không dấu: song song 59,0–60,4, bản địa 64,0–65,4.
  - **Bỏ ViHSD đổi hẳn bức tranh khối bản địa.** Ở bản 1.0, ba model đứng ngang nhau ở bản địa có dấu (khoảng 66). Ở bản 1.1, decider-2b hơn adapter đối chứng 8,3 điểm [3,7; 12,8].
- **Đọc lại kết luận cũ (mục 5):**
  - **ft_decider_v0 đổi kết luận, từ "ĐÁNG THEO" thành "CÒN LẠI: báo số, không kết luận".** Mức tăng bản địa có dấu +4,8 ở bản 1.0 thành −0,3 [−3,3; +2,6] ở bản 1.1, vì mức tăng đó gần như chỉ đến từ ViHSD.
  - strip_v0 vẫn không đạt.
  - restore: không tính lại được ở lát định trước, vì MASSIVE đã đổi chữ.
- **Nguồn thay thế (mục 6, chỉ đề xuất):** UIT-ViCTSD, UIT-ViOCD, EXAMS (tiếng Việt). **Không ứng viên nào có nhãn thứ bậc.** File đọc kiểm 15 câu mỗi nguồn ở `results/review/candidates_review.html`.

---

## 1. Cổng A: đáp án dev_v1 so với dữ liệu gốc (`scripts/bench11/gate_a.py`, `gate_a.json`)

| Nguồn | Dòng có dấu kiểm | Khớp | Trường nhãn nguồn |
|---|---|---|---|
| Belebele (EN + VI) | 600 | 600 | `correct_answer_num`. Cũng kiểm: đủ 4 lựa chọn, state và câu hỏi trùng nguồn |
| MASSIVE (EN + VI) | 600 | 600 | `label`. Cũng kiểm: tập lựa chọn = nhãn đúng + 9 nhãn nhiễu ghi trong flags, mô tả theo translations bản 1.0 |
| SIB-200 (tham khảo) | 600 | 600 | `category` |
| XCOPA VI (tham khảo) | 100 | 100 | `label` |

**Hai câu nghi lỗi nguồn.** Cả hai giữ nguyên trong dữ liệu, không sửa tay:
- `belebele:…/High_School_Earth_Science/Telescopes#2` (P4-belebele-003): đáp án dev (A) "Kể từ cuối thế kỉ 16" trùng `mc_answer2` của nguồn, `correct_answer_num` = 2.
- `massive:10457` (P4-massive-001): đáp án dev "xem nội dung danh sách" là mô tả của `label` = `lists_query` trong nguồn.

## 2. Sửa câu chữ (mục B)

- Bản config cũ lưu ở `config/archive_v1_0/`; diff ở `config_diff_*.txt`.
- Chỉ đổi đúng các mục được liệt kê; định dạng file giữ nguyên (translations.json: 9 dòng thêm, 8 dòng bớt).
- Mỗi mục đã sửa có `"_status": "người làm dự án đã duyệt ngày 2026-10-13"`, mỗi file có thêm khóa `_reviewed_items`. Trạng thái chung của các mục khác giữ nguyên ("chưa duyệt").
- **Kiểm chứng:**
  - câu hỏi choice: "…được văn bản **xác nhận**, bị văn bản bác bỏ, hay văn bản không đủ thông tin…";
  - lựa chọn: "Được văn bản **xác nhận**";
  - câu hỏi noul: "Văn bản có **xác nhận** phát biểu sau không?".
- **MASSIVE:** đổi mô tả music_settings, music_dislikeness, music_likeness, transport_query, transport_traffic, transport_ticket (chỉ tiếng Việt), general_greet, general_quirky. Đã kiểm 60 mô tả vẫn khác nhau ở cả hai ngôn ngữ.

## 3. Dựng bản 1.1 (mục C; `scripts/bench11/build_bench_v1_1.py`)

- **Không lấy mẫu lại.**
  - Lựa chọn được đổi theo khóa nhãn: chữ cũ → khóa (ý định / nhãn kiểm chứng) → chữ mới. Nhờ vậy thứ tự lựa chọn (seed xáo cũ) và nhãn nhiễu của MASSIVE giữ nguyên.
  - Câu hỏi kiểm chứng: tách nhận định khỏi mẫu cũ rồi ghép vào mẫu mới.
- **Bản không dấu** sinh lại bằng đúng phép của bench 1.0. Dòng không đổi chữ ra bản không dấu trùng từng trường với 1.0: dev 766 dòng, test 752 dòng. n_tokens tính lại cho dòng đổi chữ; tính lại cho 200 dòng không đổi ra đúng số cũ.
- **Test 1.0** chỉ đọc qua script: so hash với manifest, không in nội dung, không chạy model. File 1.0 giữ nguyên ở `data/bench_v1/`.
- **Số dòng:**

| | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC | Tổng |
|---|---|---|---|---|---|---|---|
| dev_v1_1 | 900 | 900 | 900 | 300 | 326 | 600 | 3.926 |
| test_v1.1 (đóng băng, chỉ đọc) | 816 | 360 | 900 | 900 | 600 | 600 | 4.176 |

- **Kiểm rò rỉ** (decontam như bản 1.0):
  - dev và test: 0 / 0.
  - train_v0 + val_v0 so với dev_v1.1: **0** (bản 1.0 có 1, thuộc ViHSD).
  - So với test 1.1: **2**. Cả hai là ViFactCheck chứa đoạn 13-gram trùng ViNLI, đã có từ bản 1.0, chưa xử lý.
- **Manifest** (`data/bench_v1_1/manifest_*.json`) ghi:
  - không có kiểu câu score;
  - khối bản địa chỉ gồm kiểm chứng (choice 3 lựa chọn, noul Có/Không);
  - SIB-200 có trần do nhãn theo chủ đề của bài gốc;
  - kết quả cổng A và hai câu nghi lỗi nguồn.

## 4. Chấm lại trên dev_v1.1 (mục D, E)

### 4.1 Cách chấm

- **Cổng E (đạt):** chấm lại adapter đối chứng trên 2.100 dòng không đổi (Belebele, SIB-200, XCOPA, mọi biến thể); preds trùng từng byte với bản cũ (2.100/2.100).
- **Dòng chấm lại:** toàn bộ MASSIVE (900 dòng, kể cả 205 dòng không đổi chữ) và kiểm chứng (926 dòng). Belebele, SIB-200, XCOPA dùng lại preds cũ sau khi xác nhận nội dung dòng trùng từng byte.
- **Thứ tự:** decider-2b và adapter đối chứng trước, rồi Qwen3-1.7B gốc, adapter bỏ dấu, decider-ft epoch 2. Mỗi model dùng đúng đường chấm cũ, batch 1.
- **decider-2b và decider-ft:** chấm cả ánh xạ "text" lẫn "keyed"; bảng dùng ánh xạ cho điểm cao hơn. Cả hai đều là "text": trung bình năm lát 70,8 so với 70,1 (decider-2b) và 72,5 so với 72,4 (decider-ft).

### 4.2 Năm lát chính (accuracy %, trung bình đều theo nguồn)

| Lát | n | decider-2b | adapter đối chứng | Qwen3-1.7B gốc | strip (Qwen3) | decider-ft ep2 |
|---|---|---|---|---|---|---|
| song song · EN | 1.000 | **89,4** | 86,8 | 83,5 | 87,1 | 88,7 |
| song song · VI có dấu | 1.000 | **86,2** | 80,4 | 72,8 | 81,2 | 84,6 |
| song song · VI không dấu | 1.000 | 47,9 | 59,0 | 47,8 | **60,4** | 51,0 |
| bản địa · có dấu | 463 | **81,2** | 72,9 | 57,4 | 73,6 | 81,0 |
| bản địa · không dấu | 463 | 49,4 | **65,4** | 46,3 | 64,0 | 57,4 |

Chênh lệch theo cặp, CI 95% bootstrap theo state:

| Lát | adapter đối chứng − decider-2b | decider-ft − decider-2b |
|---|---|---|
| song song · EN | −2,7 [−5,0; −0,5] | −0,8 [−2,2; +0,8] |
| song song · VI có dấu | −5,8 [−8,5; −3,0] | −1,6 [−3,5; +0,3] |
| song song · VI không dấu | +11,1 [+7,1; +15,3] | +3,1 [−0,4; +6,6] |
| bản địa · có dấu | −8,3 [−12,8; −3,7] | −0,3 [−3,1; +2,6] |
| bản địa · không dấu | +16,0 [+10,5; +21,8] | +8,0 [+3,6; +12,8] |

### 4.3 Theo nguồn ở các nguồn đổi chữ (accuracy %)

| Nguồn · biến thể | decider-2b | adapter đối chứng | Qwen3 gốc | strip | decider-ft |
|---|---|---|---|---|---|
| MASSIVE · EN | 95,0 | 88,0 | 88,3 | 89,0 | 94,0 |
| MASSIVE · VI có dấu | 93,3 | 83,7 | 70,7 | 86,0 | 91,3 |
| MASSIVE · VI không dấu | 53,7 | 63,3 | 43,0 | 65,0 | 62,7 |
| ViFactCheck · có dấu | 82,8 | 70,6 | 55,8 | 71,2 | 81,0 |
| ViWikiFC · có dấu | 79,7 | 75,3 | 59,0 | 76,0 | 81,0 |

decider-2b, cùng câu, từ bản 1.0 sang 1.1:
- MASSIVE có dấu: 92,7 → 93,3. Không dấu: 58,7 → 53,7.
- ViFactCheck có dấu: 81,0 → 82,8.
- ViWikiFC có dấu: 76,7 → 79,7.

### 4.4 Theo kiểu câu (khối bản địa)

| Kiểu · biến thể | n | decider-2b | adapter đối chứng | Qwen3 gốc | strip | decider-ft |
|---|---|---|---|---|---|---|
| choice · có dấu | 256 | 74,8 | 62,7 | 37,2 | 63,9 | 74,2 |
| choice · không dấu | 256 | 32,9 | 51,6 | 31,1 | 52,8 | 44,0 |
| noul · có dấu | 207 | 88,8 | 84,7 | 81,2 | 84,8 | 89,2 |
| noul · không dấu | 207 | 69,2 | 81,6 | 64,9 | 77,0 | 73,6 |

### 4.5 Mức sàn (mục D; dòng tiếng Việt có dấu, tính trên ĐÁP ÁN)

| Nguồn · kiểu câu | n | số lựa chọn | ngẫu nhiên | luôn chọn đáp án phổ biến nhất | luôn chọn vị trí phổ biến nhất |
|---|---|---|---|---|---|
| Belebele · reading_mc | 300 | 4 | 25,0 | 0,7 (đáp án đều khác chữ) | 27,0 (A) |
| SIB-200 · topic | 300 | 7 | 14,3 | 18,0 ("khoa học/công nghệ") | 18,0 (C) |
| MASSIVE · intent | 300 | 10 | 10,0 | 2,0 | 13,0 (G) |
| XCOPA · causal | 100 | 2 | 50,0 | 1,0 | 51,0 (A) |
| ViFactCheck · choice | 85 | 3 | 33,3 | 43,5 ("Văn bản không đủ thông tin") | 40,0 (A) |
| ViFactCheck · noul | 78 | 2 | 50,0 | **62,8 ("Không")** | 62,8 |
| ViWikiFC · choice | 171 | 3 | 33,3 | 46,2 ("Bị văn bản bác bỏ") | 35,1 (A) |
| ViWikiFC · noul | 129 | 2 | 50,0 | **61,2 ("Không")** | 61,2 |

Câu noul có sàn khoảng 62%, không phải 50%: "Không" gộp cả hai nhãn gốc bác bỏ và không đủ thông tin.

### 4.6 Câu noul của khối bản địa: tỉ lệ dự đoán "Có" và AUROC của P(Có)

Tỉ lệ đáp án đúng là "Có": 38,2% (207 câu).

| Biến thể | decider-2b | adapter đối chứng | Qwen3 gốc | strip | decider-ft |
|---|---|---|---|---|---|
| có dấu: dự đoán "Có" % / AUROC | 42,5 / 0,936 | 45,4 / 0,940 | 51,7 / 0,868 | 47,8 / 0,939 | 43,0 / 0,952 |
| không dấu: dự đoán "Có" % / AUROC | 43,5 / 0,718 | 43,5 / 0,848 | 62,8 / 0,761 | 46,4 / 0,841 | 45,4 / 0,795 |

## 5. Đọc lại các kết luận cũ trên bản 1.1 (mục F)

| Thí nghiệm | Con số chính, bản 1.0 | Bản 1.1 | Kết luận theo ngưỡng định trước |
|---|---|---|---|
| **strip_v0** (strip − đối chứng) | không dấu: +2,0 song song / +3,0 bản địa; có dấu +1,1 / +0,1; EN +0,4 | +1,4 [−2,1; +4,9] / −1,4 [−5,1; +2,3]; có dấu +0,8 / +0,6; EN +0,3 | KHÔNG ĐẠT → KHÔNG ĐẠT (không đổi). Mức +3,0 bản địa ở 1.0 là ViHSD; bỏ đi thì còn −1,4 |
| **ft_decider_v0** (ft − decider-2b, ánh xạ text) | EN −0,7; VI có dấu −1,8; bản địa có dấu +4,8 [+1,9; +7,6]; không dấu +1,4 | EN −0,8; VI có dấu −1,6 [−3,6; +0,2]; bản địa có dấu **−0,3 [−3,3; +2,6]**; không dấu +3,1 | **ĐÁNG THEO → CÒN LẠI (báo số, không kết luận)**. Mức tăng bản địa có dấu ở 1.0 đến từ ViHSD |
| **restore** (decider: có dấu − khôi phục, song song) | 4 nguồn: vit5 4,1 (≤ 5); small 11,8 (> 10) | **Không tính lại**: MASSIVE và kiểm chứng đổi chữ nên bản khôi phục cũ không còn khớp; không chạy lại vit5/small | Không xác định trên 1.1. Chỉ có số tham khảo trên Belebele + SIB + XCOPA (giống nhau ở 1.0 và 1.1): vit5 3,1, small 11,4; không áp được ngưỡng định cho 4 nguồn |

## 6. Nguồn thay thế đề xuất (mục G; chỉ đề xuất, CHƯA thêm vào benchmark)

Điều kiện: bản địa, nhãn người gán, không trùng train_v0, không phải dạng kiểm chứng. Lấy từ `results/train_v1_survey/REPORT_sources.md`.

- **Không ứng viên nào có nhãn thứ bậc.** Trong khảo sát, nguồn bản địa mở có nhãn thứ bậc gần nhất là UIT-VSFC (cảm xúc 3 mức), nhưng nó đã nằm trong train_v0.
- Muốn có lại kiểu score thì phải tìm thêm ngoài khảo sát, hoặc nhận một bộ cần thỏa thuận.

| Ứng viên | Dạng bài đề xuất | Số dòng train | Giấy phép (nguyên văn) | Độ đồng thuận người gán nhãn (theo bài báo) | Ghi chú |
|---|---|---|---|---|---|
| **UIT-ViCTSD** (`tarudesu/ViCTSD` @ `65a073f2`) | noul × 2: bình luận có tính xây dựng không; có độc hại không | 7.000 | Card: "The provided dataset is only used for research purposes!" (không có thẻ giấy phép) | Mục 3.3: "59.48% and 58.74% IAA" cho tính xây dựng và độc hại (bài báo gọi là Fleiss' Kappa nhưng báo dạng %); arXiv:2103.10069 | Bình luận VnExpress; có nội dung độc hại (cảnh báo trong file đọc kiểm). Điều kiện truy cập mâu thuẫn: repo HF của tác giả mở, trang của nhóm UIT ghi phải gửi email |
| **UIT-ViOCD** (`tarudesu/ViOCD` @ `25e18516`) | noul: đánh giá có phải lời phàn nàn không | 4.387 | Card: "The provided dataset is only used for research purposes!" (không có thẻ giấy phép) | Tóm tắt arXiv v3: "inter-annotator agreement Am of 87%"; arXiv:2104.11969 | Review thương mại điện tử; cùng dạng complaint với ViSFD trong train_v0 (khác dữ liệu). Điều kiện truy cập mâu thuẫn như ViCTSD |
| **EXAMS tiếng Việt** (`mhardalov/exams`, `crosslingual_vi` @ `4ff10804`) | choice 4 lựa chọn (câu hỏi đề thi THPT, không có đoạn văn) | 1.955 | Thẻ `cc-by-sa-4.0` | **Không công bố.** Đáp án lấy từ đề thi chính thức, không có bước gán nhãn của người; arXiv:2011.03080 | Hỏi kiến thức, không phải đọc hiểu; bản kèm đoạn văn là đoạn Wikipedia truy hồi, rất dài |

- File đọc kiểm: `results/review/candidates_review.html`, 15 câu ngẫu nhiên mỗi nguồn (seed 20261013, lấy qua dataset viewer; revision lúc lấy trùng revision lúc khảo sát).
- Câu hỏi tiếng Việt cho ViCTSD và ViOCD là đề xuất của agent, chưa duyệt; nhãn hiện đúng giá trị gốc.

## 7. Điều chưa chắc

- **Khối bản địa của 1.1 chỉ còn hai nguồn kiểm chứng** (463 câu có dấu), nên CI rộng hơn. Kết luận về khối bản địa giờ hoàn toàn là kết luận về kiểm chứng.
- **Câu noul có sàn 61–63%** do "Không" gộp hai nhãn. Accuracy noul cần đọc cạnh sàn này và AUROC (mục 4.6).
- **Hai dòng test 1.1 trùng 13-gram với ViNLI** trong train_v0, có từ bản 1.0, chưa xử lý.
- **Hai câu nghi lỗi nguồn** giữ nguyên. Công cụ soát nhãn (đang chờ người soát) sẽ cho biết còn câu tương tự không.
- **Thí nghiệm restore không còn số trên 1.1** ở lát định trước. Muốn có thì phải khôi phục lại các dòng MASSIVE và kiểm chứng (chưa làm, theo brief).
- **decider-ft dùng adapter train trên train_v0**, có ba bộ UIT đang chờ thỏa thuận. Chỉ dùng nội bộ.

## 8. File

- `data/bench_v1_1/`:
  - `dev_v1_1.jsonl`, `test_v1.1.jsonl` (chỉ đọc), `FROZEN_test_v1.1.sha256`;
  - `manifest_dev_v1_1.json`, `manifest_test_v1.1.json`, `leakage_v1_1.json`.
- `config/`:
  - `bench_v1_questions.json`, `translations.json` (đã sửa);
  - `archive_v1_0/` (bản dùng cho 1.0).
- `results/dev_v1_1/`:
  - `gate_a.json`, `config_diff_*.txt`, `gates/` (cổng E);
  - `rows/` (dòng chấm lại / dùng lại), `eval/` (preds ghép), `eval_rescore/` (preds chấm lại);
  - `tables_v1_1.md`, `metrics_v1_1.json`, `pipeline.log`.
- `results/review/`: `candidates_review.html`, `candidates_rows.json`.
- `tools/review_app/app.py`: công cụ soát nhãn (phần 1 của brief 2; lệnh khởi động ở docstring).
- Code: `scripts/bench11/{gate_a,build_bench_v1_1,split_rows,merge_preds,report_dev_v1_1,candidates_review}.py`, `setup/b11_*.sh`.
