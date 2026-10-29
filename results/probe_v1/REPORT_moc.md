# probe_v1: chấm mốc M0, M1, R trước khi train

Ngày 26/10/2026. Làm theo bộ phép thử đã chốt lại (mục "Cập nhật 26/10" của `results/BRIEF_phan_tich_xu_huong.md`).

**Phạm vi:**
- Chỉ dùng dev 1.3, sha `c6a9a75d…`. Không đụng test.
- **Không train.** Train bản C cần người dùng cho phép riêng.
- Không commit, không xóa file, không ghi `data/`.
- Đây là bộ phân tích, không phải benchmark, không dùng để train hay chọn checkpoint.

**Các model mốc:**

| Mã | Model |
|---|---|
| M0 | Qwen3-1.7B gốc |
| M1 | model dự án (full_vocab s1, adapter `results/ft_loss/train/Qwen3-1.7B-s1/adapter`) |
| R1 | decider-2b, ánh xạ text |
| R2 | Quyet-1.0-Medium (chạy offline, `.venv-baselines`) |

Số liệu đầy đủ ở `phan_tich/moc_M0_M1_R.json` và `.md` (bảng đủ mọi lát, mọi nhóm).

**Ghi chú 26/10 (sau khi train C):** `analyze_probe.py` đã đổi sang seed riêng cho mỗi lần tính CI bootstrap. Trước đó các nhóm dùng chung một bộ sinh ngẫu nhiên, nên CI phụ thuộc tập model được phân tích. JSON và `.md` đã được chạy lại. Một vài cận CI trong báo cáo này có thể lệch vài phần mười điểm so với JSON; **JSON là số chuẩn**. Accuracy và chênh lệch không đổi.

## 1. Việc đã làm

| Việc | Kết quả |
|---|---|
| Sửa `build_p1_p5.py` theo bộ chốt lại | P1′ = câu hỏi P5 kiểm chứng choice + 3 lựa chọn mới. P5 chỉ còn 4 khung. Mỗi khóa lấy lần duyệt mới nhất, bỏ qua lần từ chối. Thiếu giá trị duyệt thì bỏ qua và ghi vào manifest, không dừng. Đã chạy thật: **chỉ dựng được P5 kiểm chứng** (2 khung đã duyệt, 886 câu); P1′, P5 SIB và P5 MASSIVE ghi "chưa đủ duyệt" |
| (a) `pa_lua_chon_khong_dau.jsonl` | 1.036 câu VI có dấu: ViFactCheck 158, ViWikiFC 285, SIB 299, MASSIVE 294. Lựa chọn bỏ dấu; văn bản và câu hỏi giữ dấu. sha `d94c4abc…` |
| (b) `pb_nhan_tieng_anh.jsonl` | 593 câu VI có dấu: SIB 299 (7 nhãn), MASSIVE 294 (60 nhãn). Lựa chọn lấy từ dòng EN cùng cặp; ánh xạ VI↔EN đã kiểm nhất quán trên toàn dev. sha `fb9bf69e…` |
| Kiểm (a), (b) trùng train_v1 | state và câu hỏi: **0 trùng** ở A, B, C (bỏ câu hỏi mẫu, như P3/P4). Xem chi tiết dưới bảng |
| (c), (d) trong `analyze_probe.py` | Đã thêm, kết quả ở mục 3 |
| `train_lora.py --train_file / --val_file` | Mặc định giữ train_v0/val_v0. Chạy giả `--dump_batches 3`, không train: xem chi tiết dưới bảng |
| Chấm mốc | 22/22 lượt exit 0, batch 1, khoảng 80 phút GPU. Log: `eval/run_log.jsonl`, `eval/logs/` |

**Lưu ý về kiểm trùng (a), (b):**
- Lượt kiểm đầu báo 1.186 chỗ trùng. Tất cả đều là câu hỏi mẫu "Người dùng muốn trợ lý làm gì?" và "Văn bản này thuộc chủ đề nào?", vốn phần A dùng lại có chủ ý.
- Chữ lựa chọn bỏ dấu (a): **0 lần** xuất hiện làm lựa chọn trong C.
- Nhãn tiếng Anh (b): 6/67 nhãn có trong C làm lựa chọn (sports 162, health 91, politics 45, entertainment 34, travel 4, geography 1). Đây không phải rò rỉ, chỉ cho biết model đã gặp các chữ này lúc train.

**Kết quả chạy giả `train_lora.py`:**
- Mặc định, hoặc truyền tường minh train_v0/val_v0: `batch_dump.jsonl` **giống từng byte** bản trước khi sửa (sha `0320706D…`).
- Với `--train_file results/train_v1/components/C.jsonl --strip_p 0.3 --save_fracs 0.25,0.5,0.75`: 1.938 bước (31.034 // 32 × 2), lưu adapter ở [484, 969, 1454].
- `summary.json` chỉ thêm trường `train_file`/`val_file` khi khác mặc định.

**Tái lập preds dev:** chấm lại dev 1.3 cho cả 4 model, kết quả **trùng 3.847/3.847 dự đoán, lệch xác suất 0** so với preds đã có ở `bench_v1_3_trial/eval`.

| Model | Accuracy dev |
|---|---|
| M0 | 65,87% |
| M1 | 75,80% |
| R1 | 73,46% |
| R2 | 81,26% |

**State rỗng ở P3 với decider và Quyet:** cả hai chấm bình thường, **0 câu excluded/rejected/truncated**.

**Câu noul ở Pa:** decider và Quyet trả lời Có/Không bằng đường riêng, không đọc chữ lựa chọn, nên bỏ 202 dòng noul khi chấm hai model này. Pa của R chỉ tính 834 câu choice, nên dòng "toàn bộ" của R không so thẳng được với M.

## 2. Lát cắt dev (accuracy %, CI 95% bootstrap theo cụm)

| Lát | Nhóm | n | M0 | M1 | R1 decider | R2 Quyet |
|---|---|---|---|---|---|---|
| Toàn dev | | 3.847 | 65,9 [64,1; 67,6] | 75,8 [74,1; 77,4] | 73,5 [71,9; 74,9] | 81,3 [79,8; 82,6] |
| Ngôn ngữ | VI có dấu | 1.430 | 70,7 | 80,8 [79,0; 82,7] | 86,9 | 89,2 |
| | VI không dấu | 1.430 | 48,3 | 62,8 [60,2; 65,3] | 48,7 | 65,0 |
| | EN | 987 | 84,3 | 87,3 [85,1; 89,4] | 90,0 | 93,3 |
| VI có dấu | nguồn gốc VI (ViFactCheck, ViWikiFC) | 443 | 59,4 | 75,2 [71,4; 79,3] | 83,1 | 84,0 |
| | nguồn dịch (4 nguồn) | 987 | 75,8 | 83,4 [81,1; 85,6] | 88,5 | 91,6 |
| Kiểu câu | chọn một | 3.443 | 65,1 | 75,0 | 72,7 | 81,9 |
| | Có/Không | 404 | 72,8 | 82,4 | 79,7 | 75,5 |
| Số lựa chọn | 2 | 704 | 72,7 | 79,4 | 78,7 | 79,5 |
| | 3–4 | 1.364 | 57,1 | 70,7 | 69,1 | 78,8 |
| | ≥ 5 | 1.779 | 69,9 | 78,3 | 74,7 | 83,8 |
| Độ dài văn bản | ngắn / vừa / dài | 1.387 / 1.256 / 1.204 | 66,8 / 65,2 / 65,5 | 76,5 / 75,4 / 75,4 | 72,4 / 74,2 / 73,9 | 81,2 / 80,2 / 82,4 |
| Đã sửa ở 1.2 (dòng VI) | đã sửa / giữ nguyên | 598 / 2.262 | 62,0 / 58,8 | 70,7 / 72,1 | 68,6 / 67,5 | 78,4 / 76,8 |
| Cờ mơ hồ | có / không | 74 / 3.773 | 18,9 / 66,8 | 16,2 / 77,0 | 13,5 / 74,6 | 14,9 / 82,6 |

**Theo nguồn, VI có dấu:**

| Nguồn | M0 | M1 | R1 | R2 |
|---|---|---|---|---|
| Belebele | 75,2 | 82,7 | 88,1 | 94,2 |
| SIB-200 | 75,6 | 82,6 | 82,3 | 83,6 |
| MASSIVE | 75,2 | 86,7 | 96,3 | 96,6 |
| XCOPA | 80,0 | 78,0 | 86,0 | 93,0 |
| ViFactCheck | 57,6 | 72,8 | 83,5 | 88,6 |
| ViWikiFC | 60,4 | 76,5 | 82,8 | 81,4 |

Ngưỡng độ dài (tam phân riêng từng nguồn) ghi trong JSON, mục `cai_dat`.

**Hiệu chỉnh (P7, toàn dev):**

| Model | Brier | ECE | tự tin trung bình khi sai | tự tin trung bình khi đúng |
|---|---|---|---|---|
| M0 | 0,625 | 0,297 | 0,917 | 0,976 |
| M1 | 0,350 | 0,087 | 0,699 | 0,891 |
| R1 | 0,347 | 0,015 | 0,520 | 0,810 |
| R2 | 0,270 | 0,029 | 0,665 | 0,882 |

**Trùng lỗi với model tham chiếu:**

| Model | Jaccard tập câu sai với decider | P(decider sai \| M sai) | Jaccard với Quyet | P(Quyet sai \| M sai) |
|---|---|---|---|---|
| M0 | 0,379 | 0,488 | 0,319 | 0,375 |
| M1 | 0,382 | 0,579 | 0,350 | 0,460 |

Tỉ lệ sai riêng: decider 26,5%, Quyet 18,7%.

## 3. Phép thử

### P2: đảo thứ tự lựa chọn (chỉ model M)

Tỉ lệ chọn cùng đáp án khi đảo:

| Phạm vi | M1 | M0 |
|---|---|---|
| Toàn dev | **88,7%** | 69,0% |
| VI có dấu, theo nguồn | 82–99% | — |
| VI không dấu | 73–97% | — |

Accuracy khi đảo: M1 75,8 → 74,9; M0 65,9 → 63,0.

### P3: bỏ văn bản

Accuracy %, kèm mốc "đoán mò / luôn chọn đáp án hay gặp nhất":

| Nhóm | Mốc | M0 | M1 | R1 | R2 |
|---|---|---|---|---|---|
| Belebele EN | 25,0 / 26,9 | 42,9 | **45,6** [39,9; 51,5] | 51,4 | 58,8 |
| Belebele VI có dấu | 25,0 / 26,9 | 35,4 | 38,8 [33,1; 44,4] | 42,2 | 51,4 |
| Belebele VI không dấu | 25,0 / 26,9 | 23,5 | 31,3 | 28,2 | 30,3 |
| ViFactCheck choice VI | 33,3 / 42,5 | 27,5 | 46,2 [34,9; 57,3] | 41,2 | 38,8 |
| ViWikiFC choice VI | 33,3 / 46,6 | 33,5 | 31,1 [23,9; 38,5] | 34,8 | 44,1 |
| ViFactCheck Có/Không VI | 50,0 / 64,1 | 64,1 | 64,1 | 62,8 | 48,7 |
| ViWikiFC Có/Không VI | 50,0 / 62,1 | 63,7 | 60,5 | 56,5 | 53,2 |

**Cách đọc mốc:**
- **Belebele:** mọi model đều cao hơn đoán mò rõ rệt khi bỏ văn bản. M1 ở EN là 45,6%, so với mốc 25%. Câu hỏi và lựa chọn của Belebele tự chứa manh mối, hoặc model dùng kiến thức nền. Đây là mức nền của M1; M2 không được tăng ở đây.
- **Kiểm chứng:** M1 nằm quanh mốc "luôn chọn đáp án hay gặp".

**(d) Phân bố nhãn dự đoán ở P3 kiểm chứng choice (M1):**
- Bản có dấu: nghiêng hẳn về **"không đủ thông tin"**: 71% ở ViFactCheck, 73% ở ViWikiFC (đáp án thật: 42% và 26%). Đây là hành vi đúng khi không có văn bản.
- Bản không dấu: nghiêng về **"bác bỏ"** (68% và 58%), và **không lần nào chọn "xác nhận"**.
- Câu Có/Không: M1 trả lời "Không" 84–94%.
- So sánh: M0 nghiêng về "xác nhận" (58–60%); decider nghiêng về "không đủ thông tin"; Quyet nghiêng về "xác nhận".

### P4: văn bản không liên quan

Tỉ lệ chọn "không đủ thông tin" (choice) hoặc "Không" (Có/Không). Mốc thiên vị: trên câu dev gốc của cùng nhận định, model chọn nhãn này bao nhiêu khi đáp án thật là nhãn khác.

| Nhóm | M0 | M1 (mốc thiên vị) | R1 | R2 |
|---|---|---|---|---|
| ViFactCheck choice VI có dấu | 87,5 | **91,2** (2,2) | 88,8 | 97,5 |
| ViWikiFC choice VI có dấu | 78,9 | **88,2** (5,0) | 82,6 | 97,5 |
| ViFactCheck choice VI không dấu | 85,0 | 93,8 (6,5) | 45,0 | 78,8 |
| ViWikiFC choice VI không dấu | 55,9 | 91,9 (12,5) | 55,3 | 41,0 |
| Có/Không (4 nhóm) | 99–100 | 100 | 86–100 | 81–100 |

**Cách đọc:**
- M1 chọn "không đủ thông tin" 88–94% khi văn bản không liên quan, trong khi mốc thiên vị chỉ 2–12%. Tức là M1 thật sự phản ứng với văn bản, không phải lúc nào cũng chọn nhãn này.
- **P4 câu Có/Không bị loại khỏi phần kết luận** (điều phối chốt 26/10). Phép này không phân biệt được model: M0 và M1 đạt 100% ở cả 4 nhóm, R ở 81–100%. Đáp án luôn là "Không", và mọi model đều trả lời "Không" khi văn bản không liên quan. Số vẫn có trong JSON, nhưng không dùng để kết luận.

### Pa: lựa chọn bỏ dấu, văn bản và câu hỏi giữ dấu

Cột "mốc" là accuracy trên dòng không dấu toàn phần của cùng cặp.

| Nhóm | M0 dev → Pa (mốc) | M1 dev → Pa (mốc) | R1 dev → Pa | R2 dev → Pa |
|---|---|---|---|---|
| SIB-200 | 75,6 → 58,2 (51,2) | 82,6 → **71,6** (60,2), **−11,0** [−15,7; −6,0] | 82,3 → 51,2 (−31,1) | 83,6 → 80,3 (−3,3) |
| MASSIVE | 75,2 → 63,3 (45,6) | 86,7 → **78,6** (66,3), **−8,2** [−12,9; −3,7] | 96,3 → 77,2 (−19,1) | 96,6 → 86,7 (−9,9) |
| ViFactCheck choice | 33,8 → 45,0 | 60,0 → 57,5 (48,8), −2,5 [−11,4; +6,4] | 76,2 → 78,8 | 87,5 → 85,0 |
| ViWikiFC choice | 46,0 → 39,1 | 71,4 → 67,1 (55,9), −4,3 [−10,4; +1,3] | 78,3 → 68,9 | 82,0 → 83,2 |
| Có/Không (2 nguồn) | 0,0 | −1,3 / +0,8 | — | — |
| Toàn bộ M | 68,5 → 59,9 (47,7), −8,6 | 80,6 → **74,2** (64,4), **−6,4** [−8,9; −4,0] | — | — |

**Cách đọc:**
- M1 tụt rõ ở SIB-200 và MASSIVE khi chỉ bỏ dấu chữ nhãn. Tuy vậy, M1 vẫn cao hơn hẳn mốc không dấu toàn phần.
- Ở kiểm chứng và Có/Không, mức tụt nằm trong nhiễu.
- **Ngưỡng K2(a) cho M2** là "tụt mạnh hơn M1". Mốc của M1: −6,4 toàn bộ, −11,0 ở SIB, −8,2 ở MASSIVE, −2,5/−4,3 ở kiểm chứng.

### Pb: nhãn tiếng Anh, văn bản tiếng Việt

| Nhóm | M0 | M1 | R1 | R2 |
|---|---|---|---|---|
| SIB-200 | 75,6 → 77,6 | 82,6 → 84,3 (+1,7) | 82,3 → 82,9 | 83,6 → 85,3 |
| MASSIVE | 75,2 → 78,6 | 86,7 → 88,1 (+1,4) | 96,3 → 95,6 | 96,6 → 96,9 |

Mọi model đều giữ hoặc tăng nhẹ, CI chứa 0. Mốc dòng EN của M1: SIB 84,0, MASSIVE 90,1. M1 hiểu nghĩa nhãn chứ không chỉ khớp chữ Việt.

### (c) Nhầm nhãn SIB-200 và MASSIVE

Tỉ lệ dự đoán so với tỉ lệ đáp án của các nhãn theo dõi. TVD là khoảng cách giữa phân bố dự đoán và phân bố đáp án.

| Model, tập | TVD | health (dự đoán / đáp án) | sports | MASSIVE recommendation_locations | TVD MASSIVE |
|---|---|---|---|---|---|
| M1 dev VI có dấu | 0,074 | 14,7 / 13,4 | 14,1 / 13,4 | 1,7 / 1,7 | 0,088 |
| M1 dev VI không dấu | 0,141 | 14,4 / 13,4 | 7,4 / 13,4 | 0,0 / 1,7 | 0,211 |
| M1 Pa | 0,144 | 12,0 / 13,4 | 11,4 / 13,4 | 1,7 / 1,7 | 0,153 |
| M0 dev VI có dấu | 0,080 | 13,0 / 13,4 | 11,7 / 13,4 | 1,7 / 1,7 | 0,146 |
| R1 dev VI không dấu | 0,378 | 9,7 / 13,4 | **51,2** / 13,4 | 0,7 / 1,7 | 0,347 |

**Mốc của M1:** không dồn nhãn ở VI có dấu (health +1,3 điểm, sports +0,7 điểm). M2 sẽ được so với các số này. Decider dồn mạnh về "sports" ở SIB không dấu (51%). Đây là đặc điểm riêng của model đó.

### P5 và P1′ (người dùng duyệt 7 chữ ngày 26/10, `quyet_dinh_chat_pv1b.json`; bổ sung sau)

**Dựng và kiểm trùng:**
- P1′ (`p1b_doi_chu_nhan.jsonl`, 482 câu, sha `a926cb11…`): câu hỏi P5 kiểm chứng choice cộng 3 lựa chọn "Nhận định đúng" / "Nhận định sai" / "Chưa đủ căn cứ để đánh giá".
- P5 (`p5_doi_cau_hoi.jsonl`, 4 khung, 2.072 câu, sha `81e0b7f4…`; thay bản 2 khung `7a94058c…` chưa từng chấm). Gồm:
  - kiểm chứng choice và Có/Không;
  - SIB-200: "Đoạn văn này nói về chủ đề gì?";
  - MASSIVE: "Câu nói này yêu cầu trợ lý làm việc gì?".
- Kiểm trùng train_v1: **0 dòng** A/B/C dùng câu hỏi mới hay chữ lựa chọn mới.
- Chấm: 8/8 lượt exit 0, khoảng 28 phút GPU.

**Cách đọc bảng:**
- P5 chỉ đổi câu hỏi.
- P1′ dùng **chính câu hỏi P5** và đổi thêm chữ nhãn, nên **P1′ − P5 là phần do đổi chữ nhãn**.
- Số là accuracy %, kèm CI 95% bootstrap theo cụm và p McNemar.

**Kiểm chứng choice, gộp hai nguồn (241 câu mỗi biến thể):**

| Model | Biến thể | dev | P5 | P5 − dev | P1′ | P1′ − P5 |
|---|---|---|---|---|---|---|
| M0 | có dấu | 41,9 | 38,6 | −3,3 [−9,9; +3,9] | 52,3 | **+13,7** [+7,4; +20,2], p 0,0001 |
| M0 | không dấu | 31,1 | 33,6 | +2,5 [−5,3; +10,4] | 41,1 | +7,5 [−0,9; +15,1] |
| **M1** | có dấu | 67,6 | 70,5 | +2,9 [−0,4; +6,8] | 73,4 | +2,9 [−2,4; +7,8], p 0,32 |
| **M1** | không dấu | 53,5 | 53,1 | −0,4 [−6,0; +5,3] | 60,2 | +7,0 [+0,4; +13,5], p 0,057 |
| R1 decider | có dấu | 77,6 | 75,1 | −2,5 [−6,3; +1,3] | 73,9 | −1,2 [−6,6; +3,9] |
| R1 decider | không dấu | 33,2 | 41,5 | +8,3 [+1,2; +15,0] | 48,5 | +7,0 [−0,9; +15,3] |
| R2 Quyet | có dấu | 83,8 | 82,6 | −1,2 [−3,5; +0,8] | 82,2 | −0,4 [−3,0; +2,1] |
| R2 Quyet | không dấu | 49,4 | 55,2 | +5,8 [+0,4; +12,4] | 57,7 | +2,5 [−2,5; +7,5] |

**M1 theo nguồn (VI có dấu), kiểm chứng choice:**

| Nguồn | dev | P5 | P5 − dev | P1′ | P1′ − P5 |
|---|---|---|---|---|---|
| ViFactCheck (80) | 60,0 | 65,0 | +5,0 [−1,3; +12,8] | 75,0 | +10,0 [+1,3; +19,2], p 0,057 |
| ViWikiFC (161) | 71,4 | 73,3 | +1,9 [−1,9; +5,8] | 72,7 | −0,6 [−6,7; +5,1] |

**P5, các khung còn lại** (chênh P5 − dev):

| Khung | M0 | M1 | R1 | R2 |
|---|---|---|---|---|
| Kiểm chứng Có/Không, có dấu (202) | −0,5 | +0,0 [−3,0; +3,3] | +0,5 | +2,0 |
| Kiểm chứng Có/Không, không dấu | +7,9 | −0,5 [−3,9; +2,9] | +0,5 | −0,5 |
| SIB-200, có dấu (299) | +2,7 | +0,3 [−2,0; +2,7] | −0,3 | +0,0 |
| SIB-200, không dấu | +0,3 | +3,0 [−0,7; +6,7] | +1,3 | +0,0 |
| MASSIVE, có dấu (294) | −2,0 | −0,3 [−2,7; +1,7] | −0,3 | +0,3 |
| MASSIVE, không dấu | −5,4 | −0,3 [−3,4; +3,1] | +1,0 | −0,7 |

**Mốc cho M2** (theo định nghĩa K1/K2 trong mục "Cập nhật 26/10" của brief):
- M1 ở P5 nằm trong khoảng −0,5…+3,0 điểm ở cả 4 khung (VI có dấu và không dấu); không khung nào có CI lệch hẳn khỏi 0.
- M1 ở P1′ − P5 (kiểm chứng choice): +2,9 điểm (có dấu) và +7,0 điểm (không dấu). Nghĩa là với M1, chữ nhãn mới **không làm tụt** điểm, mà còn tăng nhẹ.
- Bảng đủ theo từng nguồn và biến thể nằm ở `phan_tich/moc_M0_M1_R.md`, mục "dev → P5 → P1′".

## 4. Chưa chấm được và vì sao

| Phép | Lý do |
|---|---|
| ~~P1′, P5~~ | Đã chấm sau khi người dùng duyệt 7 chữ (mục 3, "P5 và P1′") |
| P2 cho decider và Quyet | Không chạy. Hai model này đã có kết quả hai thứ tự ở bước E (dev 1.2) |
| Pa câu Có/Không cho decider và Quyet | Bỏ có chủ ý (mục 1) |
| P4 câu Có/Không | Đã chấm, nhưng **loại khỏi phần kết luận** vì không phân biệt được model (mục 3, P4) |

## 5. File

**Dữ liệu và kết quả** (trong `results/probe_v1/`):
- **Phép thử mới:** `pa_lua_chon_khong_dau.jsonl`, `pb_nhan_tieng_anh.jsonl`, `p5_doi_cau_hoi.jsonl` (4 khung), `p1b_doi_chu_nhan.jsonl`.
- **Quyết định duyệt 7 chữ:** `quyet_dinh_chat_pv1b.json`, do điều phối ghi; `build_p1_p5.py` đọc file này.
- **Manifest:** `manifest_probe_v1.json`, có thêm mục `Pa_Pb` và `P1_P5`.
- **Preds:** `eval/<phép>/<tên model>/preds.jsonl`; Quyet nằm ở `eval/<phép>/normal/quyet-medium-text/`. decider-2b-keyed cũng được ghi vì `eval_decider_v1.py` luôn chạy cả hai cách ánh xạ, nhưng không dùng.
- **Phân tích:** `phan_tich/moc_M0_M1_R.{json,md}`.
- **Bản cũ:** `phan_tich/THU_preds_co_san_M2_la_strip-s1.*` là bản thử cơ chế từ bước 0, nay được thay bằng bản trên. Không xóa.

**Script** (trong `scripts/probe_v1/`):
- `build_p1_p5.py` (viết lại), `build_pa_pb.py` (mới), `run_moc.sh` (mới);
- `score_probe.py`: thêm p1b/pa/pb, một phép lỗi không chặn các phép sau;
- `analyze_probe.py`: M2 thành tùy chọn; thêm P1′ − P5, mốc cho Pa/Pb, (c), (d), số câu bị loại;
- `build_p3_p4.py`: giữ mục của script khác khi ghi manifest; chạy lại cho cùng sha.

**Script train:** `scripts/train_lora.py` thêm `--train_file` và `--val_file`.

**Lệnh phân tích đã chạy:**

```bash
python scripts/probe_v1/analyze_probe.py --m1 "Qwen3-1.7B-fullvocab-s1 (dự án)" --m0 "Qwen3-1.7B" --r "decider-2b-text" --r "quyet-medium-text" --dev_eval results/probe_v1/eval/dev --out results/probe_v1/phan_tich/moc_M0_M1_R.json
```

Mức dùng lúc viết báo cáo: cửa sổ 5 giờ 1%, tuần 16%, mức dùng thêm 0 USD.
