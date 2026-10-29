# Kiểm toán dữ liệu train (train_v0, val_v0, thành phần C) — Fable 5.1

Ngày 2026-10-27. Phạm vi: **chỉ dữ liệu train**; phương pháp/hàm chuyển và đánh giá/chiến lược do hai agent khác lo. Chỉ đọc repo; không chạy model, không train, **không mở test 1.3**; dev 1.3 (`data/bench_v1_3/dev_v1_3.jsonl`, 3.847 dòng) được đọc để so phân bố và kiểm rò rỉ. Mọi số đếm bằng 3 script nhỏ đặt ngoài repo (mục 6); mẫu đọc tay 107 dòng (seed 20261027).

File đã đếm: `data/train_v0.jsonl` (20.000), `data/val_v0.jsonl` (1.000), `results/train_v1/components/C.jsonl` (31.034 = nền 19.953 + A 4.396 + B 6.685), `results/train_v1/part_b_en.jsonl`, `results/train_v1/translated_sonnet/full/ban_dich_s2.jsonl`.

---

## 0. Kết luận

1. **Dữ liệu không "sai" theo nghĩa kỹ thuật** (0 dòng hỏng, 0 target lệch, 0 rò rỉ với dev 1.3 theo quy tắc khử rò rỉ chính thức, nhãn người đọc mẫu sai rõ ≈ 3%, bản dịch Sonnet không đổi nghĩa ở 30/30 cặp đọc). Mức chắc chắn: cao cho các phép đếm, trung bình cho tỉ lệ lỗi nhãn/dịch (n = 30 mỗi nhóm).
2. **Vấn đề nghiêm trọng nhất là KHUNG BÀI (frame) và CƠ CHẾ TRẢ LỜI mà dữ liệu dạy, không phải số dòng.** Bằng chứng đếm được trên C.jsonl:
   - **1.109 dòng tiếng Việt (7,4% VI) không có văn bản để đọc**: health_fact 665 (chỉ một nhận định, trung vị 25 token), boolq-không-đoạn-văn 265 (trung vị 13 token), EXAMS 179 (state chỉ là "Môn X, lớp Y"; 21 câu đòi tra Atlat không có trong state). Chúng dạy model **phán đúng/sai bằng kiến thức sẵn có**. Đây là cơ chế trực tiếp của thói quen "văn bản không liên quan → chọn bác bỏ" (P4: 110/241 ở C so với 26/241 ở M1).
   - **0 dòng train** dùng khung câu hỏi kiểm chứng của bench ("Dựa vào văn bản, nhận định sau… Nhận định: …" / "Văn bản có xác nhận phát biểu sau không?"), trong khi 593 dòng vitaminc dùng **đúng ba chữ nhãn của bench** nhưng trong khung khác (hai văn bản trong state + câu hỏi chung "Hãy chọn nhãn mô tả văn bản đúng nhất"). Model học chữ nhãn mà không học khung: khớp với Pa ViWikiFC −16,8 chỉ ở nhãn "Bị văn bản bác bỏ".
   - **Khối đọc hiểu dịch (1.801 dòng) đặt câu hỏi thật trong state** (đầu hay cuối tùy nguồn: race dòng đầu 665/676, cosmos_qa câu cuối của một đoạn 345/358, dream dòng cuối 335/336, quail dòng cuối 247/354), còn ô câu hỏi là mệnh lệnh chung ở **1.724/1.801 dòng** ("Hãy chọn lựa chọn trả lời câu hỏi chính xác nhất"…); Belebele đặt câu hỏi ở ô câu hỏi. 30,3% dòng VI của C (4.569) có câu hỏi dạng mệnh lệnh chung như vậy.
   - Hai khung bench bị **dùng lại với bộ nhãn khác**: "Văn bản này thuộc chủ đề nào?" (SIB 7 nhãn) ↔ VNTC 1.000 dòng 10 nhãn; "Người dùng muốn trợ lý làm gì?" (MASSIVE 60 ý định) ↔ 1.000 dòng ý định du lịch 8 nhãn cố định, 191 dòng không dấu.
3. **train_v0 phần tiếng Việt quá hẹp để kỳ vọng mục tiêu**: 4.002 dòng VI chỉ có **16 chuỗi câu hỏi khác nhau** (5 mẫu PhoLaya + 11 mẫu khía cạnh), gần như không có văn bản dài (≈0% ≥ 200 token, dài nhất 220; dev VI: 16,1% ≥ 200 token, ViFactCheck trung vị 728 token), chỉ 2/3/4 lựa chọn (dev VI có 7 và 10). Dev VI có hàng trăm câu hỏi riêng theo bài (Belebele); train_v0 không có dòng VI nào như vậy.
4. **Quy mô**: 20.000 dòng × 2 epoch = 1.250 bước (C: 1.938 bước) so với Quyet-Medium 120.000 bước LoRA (dữ liệu không công bố, nền 4B), Jeff 284.747 câu hỏi full fine-tune (có MASSIVE/CLINC150 train), decider-2b ~95 bộ dữ liệu (tổng không công bố; riêng tập hồi quy/đánh giá đã 102.804 dòng). Nhưng bằng chứng nội bộ cho thấy **thêm cùng loại dữ liệu không giúp**: decider-ft-ep2 trên train_v0 (84,9) < decider-2b gốc (86,3); C@25% 81,07 ≈ C@100% 81,65; val loss epoch 1→2 phẳng. Thiếu ở đây là **dòng VI đúng khung bench** (hiện 1.396 dòng VI có câu hỏi riêng theo bài = 9,3% VI của C, 0% ở v0), không phải tổng số dòng.
5. **Khuyến nghị số 1**: dựng lại khối kiểm chứng tiếng Việt về **đúng khung bench** (văn bản ở state, nhận định trong câu hỏi, 3 nhãn bench, NEI ≈ 30%, có cặp "văn bản lạc đề → không đủ thông tin" dựng từ dòng train), **bỏ 1.109 dòng không văn bản**, và **đưa câu hỏi của khối đọc hiểu dịch lên ô câu hỏi**. Không cần dịch thêm, không cần chữ tiếng Việt mới ngoài quy trình duyệt hiện có (chỉ dùng chuỗi đã duyệt). Chi tiết và xếp hạng ở mục 5.

---

## 1. Dữ liệu có ĐÚNG không

### 1.1 Toàn vẹn dòng (đếm hết, không lấy mẫu)

| Phép kiểm | train_v0 | val_v0 | C.jsonl | Mức chắc chắn |
|---|---|---|---|---|
| Lỗi parse JSON | 0 | 0 | 0 | cao |
| Số lựa chọn ngoài 2–10; len(options) ≠ len(target) | 0 / 0 | 0 / 0 | 0 / 0 | cao |
| Tổng target ≠ 1 | 0 | 0 | 0 | cao |
| `target_argmax` lệch với target | 0 | 0 | 0 | cao |
| Dòng target mềm (max < 1) / max < 0,5 | 1.145 / 57 | 64 / 7 | 1.145 / 57 | cao (mềm là chủ ý: votes, score) |
| Lựa chọn trùng nhau trong một dòng | 4 (sciq, ScienceQA, medwiki) | 0 | 5 (+1 VSMRC) | cao |
| State/câu hỏi/lựa chọn rỗng | 0 | 0 | 0 | cao |
| State chưa NFC | 198 (189 PhoLaya) | 6 | 224 (PhoLaya 188, ViANLI 28) | cao; vô hại cho train, nhưng bench đã NFC |
| Ký tự điều khiển/ẩn (U+200B…) trong state | 20 | 0 | 30 | cao; nhỏ |
| Lựa chọn trích chữ cái lựa chọn khác (regex v0, tiếng Anh) | 0 | 0 | 0 | cao |
| **Lựa chọn trích chữ cái bằng tiếng Việt ("Cả B và C", "Đáp án A và B")** | — | — | **21 dòng VSMRC** (bộ lọc v0 chỉ bắt tiếng Anh) | cao. **Vòng train xáo lựa chọn mỗi lần nạp → 21 dòng này thành nhãn sai** |
| Lựa chọn "Tất cả các đáp án trên / Cả ba…" | 57 (cosmos_qa 44, EN) | — | +113 VSMRC (87 là đáp án) | cao; sau xáo vẫn hiểu được, nhưng là mẹo "chọn câu tổng" (7,1% VSMRC) |

### 1.2 Trùng lặp và val

- Văn bản (state chuẩn hoá) khác nhau: train_v0 **19.748/20.000**; VI 3.868/4.002 (ViNLI 1.321/1.330, VSFC 1.286/1.337, ViSFD 1.259/1.333). C: 30.547/31.034; VI 14.706/15.075. Trùng trọn dòng (state+câu hỏi+lựa chọn) 16 (v0) / 33 (C). Cùng state+câu hỏi mà đáp án khác: 6 / 7 (nhãn mâu thuẫn, rất ít). → **Trùng lặp không đáng kể.**
- Phần A và B **0 state trùng nền**.
- train ∩ val: 0 state trùng nguyên văn (đúng như build), nhưng **145/1.000 dòng val chia sẻ một 13-gram với train** (FOL-nli 12, procedural-typed-decisions 62 cộng lại, spartqa 8…): đó là văn bản sinh theo mẫu, không phải rò rỉ nội dung, nhưng val không hoàn toàn độc lập ở các nguồn này. *(Chuyển cho agent đánh giá: val_v0 dùng để fit T và ngưỡng noul; noul trong val bị xáo thứ tự Có/Không lúc dựng — 37 Có trước / 30 Không trước — còn bench giữ cố định Có, Không.)*

### 1.3 Rò rỉ với dev 1.3 (kiểm lại độc lập; không mở test)

Chỉ mục: toàn bộ state, câu hỏi, lựa chọn của dev 1.3 (EN, VI, VI không dấu; 29.175 chuỗi), bỏ 401 chuỗi mẫu lặp ≥ 3 lần (câu hỏi cố định, nhãn). Phép: trùng nguyên văn, chứa 13-gram, chuỗi 4–12 từ nằm trọn trong train. Không chạy MinHash (`datasketch` không có trong Python Windows).

| File | Dòng chạm (mọi phép) | Trong đó chỉ là cụm từ chung trong **lựa chọn Belebele** (4–12 từ) | Theo quy tắc chính thức (lựa chọn Belebele không xét "chứa đoạn ngắn") |
|---|---|---|---|
| train_v0 | 134 | 134 ("in the united states" 108, "united states of america" 5…) | **0** |
| val_v0 | 6 | 6 | **0** |
| C | 384 | 384 ("in the united states" 105, "điện thoại di động" 63, "chương trình truyền hình" 49, "liên minh châu âu" 24…) | **0** |

Kết luận: **0 rò rỉ nội dung với dev 1.3**; các cụm bắt được đều là cụm từ thông dụng. Với test 1.3 tôi không mở; dựa theo `data/bench_v1_3/leakage_v1_3.json`: train_v0 chạm test 2 dòng (contain13, ViNLI ↔ ViFactCheck), và REPORT_giai_doan_0 ghi hai dòng này đã bỏ khỏi nền của C. Mức chắc chắn: cao.

### 1.4 Lệch vị trí đáp án (thứ tự lưu trong file)

- v0, choice k=4: vị trí 0/1/2/3 = 1.485/1.046/900/1.230 (lệch nhẹ về đầu-cuối); VSFC topic 71% ở vị trí 0, ViSFD aspect 71% ở vị trí 3 (là lệch **nhãn** "giảng viên" 480/676 và "không đề cập" 470/664, không phải lệch vị trí).
- C thêm: headline_cause 72% vị trí 0 ("không có quan hệ nhân quả" 350/465), nli_fever "kéo theo" 55% vị trí 0, boolq "Đúng" 65%.
- **Vô hại cho choice** vì `train_lora.py` xáo lựa chọn mỗi lần nạp; P2 (đảo thứ tự) của C 89,6% không xấu hơn M1 88,7%. *(Chuyển cho agent phương pháp: `shuffled()` không có cờ `kind`, nên xáo cả noul (bench cố định Có/Không) và score — PLAN_decisions 12/10 đã ghi "ngừng xáo score" cho train chính thức nhưng C vẫn xáo.)*

### 1.5 Nhãn sai / recast sai / dịch đổi nghĩa (đọc mẫu, seed 20261027)

| Nhóm | n | Sai rõ | Tranh cãi | Ghi chú | Wilson 95% cho "sai rõ" |
|---|---|---|---|---|---|
| PhoLaya v0 (5/tác vụ × 6) | 30 | **1** (`pl:visfd:49191`: bình luận chê loa ngoài "như loa máy tầu" nhưng nhãn "không phàn nàn") | 6 (ViSFD aspect 3: "máy đẹp" nhưng khía cạnh thiết kế = "không đề cập"; VSFC sentiment 2: lời đề nghị "cần tăng tiết thực hành" gán tiêu cực theo quy ước bộ gốc; ViNLI 1) | nhãn người, đúng quy ước bộ gốc nhưng quy ước hẹp | 3,3% [0,6; 16,7] |
| Phần A | 27 | 0 | 2 EXAMS/4 đòi tra Atlat (không trả lời được từ state); VSMRC 3/8 có đoạn văn bị cắt hoặc mất số do mẫu Wikipedia ("diện tích khoảng với…"); VNTC 5/5 chữ thường, mất dấu câu, mất chữ số, lẫn header email | VSMRC nhãn LLM: 8/8 đúng | 0% [0; 12,5] |
| Phần B: EN gốc ↔ VI Sonnet (6/dạng bài) | 30 cặp | **0 đổi nghĩa, 0 đổi đáp án** | 1 giữ nguyên lỗi gốc ("higood morning"), thuật ngữ giữ tiếng Anh hợp lý (SSI, Guarantee Credit) | khớp với báo cáo dịch (đọc lại sửa 5,2% dòng; Opus thấy 3/200 lỗi nghĩa còn lại). Lưu ý: state được cắt ở ~1.400–1.500 ký tự khi đọc, nên đoạn dài (quail, race) chỉ soát phần đầu | 0% [0; 11,4] |
| Nền tasksource EN | 20 | 0 | **3 nhãn không suy được từ văn bản** (stackoverflow "độ phổ biến của câu hỏi"; HatemojiBuild một emoji 🈵 = hate-speech; autotnli văn bản sinh vô nghĩa) | định dạng rất tạp: JSON chính sách, DOM HTML, CSV, logic hình thức | 15% [5; 36] |

Kết luận 1.5: nhãn người của PhoLaya có ~3% sai rõ và ~20% "đúng theo quy ước hẹp của bộ gốc"; bản dịch Sonnet không phát hiện đổi nghĩa; recast v0 không sai cấu trúc; **lỗi recast thật sự là 21 dòng VSMRC "Cả B và C" bị xáo**, và các dòng "không có văn bản" (mục 2.3).

### 1.6 Chuỗi chưa dịch / tiếng Anh còn sót trong dòng VI (đếm hết)

- Dòng VI có ≥ 3 hư từ tiếng Anh trong state: phần B 207/6.685 (nli_fever 86, vitaminc 39, boolq_passage 35, race/high 21): phần lớn là tên tác phẩm, tên báo, trích dẫn giữ nguyên có chủ ý theo hướng dẫn dịch v2. Phần A 23 (VSMRC 13, VNTC 8). Không phải lỗi dịch.
- **Giàn tiếng Anh giữ nguyên trong dòng VI: 2.423 dòng** ("text_A/text_B", "Item A/B/C", "Passage A/B", "Turn 0:"): multidogo 789, nli_fever 660, vitaminc 335, headline_cause 252, sharc 149, boolq_passage 141. Không có trong bench. Vô hại cho nghĩa, nhưng là một "khung" lạ chiếm 16% dòng VI của C.
- Nhãn tiếng Anh trong dòng VI: 0 (khớp báo cáo trước).
- 403 dòng `text_lang=multi` (Hàn, Trung, Tamil, Kannada, Đức, Ả Rập…) dùng khung prompt tiếng Anh; 2 dòng X-CSQA-vi dùng khung VI. Không gây hại, không giúp VI.

---

## 2. Dữ liệu có TỐT / PHÙ HỢP với dev không

### 2.1 Phân bố cơ bản so với dev 1.3 (VI có dấu, bản gốc, 1.430 dòng)

| Chỉ tiêu | dev 1.3 VI | train_v0 VI (4.002) | C VI (15.075) |
|---|---|---|---|
| Số lựa chọn | 2: 302, 3: 241, 4: 294, **7: 299, 10: 294** | 2: 1.333, 3: 661, 4: 2.006, 5: 2; **không có 7, 10** | 2: 3.971, 3: 2.669, 4: 5.631, 8: 1.000, 10: 1.789; **không có 7** |
| Token state p50 / p90 / p99 | 57 / 445 / 1.298 | 41 / 92 / 138 (max 220) | 66 / 342 / 724 |
| Tỉ lệ state ≥ 200 token | **16,1%** | **≈0% (dài nhất 220)** | 20,1% |
| State ≥ 500 / ≥ 700 token | 133 / 95 (ViFactCheck 133/158 ≥ 500) | 0 / 0 | 846 / 172 |
| Câu hỏi khác nhau | hàng trăm (Belebele mỗi bài một câu) + 4 mẫu | **16** | 1.785 (VSMRC 1.217 + EXAMS 179 + 154 mẫu banking/clinc + ~17 mệnh lệnh chung) |
| Dòng VI có câu hỏi riêng theo bài | ~294 Belebele (+ nhận định trong câu hỏi kiểm chứng 241+202) | **0** | 1.396 (9,3%) |
| Dòng không dấu | 1.430 (bản nodiac = 37% toàn dev) | 2 state (0,05%) | 197 state (1,3%; ý định du lịch 191). `strip_p = 0` ở cả M1 và C |
| noul: tỉ lệ "Không" | 62,9% | 59,2% | nền 59,2%, ViANLI 67,6%, B 49,2% |
| Kiểm chứng choice: tỉ lệ NEI | **31,1%** (75/241) | không có khối kiểm chứng VI | **12,8%** (228/1.777; vitaminc 15,2%, nli_fever 19,2%, health_fact 3,9%) |
| Tỉ lệ VI / EN+multi | — | 20,0% / 80,0% | 48,6% / 51,4% |

Riêng về số lựa chọn: dev MASSIVE là 10 lựa chọn lấy mẫu từ 60 ý định; train VI chỉ có 10 lựa chọn ở multidogo-bảo-hiểm (789) và VNTC (1.000), đều **bộ nhãn cố định**, không phải lấy mẫu ngẫu nhiên từ kho nhãn lớn như bench.

### 2.2 Khung câu hỏi: train so với bench, theo nguồn bench

| Nguồn bench (VI có dấu, điểm dự án / decider) | Khung bench | Khung tương ứng trong C | Đếm | Nhận xét về tính có hại |
|---|---|---|---|---|
| ViFactCheck 72,8 / 83,5; ViWikiFC 76,0 / 82,2 | state = bài báo/đoạn Wikipedia (p50 728 / 103 token); câu hỏi chứa nhận định; 3 nhãn, NEI có đuôi "để kết luận" | vitaminc 593: state = "Văn bản thứ nhất / thứ hai" hoặc "text_A/text_B" (p50 76 token), câu hỏi chung, 3 nhãn bench **không** đuôi; nli_fever 598: nhãn NLI "kéo theo/trung lập/mâu thuẫn"; health_fact 586: **chỉ nhận định**, 4 nhãn đúng/sai/lẫn lộn/chưa chứng minh | 0 dòng khung bench; 593 dòng chữ nhãn bench; 1.153 dòng khung hai văn bản; 581 dòng chỉ nhận định | **Có hại, bằng chứng cao**: học chữ nhãn ngoài khung (Pa −16,8 chỉ ở "bác bỏ"); học phán nhận định không cần văn bản (P4 chọn bác bỏ 110/241); NEI 12,8% so với 31,1% ở dev; **0 dòng** dạng "văn bản lạc đề → NEI" (vitaminc luôn cùng chủ đề) |
| MASSIVE 86,7 / 96,3 | "Người dùng muốn trợ lý làm gì?", 10 mô tả ý định lấy mẫu từ 60 | cùng câu hỏi, **8 nhãn cố định** miền hành chính/du lịch (1.000, mẫu câu, 191 không dấu); multidogo 10 nhãn bảo hiểm (789, giàn "text_A: Turn 0:"); banking77/clinc chỉ noul "Có phải X là nhãn đúng…" (325) | 1.000 + 789 + 325 | Trung bình: cùng câu hỏi nhưng kho nhãn hẹp; không có nguồn nào mang 60 ý định kiểu MASSIVE (bị cấm: MASSIVE/SLURP/IntentGrasp/HWU64). MASSIVE có dấu chỉ +1,4; không dấu +7,8 nhờ 191 câu không dấu |
| Belebele 82,3 / 87,8 | đoạn + câu hỏi riêng ở ô câu hỏi + 4 lựa chọn | VSMRC 1.217 **đúng khung** (nhãn LLM; **đáp án đúng là lựa chọn dài nhất 45%** so với 22% ở Belebele dev; đáp án có độ trùng từ với đoạn cao nhất 28% so với 12%); đọc hiểu dịch 1.801 **câu hỏi nằm trong state**, ô câu hỏi là mệnh lệnh chung | 1.217 đúng khung / 1.801 sai khung | Trung bình: VSMRC dạy mẹo "dài nhất / trùng từ nhất" (khớp Opus đọc mù: 28% câu C mới đúng là "khớp bề mặt"); khối dịch dạy đọc câu hỏi ở chỗ khác với bench |
| XCOPA 78,0 / 86,0 | tiền đề ở state, câu hỏi "Nguyên nhân/Kết quả nào hợp lý hơn?", 2 lựa chọn đều hợp lý bề mặt | e-CARE 491: tiền đề + "Nguyên nhân của việc này là gì?" **trong state**, ô câu hỏi mệnh lệnh chung; lựa chọn sai không chung từ với tiền đề 39,7% (XCOPA dev 39,0%, tương đương) | 491 | Thấp: C tăng XCOPA +7,0 (n = 100, p = 0,12); độ khó nhiễu tương đương |
| SIB-200 82,6 / 82,3 | "Văn bản này thuộc chủ đề nào?", 7 nhãn | VNTC 1.000: **cùng câu hỏi, 10 nhãn khác**, hai nhãn trùng chữ (sức khỏe, thể thao), văn bản báo 2007 chữ thường, không dấu câu, không chữ số, p50 473 token | 1.000 | Trung bình: SIB −2,7 ở đúng hai nhãn trùng chữ (chưa có ý nghĩa, p = 0,20); văn bản VNTC chất lượng thấp |
| Có/Không (ViFactCheck/ViWikiFC noul) | "Văn bản có xác nhận phát biểu sau không? Phát biểu: …", Có/Không cố định | ViNLI/ViANLI "Câu nói có được suy ra từ đoạn văn không?" 1.662 (state "Đoạn văn:/Câu nói:"); label_verification "Có phải "X" là nhãn đúng cho ví dụ này không?" 421; "…là câu trả lời đúng cho câu hỏi không?" 90 | 0 dòng khung bench | Trung bình: khung gần (suy ra ↔ xác nhận) nhưng ViANLI 1.000 dòng lệch 67,6% Không; C tụt Có/Không không dấu −7,9 chưa quy được nguyên nhân |

### 2.3 Dòng "không có văn bản để đọc" (đếm hết trên C, phần A+B)

| Nguồn | Dòng | Trung vị token state | Không xuống dòng | Bản chất |
|---|---|---|---|---|
| health_fact | 665 (586 choice, 70 noul, 9 score) | 25 | 597 | nhận định y tế / tiêu đề báo; nhãn đúng/sai theo PolitiFact-style, **không có bằng chứng** |
| super_glue/boolq (không đoạn văn) | 265 | 13 | 236 | câu hỏi yes/no trần ("tinh bột ngô có giống siro ngô không") |
| exams/vi | 179 | 9 | 179 | đề thi Sử/Địa/GDCD; 21 câu "Căn cứ vào Atlat…" |
| **Tổng** | **1.109 = 7,4% dòng VI của C** | | | dạy "trả lời bằng kiến thức, không cần văn bản" |

Cột "không xuống dòng" là số đếm trực tiếp (health_fact 597/665, boolq 236/265, EXAMS 179/179); số dòng còn lại của health_fact/boolq có xuống dòng nhưng vẫn chỉ là nhận định hoặc câu hỏi (trung vị 25 và 13 token), nên tổng 1.109 là "không có văn bản làm căn cứ", không phải "một dòng duy nhất".

Đối chiếu dev: mọi câu bench đều có văn bản làm căn cứ (P3 bỏ văn bản của M1 chỉ còn 31–46 ở kiểm chứng). Đây là chỗ **lệch khung có hại rõ nhất** và là nguyên nhân kiểm được của P4.

### 2.4 Nhãn người so với nhãn máy / sinh theo mẫu

| Khối | Dòng | Nguồn nhãn |
|---|---|---|
| Nền v0 tasksource | 15.961 | không gắn cờ; theo tên nguồn: **~26% (5.255) là dữ liệu sinh theo chương trình hoặc NLI recast tự động** (procedural-typed-decisions 1.730 câu đố JSON/CSV, FOL-nli 264, doc-nli 261, webinstruct 470 do LLM, spartqa, babi, ruletaker…); 864 dòng là ưu tiên/RLHF (hh-rlhf, oasst2, HelpSteer…); 1.097 dòng (6,9% EN) có câu hỏi mệnh lệnh chung "Which of the supplied criteria best matches the state?" / "Select the label that best applies to the state." |
| Nền PhoLaya | 3.992 | người gán (UIT), quy ước hẹp (mục 1.5) |
| A | 4.396 | người gán 2.179 (VNTC, ViANLI, EXAMS), **LLM sinh 1.217 (VSMRC)**, **mẫu câu 1.000 (ý định)** |
| B | 6.685 | người gán theo bộ gốc; bản dịch Sonnet (đầu ra model, người dùng chấp nhận nội bộ) |

Tỉ lệ nhãn/văn bản không do người: VI của C ≈ (1.217 + 1.000)/15.075 = 14,7% ở phần nhãn, cộng 6.685 dòng văn bản dịch máy (44% VI). Dev 1.3: nhãn nguồn người 100% (ViFactCheck, ViWikiFC, Belebele, SIB, MASSIVE, XCOPA), bản dịch người.

### 2.5 Tỉ lệ dạng bài (ước theo tên nguồn) so với dev

| Dạng bài | dev 1.3 (cặp VI) | train_v0 VI | C VI |
|---|---|---|---|
| Đọc hiểu đoạn + câu hỏi | 294 (Belebele) | 0 | 3.018 (VSMRC 1.217 đúng khung; dịch 1.801 sai khung) + 179 EXAMS không đoạn |
| Chủ đề | 299 | 676 (VSFC 4 lớp phản hồi sinh viên) | 1.676 |
| Ý định | 294 | 0 | 2.114 |
| Nhân quả | 100 | 0 | 989 |
| Kiểm chứng choice | 241 | 0 (666 ViNLI relation gần) | 1.988 (0 đúng khung) |
| Có/Không | 202 | 1.333 | 3.124 |
| Cảm xúc/khía cạnh (không có trong dev) | 0 | 1.325 | 1.325 |
| Score (không có trong dev 1.3) | 0 | 0 (EN 1.463) | 47 VI (EN 1.463) |

Nền EN 15.961 dòng, 637 nguồn (C tổng 642 nguồn), 253 nguồn < 10 dòng: rất rộng nhưng mỗi nguồn chỉ vài chục dòng; 1.463 dòng score (7,3%) phục vụ loại câu không có trong dev 1.3.

---

## 3. So với dữ liệu của decider-2b, Quyet, Jeff: quy mô có đủ không

Nguồn: `results/baselines/notes_model_cards.md` (decider-2b), `results/baselines_v1_2/notes_model_api.md` (Quyet, Laya, Jeff). Chỉ dùng số card ghi.

| Model | Nền | Dữ liệu train (theo card) | Quy mô có thể đếm | VI có dấu dev 1.3 |
|---|---|---|---|---|
| Dự án M1 | Qwen3-1.7B | train_v0: 20.000 dòng (16.000 EN, 4.000 VI), 637 nguồn, trần 5% | 2 epoch = **1.250 bước** batch 32 | 79,9 |
| Dự án C | Qwen3-1.7B | 31.034 dòng (15.075 VI) | **1.938 bước** | 81,65 |
| decider-2b v11 | Qwen3.5-2B-Base | "~95 bộ dữ liệu công khai" (intent, topic, NLI, fact verification, RC, MCQA, thang điểm, ưu tiên, tool), + agent trajectories, Mind2Web, câu teacher viết; 1 epoch + các epoch tiếp; v11 LoRA r64 2 epoch trên 42.749 dòng; `massive-en-US trained: true`; card "English only" | Không công bố tổng mixture (lớn hơn nhiều so với dự án; tập hồi quy/đánh giá trên các split test của 61 tác vụ đã 102.804 dòng) | 86,5 |
| Quyet-1.0-Medium | Qwen3.5-4B + LoRA r16 merged | **không nêu**; "English; also tuned for Vietnamese" | `source.step = 120.000` (≈ 60–100× số bước của dự án) | 89,6 |
| Jeff-Qwen3.5-2B | Qwen3.5-2B full FT | 284.747 câu hỏi, 1 epoch; có **train split MASSIVE + CLINC150**, WinoGrande 40k, RAGTruth 15k, ~50k câu tổng hợp bởi teacher; "English and text only" | 284.747 | bị chặn, chưa chấm |

Nhận định:
1. **Về số bước/số dòng, dự án nhỏ hơn 1–2 bậc**; nhưng cả ba model đối chứng không công bố dữ liệu tiếng Việt (Quyet không nêu; decider/Jeff ghi English-only). Điểm VI cao của chúng nhiều khả năng đến từ **nền mạnh hơn (Qwen3.5-2B-Base/4B) + kho dạng bài rộng bằng tiếng Anh chuyển giao sang VI**, không phải từ dữ liệu VI lớn. Mức chắc chắn: trung bình (suy từ card).
2. **Bằng chứng nội bộ cho thấy tăng cùng loại dữ liệu không đóng khoảng cách**: decider-ft-ep2 (train_v0 trên decider) 84,9 < decider-2b 86,3; C@25% ≈ C@100% (81,07 so với 81,65); val loss hai epoch 0,758 → 0,757.
3. **Số học mục tiêu**: 84,0 − 79,9 = 4,1 điểm trung bình đều = 24,6 điểm cộng trên 6 nguồn. Đòn bẩy dữ liệu hiện có theo nguồn:

| Nguồn | Khoảng cách tới decider | Đòn bẩy dữ liệu được phép | Ước lượng còn lấy được |
|---|---|---|---|
| ViFactCheck −10,8; ViWikiFC −6,3 | 17,1 | chỉ dịch (vitaminc, nli_fever, fever-evidence-related, hover, wice, scifact đều trong tasksource) + dựng cặp lạc đề → NEI; nguồn bản địa bị cấm/không rõ giấy phép | lớn nhất: NEI của ViFactCheck choice hiện đúng 20,6%; nếu lên 60% ≈ +8 ViFactCheck ≈ +1,4 trung bình đều (ước của FABLE_doc_ket_qua) |
| MASSIVE −9,5 | 9,5 | banking77 (77 nhãn) và clinc (151 nhãn) **đã có bản dịch cho noul**, nhưng bộ lọc ≤ 10 lựa chọn của v0 loại dạng choice; recast "nhãn đúng + 9 nhiễu lấy mẫu" (cách decider-2b làm) cần ~230 chuỗi nhãn dịch (ngắn) | trung bình; MASSIVE có dấu của C +1,4 chưa có ý nghĩa |
| XCOPA −8,0 | 8,0 | e-CARE 491 đã có; kho e-CARE còn 739 − 500 | C đã +7,0 (n = 100) |
| Belebele −5,4 | 5,4 | VSMRC còn ~10.000 câu nhưng vướng trần 5% và mẹo dài nhất; đọc hiểu dịch 1.801 cần đổi khung; bộ UIT RC (ViMMRC, ViQuAD) cần thỏa thuận | trung bình |
| SIB-200 ngang | 0 | VNTC gây −2,7 (chưa có ý nghĩa) | giữ hoặc bỏ |

Trả lời câu 3: **không phải thiếu quy mô theo nghĩa số dòng**; thiếu dòng VI **đúng khung bench, có văn bản, nhãn người, dài**. Với các nguồn được phép hiện có, lợi ích ước được (~+1,4 kiểm chứng, +0,45 SIB nếu bỏ VNTC, +1,2 XCOPA nếu giữ e-CARE) cộng +1,77 của C cho vùng **83–84**, sát ngưỡng; để chắc qua 84 cần thêm đòn bẩy ở MASSIVE (recast banking/clinc) hoặc nền mạnh hơn (ngoài phạm vi dữ liệu). Mức chắc chắn: trung bình.

---

## 4. Những điểm khác cần ghi

- **Val cho train chính thức**: val_v0 không có dòng nào thuộc các dạng bài mới (đọc hiểu, ý định, kiểm chứng VI), nên val loss của C không đo được phần dữ liệu mới. Nên có val_v1 tách theo văn bản từ A/B (giao agent đánh giá).
- **Không dấu**: dev có 1.430 dòng không dấu (37%); train chỉ 1,3% VI không dấu và `strip_p = 0`. strip_v0 trước đây không đạt (p = 0,3), nhưng C vẫn tăng không dấu +1,6 nhờ 191 câu ý định không dấu. Đây là quyết định phương pháp, ghi nhận ở đây vì dữ liệu hiện không cung cấp bản không dấu.
- **Giấy phép theo dòng (C)**: commercial 12.130 + commercial có ghi chú 2.168 + CC0/MIT 989; unspecified 7.563; non-commercial 1.964 (quail 364 ở B); research-only UIT 3.992 + ViANLI 1.000 (gốc CC BY-NC-SA); VNTC 1.000 không rõ; VSMRC 1.217 thẻ Apache trên văn bản Wikipedia CC BY-SA; bản dịch Sonnet 6.685 (điều khoản Anthropic, người dùng chấp nhận nội bộ). Model nội bộ đã chốt nên không chặn; nếu sau này công bố: bỏ UIT/ViANLI/non-commercial/VNTC và xem lại khối dịch (≈ 14.000 dòng, trong đó 10.100 VI).

---

## 5. Khuyến nghị chiến lược dữ liệu, xếp theo lợi ích/chi phí

Lợi ích = điểm trung bình đều ước được trên dev 1.3 (± nhiễu một seed ~1); chi phí = GPU/dịch/duyệt chữ tiếng Việt. Mọi chuỗi VI mới vẫn đi qua `config/vi_proposals.jsonl`. Nền 16.000 EN được giữ theo quyết định người dùng; khuyến nghị 7 đề nghị xem lại điều đó, không tự đổi.

| # | Việc | Nguồn / số dòng | Lợi ích ước | Chi phí | Giấy phép | Cần chữ VI mới? |
|---|---|---|---|---|---|---|
| **1** | **Dựng lại khối kiểm chứng VI về đúng khung bench**: văn bản ở state, nhận định trong câu hỏi (dùng đúng câu hỏi bench đã duyệt), 3 nhãn bench **kể cả đuôi "để kết luận"**, nâng NEI lên ~30%; thêm 200–300 cặp "văn bản lạc đề → NEI" bằng cách ghép nhận định với văn bản của **dòng train khác** (không chạm dev/probe); **bỏ health_fact 665** | vitaminc 663 + nli_fever 660 đã dịch (đổi khung, không dịch lại); bỏ 665; +250 cặp ghép | +1 đến +1,5 (ViFactCheck NEI 20,6% → ~60%; sửa P4, Pa) | thấp: script, 0 dịch, ~3 giờ GPU train thử | như hiện có | **Không** (dùng câu hỏi/nhãn bench đã duyệt) |
| **2** | **Bỏ 444 dòng còn lại không có văn bản**: boolq-không-đoạn-văn 265, EXAMS 179 (hoặc giữ EXAMS nhưng bỏ 21 câu Atlat và gắn cờ "kiến thức") | −444 | +0,3 đến +0,5 (giảm thói quen phán bằng kiến thức) | rất thấp | — | Không |
| **3** | **Đưa câu hỏi của khối đọc hiểu dịch lên ô câu hỏi**, quy tắc **theo nguồn** (đếm trên C): race/high, race/middle tách **dòng đầu** (665/676 có "?" hoặc "_"); dream tách **dòng cuối** (335/336); quail tách **dòng cuối** (247/354 nhiều dòng; 39 dòng một đoạn cần tách câu cuối; ~70 dòng không có dấu hỏi phải kiểm tay); cosmos_qa tách **câu cuối** của đoạn duy nhất (345/358 kết thúc bằng "?") | 1.724 dòng đổi cấu trúc, không dịch lại | +0,3 đến +0,8 ở Belebele | thấp: script + kiểm tay 30 dòng **trải đủ 5 nguồn** | — | Không |
| **4** | **Sửa VSMRC**: bỏ 21 dòng "Cả B và C"; hoặc gắn cờ không xáo cho dòng có lựa chọn tổng hợp; cân bằng độ dài lựa chọn (loại dòng mà đáp án dài hơn mọi nhiễu > 1,5 lần) | −21 … −300 | +0,2 (bớt mẹo dài nhất) | rất thấp | Apache/CC BY-SA | Không |
| **5** | **Recast ý định banking77 + clinc_oos thành choice 10 lựa chọn** (nhãn đúng + 9 nhiễu lấy mẫu từ 77/151 nhãn, cách decider-2b dùng), thay cho chỉ noul; nhãn 9 clinc đã rà | ~1.000–1.500 dòng mới; **154/228 mô tả nhãn đã có bản dịch** trong câu hỏi của 325 dòng noul hiện có (banking77 70, clinc 84), chỉ còn ~74 nhãn cần chuỗi mới; câu mẫu người dùng của 325 dòng đã dịch | +0,5 đến +1,0 ở MASSIVE có dấu (kho nhãn rộng, lựa chọn lấy mẫu như bench) | thấp–trung bình: dịch ~74 nhãn ngắn, kiểm trùng MASSIVE, duyệt ~74 chuỗi; nên xếp ngay sau 1–4 | CC BY 4.0 / CC BY 3.0 | **Có** (~74 nhãn) |
| **6** | **Đổi câu hỏi VNTC** để không trùng SIB (hướng "chuyên mục báo"), hoặc bỏ VNTC | 1.000 đổi 1 chuỗi, hoặc −1.000 | +0,3 đến +0,45 (SIB về mức M1) | rất thấp | VNTC không rõ | Có (1 chuỗi) nếu đổi |
| **7** | **Xem lại quy tắc "giữ 16.000 EN"**: cắt nền EN xuống ~10.000 bằng cách bỏ các nguồn xa bench (procedural-typed-decisions 1.730, ưu tiên/RLHF 864, score 1.463, weblinx/DOM, code), giữ NLI/RC/QA/intent/topic/fact-check | −4.000 … −6.000 EN | không rõ dấu (+0,5 nếu VI được học sâu hơn ở cùng số bước; rủi ro EN −1) | thấp, nhưng mất so sánh với các lần trước | — | Không |
| 8 | Dịch thêm kiểm chứng có **bằng chứng dài** (fever-evidence-related, hover, wice, scifact_entailment trong tasksource) để khớp ViFactCheck p50 728 token | ~1.000 dòng, ~300 nghìn token EN | +0,5 ở ViFactCheck (độ dài) | trung bình: một vòng dịch Sonnet (~15% cửa sổ 5 giờ), khử rò rỉ | commercial/unspecified | Không (dùng nhãn bench) |
| 9 | Bản không dấu dựng sẵn (thay `strip_p`) cho khối VI mới, gắn cờ để chấm riêng | +3.000 dòng (20% VI) | không dấu +1–3; có dấu ±0 (strip_v0 không đạt) | thấp | — | Không |
| 10 | Mở rộng PhoLaya (ViNLI/ViSFD/VSFC còn 7.000–77.000 câu) | +3.000 | ~0 (dạng bài không có trong dev; decider-ft-ep2 là bằng chứng) | thấp | research-only UIT | Không |
| 11 | Bộ UIT đọc hiểu (ViMMRC 1.0/2.0, ViQuAD 2.0) | +2.000 | +0,5–1 ở Belebele (nhãn người, đúng khung) | cao: email/thỏa thuận | research-only | Không |

Thứ tự thực hiện đề nghị: **1 + 2 + 3 + 4 + 6 gộp thành một bản C′ (không cần dịch, chỉ 1 chuỗi VI mới nếu đổi VNTC), train thử 1 seed ~3 giờ, chấm dev + probe (P4, Pa, P5) theo tiêu chí FABLE_doc_ket_qua mục 4.3**; nếu C′ qua P4/Pa nhưng vẫn < 84 thì mới mở 5 (rẻ, ~74 chuỗi) và 8 (cần dịch). Không khuyến nghị tăng tổng số dòng trước khi sửa khung.

Đánh đổi của khuyến nghị 1 phải ghi rõ: C′ train trên **đúng câu hỏi và chữ nhãn bench đã duyệt** (tiền lệ: VNTC và ý định du lịch đã dùng lại câu hỏi bench), nên điểm dev sau đó đo khả năng **trong khung**; P5 (đổi câu hỏi) và Pa (bỏ dấu lựa chọn) phải được báo cùng C′ làm hàng rào khái quát hoá. Giả thuyết kiểm được, chuyển agent đánh giá: nhãn mà model chọn thiếu nhất ở dev (NEI, đúng 20,6–43,9%) cũng là nhãn duy nhất có chữ ở dev ("Văn bản không đủ thông tin **để kết luận**") khác chữ đã học ở train ("Văn bản không đủ thông tin", 593 dòng); hai nhãn còn lại trùng chữ hoàn toàn. Chưa phải phát hiện, cần kiểm bằng probe đổi chữ nhãn.

---

## 6. Phương pháp, file, giới hạn

- Script (ngoài repo): `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\audit\a1_stats.py` (toàn vẹn, phân bố, khung), `a2_leak.py` (rò rỉ với dev 1.3), `a3_sample.py` (mẫu đọc tay); kết quả `a1_stats.json`, `a2_leak.json`, `sample_vi_native.txt`, `sample_translation.txt`, `sample_en.txt` cùng thư mục. Chuẩn hoá văn bản chép từ `scripts/decontam.py::normalize`.
- Đọc: `data/manifest_train_v0.json`, `scripts/build_train_v0.py`, `scripts/train_lora.py` (chỉ phần nạp dữ liệu), `scripts/common.py`, `config/train_format.json`, `config/prompt_frames.json`, `config/bench_v1_questions.json`, `data/bench_v1_3/manifest_dev_v1_3.json`, `leakage_v1_3.json`, `results/train_v1/REPORT_giai_doan_0.md`, `REPORT_train_C.md`, `phan_tich_C/FABLE_doc_ket_qua.md`, `doc_loi_opus.md`, `translated_sonnet/REPORT_dich_day_du.md`, `decontam_vi_stats.json`, `results/train_v1_survey/REPORT_sources.md`, `results/BRIEF_train_chinh_thuc.md`, `results/baselines/notes_model_cards.md`, `results/baselines_v1_2/notes_model_api.md`, `REPORT_baselines_v1_2.md`, `PLAN_decisions.md`.
- Giới hạn: không chạy MinHash (thiếu `datasketch`); phân loại dạng bài của nền EN theo tên nguồn (ước); tỉ lệ lỗi nhãn/dịch từ mẫu 30 dòng mỗi nhóm nên khoảng tin cậy rộng; mọi ước lượng "lợi ích" là ngoại suy từ số của C, chưa kiểm bằng train.
- Chuyển cho agent khác (một dòng mỗi mục, không điều tra thêm): (a) phương pháp: `shuffled()` trong `train_lora.py` không có cờ `kind`, xáo cả noul và score; (b) đánh giá: val_v0 noul bị xáo Có/Không lúc dựng và 145/1.000 dòng val chia 13-gram với train; val_v0 không chứa dạng bài mới của C.
