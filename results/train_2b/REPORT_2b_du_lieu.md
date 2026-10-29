# Dữ liệu 2b: C′ và khối kiểm chứng KC-Wiki (28/10/2026)

Báo cáo này làm theo `results/BRIEF_2b_du_lieu.md`. Người dùng duyệt lúc khoảng 08:00 ("được, sang 2b đi"), gồm cả việc tải Wikipedia tiếng Việt.

**Phạm vi:**
- Chỉ dùng CPU, không chạy model nào. Tokenizer Qwen3-1.7B chỉ dùng để đếm token.
- Không train. Test 1.3 chỉ dùng để dò khử trùng, không in nội dung.
- Chỉ tải 4 file parquet của Wikipedia vi.
- Không viết chữ tiếng Việt mới. Câu hỏi và lựa chọn chép từ `data/bench_v1_3/dev_v1_3.jsonl`. Nhận định là câu trích từ bài hoặc câu trích đã đổi cơ học.
- Không sửa `config/`, `data/bench_v1_3/`, `scripts/train_lora.py`. Không ghi đè C.jsonl. Không xóa, không commit.

## Kết luận: cổng D1–D8 (đúng bảng đã chốt trước)

| Mã | Kiểm | Số | Ngưỡng | Kết quả |
|---|---|---|---|---|
| D1 | Chữ nhãn (`check_label_strings.py`) | **0** chuỗi gần trùng mà lệch ở nhãn kiểm chứng, Có/Không và NEI (C có 947 lần). Chuỗi gần trùng khác: 82 trên C′+KC, **không thêm** chuỗi nào so với C (C có 104). KC: 0 | 0 lệch; chỉ được bớt | **Đạt** |
| D2 | Kho chuỗi | 0 vi phạm trên 4.400 dòng KC (gồm 300 dòng val), 1.160 dòng 2d và 250 dòng 2e. Chuỗi NEI thêm ở 2h (202 dòng) đúng chuỗi dev. **0 chuỗi khung mới** | 0 | **Đạt** |
| D3 | Khử trùng (KC + val + 2e có Wikipedia) | **0 chạm** trên 4.525 dòng (state, nhận định, từng câu của state) | 0 | **Đạt** |
| D4 | Probe chỉ đọc nhận định (KC choice, 2.286 dòng) | accuracy 5-fold **0,370**; đa số 0,402 (−3,2 điểm). Chỉ dòng có dấu: 0,378 so với 0,405 | ≤ đa số + 10 | **Đạt** |
| D5 | Thống kê theo nhãn (KC choice có dấu) | trung vị token nhận định 38 / 38 / 37 (lệch 2,7%); phủ định 8,1% / 7,6% / 7,2% (lệch 0,9 điểm). S-lược / S = 37,3% | ≤ 20%; ≤ 2 điểm; ≥ 1/3 | **Đạt** |
| D6 | Trùng từ theo nhãn (chỉ báo) | KC: trung vị Xác nhận 1,00 / Bác bỏ 0,94 / NEI 0,23. proxy_v1: 0,68 / 0,67 / 0,50 | không chặn | **Xem giới hạn 1** |
| D7 | Phân bố nhãn | Khối kiểm chứng VI choice (2d + 2e + KC mix, 3.050 dòng): **NEI 36,1%**. Noul KC "Có" **38,7%** (dev 37,1%) | NEI ≥ 30%; "Có" 37–40% | **Đạt** |
| D8 | Mẫu đọc mù | `mau_doc.jsonl`, 310 dòng: 30 × 6 kiểu KC, 30 dòng 2d, 30 dòng 2f, 30 dòng 2g, 20 × 2 kiểu 2h | xuất file | **Đã xuất**, chờ điều phối đọc |

Worker chỉ báo số, không tự quyết bước tiếp.

- **Điểm đáng đọc nhất là D6 (giới hạn 1).** Theo đúng luật dựng, trùng từ giữa nhận định và văn bản gần như tách hẳn ba nhãn KC, xa với phân bố của dữ liệu thật.
- **D8 chưa có tỉ lệ sai.** Tỉ lệ sai rõ ở mẫu đọc (ngưỡng tiền lệ ≤ 5%) chờ điều phối đọc mù.

## 1. Tải Wikipedia

- **Nguồn:** `wikimedia/wikipedia` subset `20231101.vi`, revision `b04c8d1ceb2f…`, tổng 729.557.588 byte, tải trong 41 giây.
- **Thư mục:** `results/train_v1/raw_refs/wikimedia__wikipedia_20231101.vi/`, thư mục mới.
- **Cách tải:** dùng `huggingface_hub`, `allow_patterns="20231101.vi/*"`, không chạy code nào của repo dataset.
- **Kiểm sha256:** sha256 từng file khớp LFS oid (`ec53cc98…`, `d1e724eb…`, `f364415c…`, `0f5c15b2…`). Ghi ở `manifest_tai_wiki.json` và `manifest_2b.json`.
- **Script đọc:** để ở `scripts/train_2b/`, chạy bằng `python -I`.

## 2. C′ (mục 2 của brief)

Đầu vào là `C.jsonl`, sha `005ca3d5…`, đã kiểm. **Phần EN (15.556 dòng) và multi (403 dòng) giữ nguyên.**

| # | Việc | Kết quả |
|---|---|---|
| a | Bỏ nguồn không có văn bản | health_fact −665, boolq không đoạn văn −265 (giữ boolq_passage), exams/vi −179. Chỉ bỏ dòng VI, vì "phần EN giữ nguyên" |
| b | Bỏ VNTC | −1.000 |
| c | Bỏ dòng teacher bất đồng tự tin | Quy tắc chọn ra 146 dòng: 47 đã bị bỏ ở (a), (b); **7 giữ lại theo ngoại lệ** (nhãn "trung lập"); **bỏ thêm 92** |
| i | VSMRC | −21 có lựa chọn trỏ chữ cái ("Cả A và B"…); −117 có lựa chọn trỏ vị trí ("Tất cả các đáp án trên", "Không có đáp án nào đúng"…); −198 có đáp án dài hơn mọi nhiễu quá 1,5 lần (đếm ký tự). **Còn 881/1.217** |
| d | vitaminc, nli_fever về khung bench | **1.160 dòng** (1.132 choice, 28 noul). Bỏ: 82 dòng packed (Item A/B…), 26 dòng score, 21 dòng noul không suy ra chắc Có/Không, 22 dòng mất một vế (nli_fever có `text_A` rỗng). Nhãn choice: xác nhận 643, bác bỏ 319, NEI 170 |
| f | Đưa câu hỏi đọc hiểu lên ô câu hỏi | Đã tách: race 663/676, dream 335/336, cosmos_qa 342/358, quail 319/354. Không tách chắc (giữ nguyên): 13, 1, 16, 35 |
| g | quail: sửa chữ nhãn NEI | 354/354 dòng: "không đủ thông tin" → "Văn bản không đủ thông tin để kết luận" |
| h | Tăng cường "từ chối" | Đủ điều kiện 1.946 dòng (bỏ 658 dòng đã có lựa chọn kiểu NEI hoặc "không lựa chọn nào", gồm toàn bộ quail sau 2g). Chọn 10%: **151 `abstain_add`, 51 `abstain_replace`** |
| e | Ghép lạc đề → NEI | **250 dòng choice**: 125 dùng bằng chứng của dòng 2d khác, 125 dùng 1–3 đoạn Wikipedia của bài "cho". Bài "cho" đã khử trùng và không dùng ở KC. Văn bản ghép không chung số hoặc tên với nhận định, trùng từ < 0,2 |

**VSMRC: tỉ lệ đáp án là lựa chọn dài nhất** (đếm ký tự; điều phối yêu cầu thêm, brief mục 7b). Hai cách đếm: "kể cả hòa" tính cả khi đáp án dài bằng lựa chọn dài nhất; "dài nhất duy nhất" chỉ tính khi không có lựa chọn nào dài bằng.

| Tập | n | Kể cả hòa | Dài nhất duy nhất |
|---|---|---|---|
| VSMRC trong C | 1.217 | 52,8% | 37,6% |
| VSMRC trong C′, lựa chọn trước 2h (bỏ 24 dòng `abstain_replace`) | 857 | 47,8% | 27,7% |
| VSMRC trong C′ sau 2h | 881 | 45,4% | 27,2% |
| Belebele dev (VI) | 294 | 27,6% | 20,1% |

Con số 22% của điều phối nằm giữa hai cách đếm của mình trên Belebele dev. Sau lọc, VSMRC vẫn nghiêng về đáp án dài hơn Belebele khoảng 7 điểm (dài nhất duy nhất) đến 20 điểm (kể cả hòa).

**Cách làm 2d:**
- State = bằng chứng (vitaminc: văn bản thứ hai; nli_fever: text_A). Câu hỏi = tiền tố của dev + nhận định.
- Ánh xạ nli_fever, ghi vào manifest như một quy ước: kéo theo → xác nhận; trung lập → NEI; mâu thuẫn → bác bỏ.
- Đổi cơ học: bỏ khoảng trắng trước dấu câu (bản dịch giữ kiểu "nữ ." theo token hóa).
- Noul hỏi nhãn ("Có phải "X" là nhãn đúng…"):
  - Trả lời "Có" thì nhãn = X.
  - Trả lời "Không" với X = xác nhận thì đáp án khung dev là "Không".
  - Trả lời "Không" với X là nhãn khác thì không suy ra chắc được, nên bỏ.

**Cách làm 2f:** quail thường để câu hỏi **dính sau câu cuối của đoạn cuối**, không xuống dòng riêng. Vì vậy luật quail là: tách dòng cuối nếu dòng đó chỉ là câu hỏi, không thì tách câu cuối. Với quail, câu hỏi kết thúc bằng "?" hoặc ":" (câu hỏi điền tiếp, ví dụ "…, tác giả có lẽ:"). Với nguồn khác, câu hỏi phải kết thúc bằng "?" hoặc có "_".

**Số dòng:** C′ có 28.596 dòng (VI 12.637), C có 31.034 (VI 15.075). Mỗi dòng bị bỏ có ghi lý do trong `manifest_2b.json` → `dong_bi_bo`. Tổng 2.688 dòng bị bỏ.

## 3. Khối KC-Wiki (mục 3 của brief)

**Lọc bài.**
- **Văn xuôi:**
  - Cắt bài ở mục "Chú thích", "Tham khảo", "Liên kết ngoài", "Xem thêm"…
  - Bỏ tiêu đề mục, danh sách, bảng.
  - Mỗi đoạn văn xuôi ≥ 12 chữ, kết thúc bằng dấu câu.
- **Bài đạt:** ≥ 150 chữ văn xuôi và ≥ 3 đoạn.
- **Luật loại bài sơ khai do bot:** lấy 2 câu đầu của đoạn đầu, che số thành "0" và che chữ viết hoa đầu thành "X". Mẫu này trùng với > 500 bài khác thì loại.
- **Thêm:** loại bài có tiêu đề trùng 73 tiêu đề của ViWikiFC.
- **Kết quả:** 1.288.680 bài.
  - Loại 1.128.457 bài ngắn hoặc dưới 3 đoạn (đa số là bài sơ khai về loài).
  - Loại 1.873 bài theo luật sơ khai, 55 bài trùng tiêu đề ViWikiFC.
  - **Còn 158.295 bài đạt.**

**Khử trùng mức bài.**
- **Cách làm:** dùng `scripts/decontam.py`, cả hai chiều. Kiểm trên từng đoạn và từng câu của **cả bài**. Một câu chạm thì bỏ cả bài.
- **Tham chiếu** (số văn bản sau khi bỏ trùng):

  | Tham chiếu | Số văn bản |
  |---|---|
  | dev 1.3 | 6.370 |
  | test 1.3 | 7.481 |
  | proxy_v1 | 2.910 |
  | ViWikiFC, mọi split | 24.589 |
  | ViFactCheck, mọi split | 14.054 |
  | Belebele | 1.943 |
  | FLORES-101 | 1.532 |

  - probe_v1 và SIB-200 không còn văn bản riêng sau khi bỏ trùng: văn bản probe chép từ dev, SIB-200 là câu FLORES.
- **Đã xét** 12.000 bài theo thứ tự xáo seed 20261028. **Bỏ 2.880 bài:**
  - theo cách chạm: chứa trọn một văn bản tham chiếu 4–12 chữ 2.747; chung 13-gram 62; trùng câu ngắn 55; câu 4–12 chữ nằm trong tham chiếu 16;
  - theo tham chiếu: dev 1.452, Belebele 895, test 421, ViFactCheck 63, ViWikiFC 31, proxy 16, FLORES 2.
- **Còn 9.120 bài sạch.**
- **Khử trùng lần hai ở mức dòng:** sau khi dựng, từng dòng (state đã nối, nhận định) được kiểm lại, vì chỗ nối hai đoạn có thể tạo chuỗi chạm. Bỏ thêm 2 cửa sổ.

**Pool.**
- Mỗi bài cho tối đa 1 dòng. Val là 5% số bài, chia theo hash id; không bài nào vừa ở train vừa ở val, kiểm bằng assert.
- Độ dài state (token, dòng có dấu):

  | Tập | Dòng | Không dấu | Token state: min / trung vị / max |
  |---|---|---|---|
  | `kc_pool_ngan` | 2.500 | 526 | 80 / 169 / 399 |
  | `kc_pool_dai` | 1.600 | 311 | 500 / 802 / 1.290 |
  | `val_kc` | 300 | 65 | 82 / 280 / 1.252 |

  `val_kc` gồm 180 dòng ngắn và 120 dòng dài.
- 400 bài "cho" dành cho 2e.
- Mix mặc định là 1.800 dòng ngắn + 1.200 dòng dài, lấy từ đầu pool; pool đã xáo theo seed.

**Kiểu dựng và nhãn của mix mặc định (3.000 dòng).**
- **Theo kiểu dựng:**

  | Kiểu | Số dòng |
  |---|---|
  | S-nguyen | 651 |
  | S-luoc | 369 |
  | R-so | 508 |
  | R-ten | 331 |
  | N-cungbai | 947 |
  | N-lacde | 194 |

  N-lacde chiếm 17,0% số NEI.
- **Choice:**
  - Xác nhận 505, Bác bỏ 483, NEI 680 (NEI 40,8%).
  - Tỉ lệ rút là 29/29/42, cao hơn đích 30/30/40 một chút: bài không dựng được NEI cùng bài hoặc Bác bỏ thì bị bỏ, làm tỉ lệ thực lệch về Xác nhận.
- **Noul:** Có 515, Không 817 (Có 38,7%). Tỉ lệ rút: Xác nhận 37 / Bác bỏ 28 / NEI 35.
- **Không dấu:** 591 dòng (19,7%), dùng `common.strip_diacritics` (hàm của `build_bench_v1`). Chuỗi nhãn và tiền tố không dấu trùng chuỗi nodiac của dev (assert từng dòng).
- **Lược:** áp cùng xác suất 0,4 cho cả ba nhãn. Tỉ lệ lược thực tế 36,9% / 37,0% / 36,8%.

**Luật dựng tự chọn** (brief để worker tự chọn):
- Mọi nhận định dài 8–45 chữ và có ≥ 1 số hoặc tên riêng, để ba nhãn cùng hình dạng.
- **R-so:** đổi số cùng loại. Có 4 loại: năm (1000–2100), ngày (sau chữ "ngày"), tháng (sau chữ "tháng", 1–12), và số khác cùng mẫu chữ số.
- **R-ten:** tên riêng là chuỗi ≥ 2 âm tiết viết hoa, không đứng đầu câu, và đứng trước dấu câu, chữ hoa hoặc một từ chức năng. Luật này để không cắt nhầm tên dạng "Họ Cá trích".
- **N-cungbai:** câu cách cửa sổ văn bản ≥ 2 đoạn.
- **KC-dài:** độ dài đích lấy theo log-chuẩn quanh 750 token.

## 4. Thành phần C so với C′ so với train_2b_mac_dinh

| Phần | C | C′ | train_2b |
|---|---|---|---|
| EN | 15.556 | 15.556 | 15.556 |
| multi | 403 | 403 | 403 |
| **VI** | **15.075** | **12.637** | **15.637** |
| VI choice / noul / score | 12.024 / 3.004 / 47 | 9.807 / 2.822 / 8 | 11.475 / 4.154 / 8 |
| **Tổng** | **31.034** | **28.596** | **31.596** |

**Nguồn VI đổi số dòng.**
- Bỏ hẳn: health_fact 665, VNTC 1.000, boolq 265, exams/vi 179.
- Đổi số dòng:

  | Nguồn | C | C′ |
  |---|---|---|
  | vsmrc | 1.217 | 881 |
  | vitaminc | 663 | 708 |
  | nli_fever | 660 | 702 |

  vitaminc và nli_fever tăng vì gồm cả dòng 2d và 2e.
- Mất 1–16 dòng do (c): vianli, tourism, multidogo, e-CARE, boolq_passage, PhoLaya…
- Thêm KC: 1.800 dòng ngắn và 1.200 dòng dài.

Bảng đủ theo nguồn ở `manifest_2b.json` → `so_dong_theo_nguon`.

**Số lựa chọn và độ dài state so với dev** (token, trung vị / p90):

| Khối (train_2b, VI) | n | Số lựa chọn | State |
|---|---|---|---|
| KC-ngắn | 1.800 | 3: 985, 2: 815 | 175 / 332 |
| KC-dài | 1.200 | 3: 683, 2: 517 | 826 / 1.124 |
| 2d | 1.160 | 3: 1.132, 2: 28 | 54 / 141 |
| 2e | 250 | 3 | 92 / 208 |
| Đọc hiểu | 2.682 | 4: 2.128, 3: 306, 5: 170, 2: 78 | 243 / 544 |
| *dev ViWikiFC* | 285 | 3: 161, 2: 124 | 103 / 206 |
| *dev ViFactCheck* | 158 | 3: 80, 2: 78 | 735 / 1.272 |
| *dev Belebele* | 294 | 4 | 142 / 210 |

Ghi chú về token: trung vị và p90 ở đây tính cả dòng không dấu. Bản không dấu dài hơn khoảng 15–20% token, nên max KC-ngắn là 498 và KC-dài là 1.505, vượt dải 400 / 1.300 vốn áp trên bản có dấu. Tất cả vẫn dưới trần 2.048.

## 5. Chi tiết các cổng

**D1.**
- 947 lần lệch của C (593 "Văn bản không đủ thông tin" ở vitaminc và 354 "không đủ thông tin" ở quail) về 0.
- 82 chuỗi gần trùng còn lại đều có sẵn trong C:
  - "gợi ý địa điểm du lịch nên đến" ≈ "du lịch" (SIB), 993 lần;
  - các lựa chọn câu như "Họ đang du lịch ở Mỹ.";
  - các lựa chọn ngắn "Thời tiết." / "Đồ ăn." gần nhãn MASSIVE.
- Hai chuỗi SIB "khoa học" và "chính trị - xã hội" mất cùng VNTC.
- **Một lỗi phương pháp đã sửa:** lần chạy đầu gọi `check_label_strings.py` với `-I`, nên bỏ qua PYTHONHASHSEED. Thứ tự set đổi giữa hai lần chạy, làm cùng một chuỗi train ghép với chuỗi dev gần khác, sinh 2 "chuỗi mới" giả. Bản cuối chạy không `-I` và so theo chữ train + khung.

**D3.**
- Ngoài phạm vi D3, chỉ báo thêm: 198 dòng C′ ở 2d, 2f, 2h (văn bản có sẵn từ C, không phải Wikipedia) chứa trọn một văn bản tham chiếu 4–12 chữ:
  - theo tham chiếu: dev 113, Belebele 56, test 28 (theo cả hai chiều chứa; câu test không in);
  - những chuỗi xem được (dev, Belebele) là cụm thông dụng 4 âm tiết: "chương trình truyền hình", "điện thoại di động", "tiếng tây ban nha", "tháng 3 năm 2020", "thế vận hội mùa hè"…
- **Luật "câu 4–12 chữ nằm trọn" đếm theo âm tiết** nên rất chặt với tiếng Việt. Đây cũng là lý do chính 2.747/2.880 bài Wikipedia bị bỏ.

**D4.** Đặc trưng nặng nhất là các từ chủ đề rời rạc ("vào năm", "hoa kỳ", "3", "5", "nguyễn"…), không có mẹo hệ thống. Không phải sửa luật dựng.

**D5** (mọi dòng KC có dấu, cả choice và noul). Tỉ lệ có số hoặc ngày: Xác nhận 68%, Bác bỏ 74%, NEI 68%. Riêng choice: 68% / 76% / 70%. Có hai chỉ số về trích nguyên:
- **Nhận định nằm nguyên trong văn bản:** Xác nhận 90,5%, Bác bỏ 0%, NEI 0%. Kiểu lược "bỏ mệnh đề sau dấu phẩy cuối" vẫn để lại một đoạn liền của câu gốc, nên vẫn tính là nằm nguyên.
- **Câu nguyên, không lược, không đổi:** Xác nhận 63%, NEI 63%, Bác bỏ 0%.

## 6. Các cách hiểu mình đã tự chọn

Ba điểm 1–3 đã hỏi điều phối lúc khoảng 08:10. **Điều phối đã xác nhận cả ba mặc định** (brief, mục 7b), cùng các cách hiểu ở điểm 4.

1. **VSMRC:** bỏ cả hai loại lựa chọn tổng hợp, 138 dòng, thay vì chỉ 21 dòng trỏ chữ cái. Nếu chỉ bỏ 21 dòng thì VSMRC còn 998 dòng thay vì 881.
2. **2h sửa tại chỗ,** không thêm bản sao. 51 dòng `abstain_replace` mất lựa chọn gốc; lựa chọn gốc lưu ở trường `options_goc`.
3. **Ngoại lệ D2 cho 2h** gồm cả câu hỏi đọc hiểu gốc (sau 2f), không chỉ lựa chọn.
4. Các điểm khác:
   - health_fact và boolq chỉ bỏ dòng VI.
   - 2d giữ nguyên kind (choice / noul) của dòng gốc.
   - State KC nối các đoạn bằng dấu cách, không xuống dòng, vì 443/443 state ViFactCheck/ViWikiFC của dev không có "\n".
   - D4 chạy bằng sklearn của Python Windows, vì `.venv` không có sklearn và không cài thêm gì.

## 7. Giới hạn

1. **Trùng từ tách nhãn quá sạch (D6, chỉ báo).**
   - Ở KC, nhận định Xác nhận luôn là câu của văn bản (trùng từ lớn nhất 1,00), Bác bỏ chỉ đổi một chỗ (trung vị 0,94), còn NEI theo luật "< 0,5" (trung vị 0,23, 0% ≥ 0,9).
   - Ở kiểm chứng thật (proxy_v1), ba nhãn chồng lên nhau nhiều: 0,68 / 0,67 / 0,50.
   - Model có thể học "trùng nhiều → Xác nhận / Bác bỏ, trùng ít → NEI", một mẹo không đúng trên dev (NEI dev có trung vị 0,50).
   - Đây là hệ quả trực tiếp của luật dựng trong brief (S trích câu; N-cungbai trùng < 0,5). Probe D4 chỉ đọc nhận định nên không bắt được mẹo này.
2. **KC-dài là văn bản Wikipedia,** khác thể loại với ViFactCheck (văn bản báo). Chỉ khớp về độ dài.
3. **Nhận định đổi cơ học có thể sai nghĩa hoặc vụng.**
   - R-ten có thể đổi tên khác loại (ví dụ "vùng Long Khánh" → "vùng Ấn Độ").
   - Lược "bỏ mệnh đề sau dấu phẩy cuối" có thể để lại câu cụt.
   - R-so có thể đổi sang giá trị vô tình vẫn đúng. QACG ước khoảng 2%; cần mẫu đọc D8 để có tỉ lệ.
4. **2d là bản dịch** (vitaminc, nli_fever từ tiếng Anh). 125 dòng 2e dùng bằng chứng dịch của dòng 2d khác làm văn bản lạc đề.
5. **Khử trùng rất chặt** (luật 4–12 âm tiết), nên bỏ cả bài vì cụm thông dụng. Mất khoảng 24% số bài đã xét; không ảnh hưởng số lượng vì còn dư bài.
6. **6 dòng noul 2d** có nhãn "khong_xac_nhan" (Bác bỏ hoặc NEI, không phân biệt), đáp án "Không". Không có trong khối choice.
7. **quail:** 35 dòng không tách được câu hỏi vẫn giữ câu hỏi chung ở ô câu hỏi. Câu hỏi kết thúc bằng ":" (điền tiếp) được tính là câu hỏi.

## 8. File

`results/train_2b/` (sha ở `manifest_2b.json`):
- `C_prime.jsonl`: 28.596 dòng, sha `0838ff55…`.
- `kc_pool_ngan.jsonl` (`7de6ba01…`), `kc_pool_dai.jsonl` (`41b83f24…`), `val_kc.jsonl` (`ea7bf753…`).
- `train_2b_mac_dinh.jsonl`: 31.596 dòng, `a6acebf5…`. Đây là C′ + KC mix 1.800/1.200, xáo theo seed. Chỉ là bản dựng sẵn; mix cuối chốt ở bước phân tích trước train.
- `mau_doc.jsonl`: D8, `f80c0f51…`. Cột cần che: `nhan` và mọi cột `nhan_*` (kiểu dựng, câu gốc, chữ bị đổi, nguồn). `nhom` chỉ ghi KC / 2d / 2f / 2g / 2h.
- `manifest_2b.json`: sha đầu vào, LFS oid Wikipedia, seed, số dòng theo khối và nguồn, dòng bị bỏ kèm lý do, kết quả lọc bài.
- `manifest_tai_wiki.json`.
- `tam/`: log và số trung gian (`loc_bai.json`, `kc_log.json`, `c_prime_log.json`, `kiem_2b.json`, `d4_probe.json`, `d1_*.json`, `bai_sach.jsonl`, `bai_cho.jsonl`).

**Mỗi dòng KC mang các trường:** `label_source=program`, `text_source=wikipedia_vi_20231101`, `license=CC BY-SA 4.0`, `kieu`, `luoc`, `tieu_de`, `wiki_id`, `doan`, `cau_goc`, `seed`, `gold_label`, `flags` (`nodiac`).

**Dòng C′ đã đổi mang `flags`:** `2d`, `2e`, `2f`, `2g`, `aug=abstain_add`, `aug=abstain_replace`; kèm `question_goc`, `options_goc`, `aug_cho`, `van_ban_tu` khi có.

**Script** (`scripts/train_2b/`, thứ tự chạy):

| Bước | Script |
|---|---|
| Tải Wikipedia | `tai_wiki.py` |
| Lọc bài, khử trùng mức bài | `wiki_bai.py` |
| Dựng KC | `build_kc.py` |
| Dựng C′ | `build_c_prime.py` |
| Ráp train_2b, cổng D1–D3, D5–D8 | `kiem_2b.py` |
| Cổng D4 (Python Windows) | `d4_probe.py` |
| Hàm dùng chung | `kc_util.py` |

**Thời gian:** khoảng 08:05–08:50, toàn bộ trên CPU. Theo log và giờ ghi file:
- tải Wikipedia 41 giây; `wiki_bai` 307 giây;
- chuỗi lần cuối `build_kc` → `build_c_prime` → `kiem_2b` hết hơn 10 phút; `build_c_prime` khoảng 1 phút (08:41:06 → 08:42:06), `kiem_2b` khoảng 4,5 phút (→ 08:46:33), `build_kc` phần còn lại;
- `d4_probe` dưới 1 phút.

Không bước nào gần ngưỡng báo 2 giờ.

## 9. KC v2: vòng sửa sau D6 (brief mục 7c, 28/10)

**Phạm vi:**
- C′ giữ nguyên, không dựng lại.
- KC v1 giữ nguyên ở các file cũ; KC v2 ghi vào file mới có hậu tố `_v2`.
- Script: `scripts/train_2b/build_kc_v2.py`, `kiem_kc_v2.py`, `d4_probe.py --v2`. Luật mới thêm vào `kc_util.py` dưới dạng hàm mới: `luoc_v2`, `cau_dung_duoc`. Hàm của v1 không đổi.

**Phép đo trùng từ** ("bao phủ"): với mỗi câu của state, lấy tỉ lệ số từ của nhận định có trong câu đó; giữ câu cao nhất. Điều phối dùng đúng phép đo này trên ViWikiFC train (0,70 / 0,71 / 0,50).

**Thiết kế đã dựng:**
- **Năm kiểu mới:** N-bobang, N-them, N-ghep, S-ghep, R-ghep. Từ duy nhất được chèn là "và", ghi trong manifest.
- **Dạng ghép của N-cungbai và N-lacde:** khoảng 2/3 số dòng hai kiểu này dựng ở dạng ghép, để NEI có khoảng 35% dạng ghép, khớp Xác nhận và Bác bỏ.
- **Biến thể định dạng:** chọn ngẫu nhiên cho 25% số dòng, như nhau cho cả ba nhãn.
- **Bốn sửa sau đọc mù vòng 1:**
  - bỏ mệnh đề chỉ khi sau dấu phẩy là từ mở đầu mệnh đề phụ;
  - bỏ ngoặc chỉ với ngoặc cân, có dấu cách đứng trước, câu còn lại không kết thúc bằng hư từ;
  - lọc câu kiểu trích dẫn và câu có tỉ lệ âm tiết mang dấu < 0,3;
  - R-so / R-ten: giá trị thay không có trong câu gốc (trước khi lược) và ở câu liền trước hay liền sau; không lấy tên năm can chi; ưu tiên ứng viên cùng từ đứng trước.
- **Bài "cho" của 2e** (400 bài) không dùng lại làm state.

**Pool:**

| File | Dòng | sha |
|---|---|---|
| `kc_pool_ngan_v2.jsonl` | 2.500 | `36df51d2…` |
| `kc_pool_dai_v2.jsonl` | 1.600 | `bb88a4be…` |
| `val_kc_v2.jsonl` | 300 | `41da2a34…` |
| `train_2b_mac_dinh_v2.jsonl` | 31.596 | `dc7737a7…` |

`train_2b_mac_dinh_v2` = C′ + 1.800 dòng ngắn + 1.200 dòng dài.

- **Theo kiểu (pool train):**

  | Kiểu | Số dòng |
  |---|---|
  | S-nguyen | 562 |
  | S-luoc | 388 |
  | S-ghep | 517 |
  | R-so | 612 |
  | R-ten | 497 |
  | N-bobang | 454 |
  | N-them | 326 |
  | N-ghep | 225 |
  | N-cungbai | 216 |
  | N-lacde | 303 |

- **Cơ cấu NEI:** bỏ bằng 29,8%, thêm 21,4%, ghép 14,8%, cùng bài 14,2%, lạc đề 19,9%. Đích là 30 / 25 / 15 / 15 / 15. N-them hụt và N-lạc đề dư, vì N-them hay không tìm được cụm chèn hợp luật.
- **Theo nhãn:** choice gồm NEI 915, Xác nhận 741, Bác bỏ 610. Có 801 dòng không dấu.

**Cổng** (số của lần dựng chốt; ngưỡng D6′ là bản đã sửa trước khi dựng):

| Mã | Số | Ngưỡng | Kết quả |
|---|---|---|---|
| D3 | 0 chạm trên 4.400 dòng (pool + val). Lúc dựng bỏ 1 cửa sổ vì chạm | 0 | **Đạt** |
| D4 | accuracy 0,376, đa số 0,404 (−2,8 điểm). Dòng có dấu: 0,384 so với 0,407 | ≤ đa số + 10 | **Đạt** |
| D5 | trung vị token nhận định 46 / 46 / 47 (lệch 2,2%). Phủ định: Xác nhận 7,2%, Bác bỏ 9,2%, NEI 7,4% → lệch **2,02 điểm** | ≤ 20%; ≤ 2 điểm | **Không đạt** (vế phủ định, vượt 0,02 điểm) |
| D6′ | NEI: trung vị bao phủ **0,333** (p10 0,12, p90 0,90) | NEI trung vị ≥ 0,40 | **Không đạt** (vế trung vị NEI) |
| | NEI có bao phủ ≥ 0,6: 27,0% (proxy 34%) | ≥ 25% | đạt |
| | Bao phủ < 0,9: Xác nhận 35,8%, Bác bỏ 46,7% | ≥ 30% mỗi nhãn | đạt |
| | Xác nhận là đoạn liền trong state: 41,9% | ≤ 50% | đạt |
| D7 | Khối kiểm chứng VI choice (2d + 2e + KC v2 mix, 3.027 dòng): NEI 36,3%. Noul KC "Có" 40,0% | ≥ 30%; 37–40% | **Đạt** ("Có" sát mép trên) |
| D9 | Tỉ lệ ghép: Xác nhận 35,2%, Bác bỏ 36,9%, NEI 36,7% (lệch 1,6 điểm). Biến thể định dạng: 15,4% / 12,3% / 16,5% (lệch 4,2 điểm) | ≤ 5 điểm | **Đạt** |
| D10 | Độ phủ token, trung vị: KC 1,00 / 0,97 / 0,56; proxy 0,88 / 0,85 / 0,78; ViWikiFC train 0,79 / 0,80 / 0,67. Probe chỉ dùng 2 đặc trưng trùng từ: KC choice có dấu **0,701** (đa số 0,407, +29 điểm); proxy kiểm chứng 0,437 (đa số 0,367, +7 điểm) | chỉ báo | — |

**Ghi chú:**
1. **Trung vị chỉ báo của D6′:** KC 1,00 / 0,91 / 0,33; ViWikiFC train 0,70 / 0,71 / 0,50; proxy 0,68 / 0,67 / 0,50. Mức chênh Xác nhận − NEI của KC là 0,67.
2. **Trung vị bao phủ theo kiểu:**

   | Kiểu | Trung vị |
   |---|---|
   | N-them | 0,88 |
   | N-ghep | 0,50 |
   | N-bobang | 0,32 |
   | N-cungbai | 0,24 |
   | N-lacde | 0,13 |
   | S-nguyen | 1,00 |
   | S-luoc | 1,00 |
   | S-ghep | 0,64 |
   | R-so | 0,93 |
   | R-ten | 0,89 |

   - N-bobang, N-cungbai và N-lacde chiếm 64% số NEI, và cả ba đều có trung vị dưới 0,40. Vì vậy trung vị NEI dưới ngưỡng, kể cả nếu cơ cấu NEI đúng tỉ lệ đích.
   - N-bobang có bao phủ thấp: bỏ câu bằng chứng đi thì phần văn bản còn lại ít trùng với nó.
3. **Biến thể định dạng chỉ đạt 12–16%,** không phải 25%. Dòng được chọn làm biến thể mà nhận định không có mẫu ngày / số / thế kỷ thì không dựng được và bị bỏ, nên tỉ lệ thực thấp hơn.
4. **Mẹo trùng từ giảm nhưng chưa hết.** D10 cho thấy chỉ hai đặc trưng trùng từ vẫn đoán đúng 70% nhãn KC, so với 44% trên proxy.
5. **Mẫu đọc `mau_doc_v2.jsonl`:** 220 dòng, sha `d02e60c4…`. Gồm 30 dòng mỗi nhóm N-bobang, N-them, N-ghep, S-ghep, R-ghep, biến thể định dạng; cộng 20 dòng S-lược và 20 dòng R-tên dựng lại. Cột cần che: `nhan` và `nhan_*`.

Worker chỉ báo số và áp ngưỡng, không sửa thêm và không chọn giữa v1 và v2.

## 10. Lọc sau đọc mù vòng 2 (điều phối, 28/10)

Không dựng lại KC. Script: `scripts/train_2b/loc_kc_v2.py`.

- **Đầu vào:** `train_2b_mac_dinh_v2.jsonl` (`dc7737a7…`).
- **Đầu ra:** `train_2b_v2_loc.jsonl`, 31.596 dòng, sha `ecaf2c68c3824e1c57b719dc598483a7d26ba684fe0076f214280c71bc8393f5`.

**Số dòng bị bỏ trong mix** (tổng 29; cả 29 đều bù được bằng dòng pool chưa dùng, cùng kiểu, kind và dải, đã qua cùng bộ lọc, theo thứ tự pool):

| Luật | Trong mix | Trên toàn pool v2 |
|---|---|---|
| (a) N-bobang vẫn được một câu của state bao phủ ≥ 0,9 | 2 | 3 |
| (b) N-bobang có câu sau câu bị xóa mang từ chỉ ngược ("này", "đó", "ấy", "khác" trong 5 chữ đầu) hoặc từ nối mở rộng | 20 | 28 |
| (c) state có "[[" hoặc "]]" | 7 | 8 |

**Cổng chạy lại trên mix sau lọc** (dòng KC có dấu):

| Mã | Số | Ngưỡng | Kết quả |
|---|---|---|---|
| D5 | trung vị token 46 / 46 / 47 (lệch 2,2%). Phủ định: Xác nhận 7,1%, Bác bỏ 10,1%, NEI 7,2% → lệch **3,01 điểm** | ≤ 2 điểm | **Không đạt** |
| D6′ | trung vị NEI **0,333** | ≥ 0,40 | **Không đạt** |
| | NEI có bao phủ ≥ 0,6: 27,4% | ≥ 25% | đạt |
| | bao phủ < 0,9: Xác nhận 34,5%, Bác bỏ 45,8% | ≥ 30% | đạt |
| | Xác nhận là đoạn liền: 42,7% | ≤ 50% | đạt |
| D7 | NEI 36,3%; noul "Có" 40,0% | ≥ 30%; 37–40% | **Đạt** |
| D9 | ghép 34,6 / 36,4 / 36,5% (lệch 1,9 điểm); biến thể 15,7 / 12,4 / 17,5% (lệch **5,08 điểm**) | ≤ 5 điểm | **Không đạt** (vế biến thể) |

**Ghi chú về cơ sở tính:** D5 và D9 ở mục 9 tính trên toàn pool v2 (4.100 dòng). Ở đây tính trên mix 3.000 dòng, đúng yêu cầu.
- Trên mix **trước** khi lọc, số đã như vậy: phủ định lệch 3,03 điểm, biến thể lệch 5,08 điểm.
- Bước lọc không làm đổi các số này. Chênh với mục 9 là do lấy mẫu mix từ pool.

Chỉ báo số, không chỉnh để ép qua cổng.

## 11. train_2b_v3: tái lấy mẫu NEI trong pool v2 (PHAN_TICH_TRUOC_TRAIN_2b mục 9.7, 28/10)

Không dựng lại KC, không chạy model. Script: `scripts/train_2b/build_v3.py`, `d4_probe.py --v3`.

- **Đầu vào:** `train_2b_v2_loc.jsonl`.
- **Đầu ra:** `train_2b_v3.jsonl`, 31.596 dòng, sha `ea604a0138c20f42b4beae67cdeb3fe02908c0fa96cd63b104c469cc54a11635`; manifest `manifest_v3.json`.

**Cách làm:**
1. C′ và các dòng KC Xác nhận / Bác bỏ giữ nguyên.
2. NEI rút lại trong từng ô (dải ngắn/dài, kind), số dòng mỗi ô giữ như cũ, lấy theo thứ tự xáo sẵn của pool.
   - Cơ cấu NEI đạt: bỏ bằng 32,1%, thêm 28,2%, ghép 19,8%, cùng bài 10,0%, lạc đề 10,0% (đích 31 / 29 / 20 / 10 / 10).
   - Mọi dòng rút mới và dòng dùng để hoán đổi đều qua bộ lọc (a), (b), (c). Trên toàn pool, bộ lọc loại 3 / 28 / 8 dòng.
3. Cân phủ định và biến thể định dạng bằng hoán đổi trong cùng ô (dải, kind, nhãn, kiểu, có/không dấu): 6 hoán đổi cho phủ định, 15 cho biến thể.
4. So với `train_2b_v2_loc`, có **173/3.000 dòng KC** khác, trong đó 152 dòng NEI. Mô phỏng của Fable dự kiến 182 dòng.

**Cổng, chạy một lần trên mix** (dòng KC có dấu):

| Mã | Số | Kết quả |
|---|---|---|
| D3 | 0 chạm trên 3.000 dòng KC | **Đạt** |
| **D4** (bắt buộc) | accuracy 0,407, đa số 0,413 (−0,6 điểm). Dòng có dấu: 0,386 (−2,7 điểm). Đặc trưng nặng nhất của NEI vẫn là "vào năm" (từ N-them) | **Đạt** |
| D5 | phủ định 7,1 / 8,3 / 8,2% → lệch **1,23 điểm**; trung vị token 46 / 46 / 46 | **Đạt** |
| D6′ | NEI: trung vị bao phủ **0,414**; dòng ≥ 0,6 chiếm 35,1%. Bao phủ < 0,9: Xác nhận 34,5%, Bác bỏ 46,0%. Xác nhận là đoạn liền 42,7%. Trung vị chỉ báo: 1,00 / 0,91 / 0,41 | **Đạt** |
| D7 | NEI 36,3% trong khối kiểm chứng VI choice (3.027 dòng); noul "Có" 40,0% | **Đạt** ("Có" sát mép trên) |
| D9 | ghép 34,6 / 36,3 / 32,4% (lệch 3,84 điểm); biến thể 15,7 / 14,3 / 15,9% (lệch **1,6 điểm**) | **Đạt** |

**D10 (chỉ báo):**
- Độ phủ token, trung vị: KC 1,00 / 0,97 / 0,67; proxy 0,88 / 0,85 / 0,78.
- Probe chỉ dùng 2 đặc trưng trùng từ: KC choice có dấu **0,674** (đa số 0,413); proxy 0,437 (đa số 0,367). Ở v2 trước tái lấy mẫu là 0,701.
- Mức khối kiểm chứng choice (KC có dấu + 2d + 2e): tỉ lệ NEI là 0,11 khi độ phủ ≥ 0,9, và 0,51 khi độ phủ < 0,9; tỉ số 4,6. Trước đây tỉ số là 5,8; dev là 1,6.

**Đo VRAM** (`--vram_probe 1664`, dùng GPU): **chưa chạy**, chờ điều phối báo GPU đã sẵn.
