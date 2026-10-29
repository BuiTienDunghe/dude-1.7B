# Fable đọc kết quả train thử bản C (C-s1 so với M1) trên dev 1.3

Ngày 26/10/2026. Người đọc: Fable 5.1, đánh giá độc lập theo yêu cầu của điều phối. **Chỉ đọc**: không chạy model, không train, không đọc test 1.3, không sửa file nào trong repo ngoài file này. Số lấy từ `results/train_v1/phan_tich_C/C-s1_vs_M1.md`, `results/train_v1/trend_C.json`, `results/probe_v1/REPORT_moc.md`; đếm thêm trên `results/train_v1/components/C.jsonl`, `data/bench_v1_3/dev_v1_3.jsonl` và preds từng câu trong `results/probe_v1/eval/` bằng bốn script nhỏ đặt ngoài repo (mục 6). McNemar tính lại theo nhị thức hai phía trên các lát chưa có p trong báo cáo.

Mã model: M1 = `Qwen3-1.7B-fullvocab-s1 (dự án)` (train_v0); M2 = `C-s1` (bản C, 31.034 dòng, 1 seed); M0 = Qwen3-1.7B gốc.

## 1. Kết luận

1. **Bản C không qua cổng đã chốt ở `BRIEF_train_chinh_thuc.md` mục 2, độc lập với việc xếp kịch bản.** VI có dấu 81,65 so với yêu cầu ≥ 84,0; chênh so với M1 +1,77 với CI [−0,24; +3,65] chứa 0. ViFactCheck có dấu 72,8 → 72,2 (p = 1) và MASSIVE có dấu 86,7 → 88,1 (p = 0,58) đều không "tăng có ý nghĩa" như cổng đòi. Độ trễ đo được 42,1 ms > 40 ms; nhiều khả năng do GPU dùng chung (cùng kiến trúc adapter, mốc 25% đo 34,4 ms, M1 33,5 ms) nhưng **vẫn là điểm chặn cho tới khi đo lại trên GPU rỗi**. Ba chỉ tiêu "giữ" đạt: không dấu 64,0 ≥ 61; EN 88,2 ≥ 87; Belebele +1,4, ViWikiFC +4,2, XCOPA +7,0 không tụt. Theo chữ brief: không lên Giai đoạn 2, không tự đổi chỉ tiêu.

2. **Xếp kịch bản (định nghĩa K1/K2 theo "Cập nhật 26/10"):**
   - **Không K1** (theo định nghĩa đã thay ở "Cập nhật 26/10"). Hỏng ở hai điều kiện: P5 kiểm chứng choice có dấu −2,5 (vượt ngưỡng 2; M1 là +2,9) và ViFactCheck Có/Không có dấu −6,4 [−11,9; −1,3] (M1 +3,9); (a) tụt mạnh hơn M1 ở ViWikiFC. (Điều kiện "P4 đúng nhiều hơn M1" của định nghĩa gốc không còn thuộc K1; nó được đọc ở K4 dưới đây và cũng không đạt.)
   - **K4 "dùng mẹo, không đọc văn bản" là tín hiệu mạnh nhất (mức chắc chắn: cao).** Khi ghép nhận định với văn bản không liên quan (P4, bản có dấu), tỉ lệ chọn "không đủ thông tin" rơi từ 91,2 → 62,5 (ViFactCheck, n = 80) và 88,2 → 50,3 (ViWikiFC, n = 161); xác suất trung bình dồn vào "Bị văn bản bác bỏ" tăng 0,16 → 0,39 (ViFactCheck) và 0,17 → 0,50 (ViWikiFC). Mốc thiên vị của M2 chỉ 2,2–2,5, nên không phải M2 "ghét" nhãn này ở dev; là M2 phán nhận định mà không cần văn bản.
   - **K2 "học thuộc chữ nhãn" kích hoạt qua điều kiện (a), không qua P1′ (mức chắc chắn: trung bình).** Pa ViWikiFC choice −16,8 [−24,5; −9,9] so với M1 −4,3 [−10,5; +1,2], hai CI không chạm. Phần tụt nằm **gọn ở nhãn "Bị văn bản bác bỏ"**: đúng 88,0% ở dev, còn 42,7% khi chữ lựa chọn bỏ dấu; hai nhãn còn lại giữ hoặc tăng. P1′ − P5 dương ở mọi model (M0 +13,7; M1 +2,9/+7,0; M2 +4,6/+5,4) nên phép P1′ hoá ra **không phân biệt được** K2; Pa là phép đã làm được việc đó.
   - **K9 "không đổi" ở mức tổng, kèm thay đổi thật ở từng lát.** VI có dấu p = 0,17; EN p = 0,81; không dấu p = 0,20. Tăng thật: MASSIVE không dấu +7,8 (p = 0,003), ViWikiFC có dấu +4,2 (p = 0,05), XCOPA có dấu +7,0 (p = 0,12, n = 100). Tụt thật: Có/Không không dấu 80,7 → 72,8 (p = 0,009); P4 như trên. Theo dõi: Belebele EN 86,7 → 83,3 (p = 0,08); SIB có dấu −2,7 (p = 0,20).
   - **Không K3** (nguồn gốc +2,5 > nguồn dịch +0,7). **Không K5** theo định nghĩa (EN −0,3, không dấu +1,6) nhưng có tụt cục bộ Có/Không không dấu ở trên. **Không K6** (Brier 0,350 → 0,344; ECE 0,087 → 0,088). **Không K7** (P2 cùng lựa chọn 88,7 → 89,6; lệch vị trí đáp án trong dữ liệu bị vòng train xáo lại, xem 3.6). **K8 chưa đóng**: 25/50/100% = 81,07/80,17/81,65 là nhiễu chứ không phải đỉnh; mốc 75% chưa chấm.

3. **Nguyên nhân chính kiểm được bằng số nằm ở khối kiểm chứng của phần B** (mục 3): 586 dòng health_fact chỉ có nhận định, không có văn bản; nhãn loại "không đủ thông tin" chỉ 12,8% trong khối kiểm chứng dịch so với 31,1% ở dev; văn bản vitaminc luôn cùng chủ đề với nhận định (0/312 dòng đo được là gần như không trùng từ) nên model chưa từng gặp "văn bản lạc đề → không đủ thông tin"; ba chữ nhãn của dev nằm 593 lần trong vitaminc. Nguyên nhân tụt Có/Không không dấu **chưa kiểm được**.

4. **Khuyến nghị: (b) sửa dữ liệu phần kiểm chứng (và VNTC), train thử lại một bản C′, chấm bằng đúng bộ probe; không lên train chính thức C nguyên trạng; không cần chạy A và B trước.** Chi tiết và rủi ro ở mục 4–5.

## 2. Căn cứ theo cổng và theo kịch bản

### 2.1 Cổng brief mục 2 (dev 1.3, batch 1)

| Chỉ tiêu | Yêu cầu | M1 | C@100% | Kết quả |
|---|---|---|---|---|
| VI có dấu, trung bình đều 6 nguồn | ≥ 84,0 và CI chênh không chứa 0 | 79,88 | 81,65; +1,77 [−0,24; +3,65] | **Không đạt** |
| VI không dấu | ≥ 61,0 | 62,37 | 63,97 | Đạt |
| EN | ≥ 87,0 | 87,95 | 88,20 | Đạt |
| Độ trễ trung vị batch 1 | ≤ 40 ms | 33,5 | 42,1 (GPU đang bị game dùng chung; C@25% 34,4) | **Chưa đạt, phải đo lại** |
| ViFactCheck có dấu | tăng, McNemar p < 0,05 | 72,8 | 72,2; đúng→sai 8, sai→đúng 7, p = 1 | **Không đạt** |
| MASSIVE có dấu | tăng, p < 0,05 | 86,7 | 88,1; 13/17, p = 0,58 | **Không đạt** |
| Belebele, ViWikiFC có dấu | không tụt | 82,7 / 76,5 | 84,0 / 80,7 (p = 0,05) | Đạt |
| XCOPA có dấu | không tụt | 78,0 | 85,0 | Đạt |

Ghi chú: MASSIVE gộp ba bản tăng +3,9 (p = 0,001) nhưng phần lớn là bản không dấu (+7,8, p = 0,003), không phải bản có dấu mà cổng đo.

### 2.2 Bảng kịch bản

| Kịch bản | Dấu hiệu đã chốt | Số của M2 (M1 trong ngoặc) | Kết luận, mức chắc chắn |
|---|---|---|---|
| K1 | P1′−P5 tụt ≤ 2; P5 cả 4 khung tụt ≤ 2; (a),(b) không tụt hơn M1; (c) không dồn nhãn (định nghĩa 26/10; P3/P4 của định nghĩa gốc chuyển sang K4) | P1′−P5 +4,6 ✓; P5: kiểm chứng choice −2,5 ✗, Có/Không −2,0, ViFactCheck Có/Không −6,4 [−11,9; −1,3] ✗, MASSIVE −0,7 ✓, SIB +1,0 ✓; (a) ViWikiFC −16,8 (−4,3) ✗; (b) +3,0 ✓; (c) TVD 0,064/0,078 ✓ | **Không K1** (cao) |
| K2 | P1′−P5 tụt > 2 và hơn M1; hoặc (a) tụt mạnh hơn M1; hoặc (c) dồn nhãn | P1′−P5 không tụt; **(a) ViWikiFC −16,8 [−24,5; −9,9] so với −4,3 [−10,5; +1,2]**, toàn bộ −9,0 (−6,4), MASSIVE −12,6 (−8,2, CI chạm), SIB −5,7 (−11,0); (c) sạch | **K2 kích hoạt qua (a), cục bộ ở nhãn kiểm chứng** (trung bình) |
| K3 | tăng ở nguồn dịch, nguồn gốc đứng yên | nguồn gốc +2,5, nguồn dịch +0,7 | Không K3 (cao) |
| K4 | P3 vẫn cao hoặc tăng; P4 vẫn chọn xác nhận/bác bỏ | P3 kiểm chứng ngang M1 (31–47 so với 31–46), Belebele VI 42,5 (38,8) trong nhiễu; **P4 NEI 62,5 (91,2) và 50,3 (88,2)** | **K4 qua P4** (cao về hiện tượng) |
| K5 | EN tụt > 1 hoặc không dấu tụt > 1,4 | EN −0,3 macro +0,25; không dấu +1,6 | Không K5; nhưng Có/Không không dấu −7,9 (p = 0,009) là tụt thật cục bộ |
| K6 | Brier/ECE xấu, tự tin khi sai tăng | Brier 0,350 → 0,344; ECE 0,087 → 0,088; tự tin khi sai 0,699 → 0,714 | Không K6 |
| K7 | P2 nhất quán giảm | 88,7 → 89,6 | Không K7 |
| K8 | đỉnh ở 50–75% rồi xuống | 81,07 → 80,17 → (75% chưa) → 81,65; SIB 84,6 → 79,9 → 79,9 | Chưa đóng; không có đỉnh rõ |
| K9 | mọi lát trong nhiễu | tổng p = 0,17 / 0,81 / 0,20; nhưng MASSIVE không dấu, ViWikiFC có dấu, Có/Không không dấu, P4 đổi thật | K9 ở tổng, không ở từng lát |

### 2.3 Đọc kỹ các tín hiệu trái chiều

- **P4 (văn bản không liên quan).** M2 chuyển sang "Bị văn bản bác bỏ" (ViWikiFC 80/161, ViFactCheck 30/80; M1 19 và 7), gần như không bao giờ "Được văn bản xác nhận". Độ trùng từ giữa nhận định và đoạn văn lạc đề **không khác** giữa câu M2 chọn NEI và câu M2 chọn bác bỏ (ViWikiFC 0,138 so với 0,131; ViFactCheck 0,356 so với 0,331), độ dài nhận định cũng không khác; vậy không phải M2 đọc nhầm đoạn văn. M2 giữ **cùng phán quyết với chính nó ở câu dev gốc** 46/161 (M1 27/161): nhận định "bị bác bỏ" ở dev thì sang P4 vẫn "bị bác bỏ" 37 lần. Đây là phán nhận định bằng kiến thức hoặc bằng cách viết, không bằng văn bản. Tỉ lệ NEI không phụ thuộc đoạn văn dài hay ngắn (0,45–0,56).
- **Pa (chỉ bỏ dấu lựa chọn) ở ViWikiFC.** Theo nhãn đúng: "Bị văn bản bác bỏ" 88,0 → 42,7; "Được văn bản xác nhận" 88,9 → 88,9; "không đủ thông tin" 43,9 → 61,0. Phân bố dự đoán của M2 đổi từ (bác bỏ 82, xác nhận 58, NEI 21) sang (39, 64, 58). Nghĩa là thiếu chuỗi có dấu "Bị văn bản bác bỏ" thì quyết định "bác bỏ" của M2 biến mất. M1 cũng tụt ở nhãn này (82,7 → 62,7) nhưng nhẹ hơn nhiều.
- **P1′ − P5 dương không phải bằng chứng chống K2.** Mọi model tăng khi đổi sang bộ nhãn P1′ (M0 +13,7, p = 0,0001), vì P1′ xoá lệch khung giữa câu hỏi và lựa chọn mà `REPORT_fable.md` đã cảnh báo. Phép này không đo được học thuộc chữ ở M2.
- **P5 (đổi câu hỏi).** M2 nhạy hơn M1 ở kiểm chứng: choice có dấu −2,5 (M1 +2,9), ViFactCheck Có/Không có dấu −6,4 (M1 +3,9): câu hỏi "Nội dung văn bản có cho thấy điều sau đây là đúng không?" kéo M2 sang "Có" (dự đoán Có 38 → 43, đáp án Có 28). MASSIVE và SIB không nhạy.
- **Có/Không −3,0 toàn bộ** thực ra là **không dấu −7,9 (p = 0,009)**, còn có dấu **+1,9**. Ở bản không dấu ViFactCheck M2 dự đoán "Có" 45/78 (đáp án 28), ở ViWikiFC mất độ nhạy với "Có" (66% so với 78,7%). Bỏ văn bản (P3) M2 trả "Có" 55% ở ViFactCheck không dấu (M1 17%). Ba mốc 25/50/100% cho 68,8 / 75,7 / 72,8, dao động lớn với n = 202.
- **ViWikiFC +4,2** đến từ nhãn xác nhận và bác bỏ (80,0 → 88,9; 82,7 → 88,0), còn NEI đứng yên (41,5 → 43,9). Chính lát này sập ở Pa và P4, nên phần tăng trong điểm chính có một phần mỏng.
- **SIB −2,7** rơi đúng ở "sức khỏe" (92,5 → 82,5) và "thể thao" (100 → 90), hai nhãn duy nhất trùng chữ với VNTC; không dồn nhãn (TVD 0,064). Chuyển đổi đúng→sai: thể thao → du lịch 3, sức khỏe → khoa học/công nghệ 2. 19 đúng→sai, 11 sai→đúng, p = 0,20.
- **MASSIVE không dấu +7,8** khớp với 191/1.000 câu ý định du lịch trong phần A viết không dấu; MASSIVE có dấu chỉ +1,4 và không có nguồn nào mang nhãn MASSIVE.
- **Belebele EN −3,4** (18 đúng→sai, 8 sai→đúng, p = 0,08): không lệch vị trí (phân bố vị trí dự đoán 81/79/67/67 so với đáp án 79/76/74/65). Chưa rõ nguyên nhân; để theo dõi.
- **Hiệu chỉnh và tham chiếu:** Brier tốt hơn một chút; tự tin khi sai nhích 0,70 → 0,71. Trùng lỗi với decider (Jaccard 0,382 → 0,365) và Quyet (0,350 → 0,361) gần như không đổi: M2 không "giống" model mạnh hơn.
- **Xu hướng theo mốc:** VI có dấu 81,07 / 80,17 / 81,65; không dấu 61,35 / 64,98 / 63,97; EN 88,30 / 88,20 / 88,20; val loss epoch 1 → 2: 0,7577 → 0,7566 (M1 là 0,7556 → 0,7762). Không có dấu hiệu quá khớp theo thời gian, cũng không có dấu hiệu "còn tăng nếu thêm dữ liệu".

## 3. Nguyên nhân trong dữ liệu train C.jsonl

Thành phần C: nền 19.953 (train_v0 bỏ 47), phần A 4.396 bản địa (VSMRC 1.217, EXAMS 179, ý định du lịch 1.000, ViANLI 1.000, VNTC 1.000), phần B 6.685 dịch (đọc hiểu 1.801, kiểm chứng 1.988, ý định 1.114, yes/no 793, nhân quả 989). `strip_p = 0,0` ở cả C và M1 (summary.json của M1 chưa có trường này): hai bản cùng không bỏ dấu lúc train.

| # | Giả thuyết | Số đếm | Trạng thái |
|---|---|---|---|
| 3.1 | **health_fact dạy "phán nhận định không cần văn bản"** | 586 dòng choice tiếng Việt có state chỉ là nhận định (trong 637 dòng health_fact choice của C, gồm 51 dòng EN của nền, chỉ 24 dòng có xuống dòng; trung vị state 24 token), nhãn "đúng/sai/đúng sai lẫn lộn/chưa được chứng minh"; thêm 70 noul + 9 score cùng kiểu | **Kiểm được**; cơ chế trực tiếp cho K4/P4 |
| 3.2 | **Nhãn "không đủ thông tin" quá hiếm** | Trong khối kiểm chứng dịch (1.777 dòng choice), nhãn loại "không đủ/trung lập/chưa chứng minh" = 228 dòng = 12,8% (vitaminc 90/593 = 15,2%; nli_fever "trung lập" 115/598; health_fact "chưa được chứng minh" 23/586). Dev: 31,1% (ViFactCheck 34/80, ViWikiFC 41/161) | **Kiểm được**; giải thích prior chống NEI (P(NEI) ở P4 0,82 → 0,49) |
| 3.3 | **Model chưa từng thấy "văn bản lạc đề → NEI"** | vitaminc: trong 312/593 dòng tách được hai văn bản (khung "Văn bản thứ nhất / thứ hai"), 0 dòng gần như không trùng từ; NEI vẫn trùng từ 0,47 (xác nhận 0,69, bác bỏ 0,55); 281 dòng còn lại dùng khung khác, chưa đo. nli_fever cũng là bằng chứng truy hồi cùng thực thể | **Kiểm được trên phần đo được**; P4 là ngoại suy ngoài phân bố train |
| 3.4 | **Học thuộc chuỗi có dấu "Bị văn bản bác bỏ"** | Ba chữ nhãn của dev (thiếu đuôi "để kết luận") nằm 593 lần làm lựa chọn, 188 lần làm đáp án "bác bỏ"; C có 0 chuỗi nhãn không dấu; không bỏ dấu lúc train | **Kiểm được về dữ liệu**, khớp hành vi Pa theo nhãn; chưa chứng minh nhân quả (cần C′) |
| 3.5 | **VNTC chia câu hỏi với SIB nhưng khác bộ nhãn** | 1.000 dòng cùng "Văn bản này thuộc chủ đề nào?" với 10 nhãn (chính trị - xã hội, đời sống, khoa học, kinh doanh, pháp luật, sức khỏe, thế giới, thể thao, văn hóa, vi tính); văn bản báo dài, chữ thường, không dấu câu | **Hợp lý, trùng đúng hai nhãn tụt**, nhưng p = 0,20, chưa đủ chắc |
| 3.6 | Lệch vị trí đáp án | VSMRC k=4: 43/31/14/12%; nli_fever "kéo theo" 55% ở vị trí 0; headline_cause 74% vị trí 0; multidogo 37% vị trí 3; boolq "Đúng" 65% | **Vô hại**: `train_lora.py` xáo lựa chọn mỗi lần nạp (`shuffled()`), P2 không xấu đi |
| 3.7 | MASSIVE không dấu tăng nhờ phần A | 191/1.000 câu ý định du lịch không dấu; phần B chỉ 3 dòng | **Kiểm được** (tăng thật nhưng miền hẹp) |
| 3.8 | **Tụt Có/Không không dấu do lệch nhãn Có/Không trong train** | VI noul trong C: nền 788 Không / 543 Có, ViANLI 676/324, phần B 331/342; tổng 59,7% Không. Dev noul 62,9% Không | **Bác**: tỉ lệ khớp dev, mà M2 lại lệch về "Có". Ứng viên còn lại: ViANLI 1.000 (A, câu hỏi "Câu nói có được suy ra từ đoạn văn không?") hoặc khung "Có phải X là nhãn đúng…" (B, 176 dòng kiểm chứng + 325 ý định). **Chưa kiểm được; cần A so với B hoặc C′** |
| 3.9 | Nhãn tiếng Anh chưa dịch trong phần B | 0 dòng; các dòng vitaminc/nli_fever mang nhãn tiếng Anh đều là dòng EN gốc của nền train_v0 | Loại bỏ giả thuyết |
| 3.10 | Không có đòn bẩy cho ViFactCheck và MASSIVE có dấu | Không nguồn tin tức kiểm chứng tiếng Việt (brief cấm vì rò rỉ); không nguồn mang 60 nhãn MASSIVE (banking77/clinc chỉ noul; multidogo 10 nhãn bảo hiểm) | Giải thích vì sao hai chỉ tiêu "phải tăng" đứng yên |

Ghi chú về định dạng: 593 dòng vitaminc dùng khung "Văn bản thứ nhất / Văn bản thứ hai" và câu hỏi chung của tasksource ("Hãy chọn nhãn mô tả văn bản đúng nhất"), khác khung dev (văn bản ở state, nhận định trong câu hỏi). Khối nli_fever dùng nhãn NLI "kéo theo / trung lập / mâu thuẫn", không dạy chữ nhãn dev; đây là điểm tốt cho đa dạng nhưng cũng nghĩa là chỉ 1/3 khối kiểm chứng khớp khung bench.

## 4. Khuyến nghị: (b) sửa dữ liệu rồi train thử lại một bản C′

### 4.1 Vì sao không (a) và không (c)

- **(a) lên train chính thức C nguyên trạng:** cổng không đạt ở chỉ tiêu chính và cả hai nguồn "phải tăng"; thêm K4 và K2 cục bộ. Hai seed × 3 giờ GPU cho một model đã biết không qua cổng.
- **(c) chạy A và B để tách nguyên nhân:** B một mình không tách được 3.1 (health_fact) khỏi 3.2–3.4 (vitaminc), vì cả bốn đều nằm trong B; sau A/B vẫn phải sửa dữ liệu rồi train lại. Các điều cần sửa ở 4.2 đã rõ từ số đếm và phần lớn không cần chữ tiếng Việt mới. Tuy vậy, A và B đã được người dùng duyệt ở quyết định (d) nên **nếu người dùng không muốn sửa dữ liệu khi chưa có bằng chứng nhân quả, chạy B nguyên trạng (khoảng 2,5 giờ GPU) là bước dự phòng hợp lệ**, đặc biệt để quy nguyên nhân của 3.8.

### 4.2 Sửa gì, ước số dòng

| Bước | Việc | Số dòng | Cần người dùng duyệt chữ? |
|---|---|---|---|
| 1 | **Bỏ health_fact** (chỉ có nhận định, không văn bản) | −665 (586 choice, 70 noul, 9 score) | Không |
| 2 | **Nâng tỉ lệ NEI trong kiểm chứng choice lên khoảng 30%.** Sau bước 1 còn 1.191 dòng, NEI 205 (17%). Cách A: thêm khoảng 220 dòng NEI từ kho vitaminc/nli_fever chưa dùng (cần một vòng dịch, kiểm, khử rò rỉ). Cách B: bớt khoảng 500 dòng xác nhận/bác bỏ (không cần dịch, C′ nhỏ hơn). Cách C: dựng cặp "văn bản lạc đề → NEI" bằng cách ghép nhận định với văn bản của dòng train khác (chỉ dùng dòng train, không chạm dev/probe), khoảng 200–300 dòng | +220 hoặc −500 hoặc +250 | Không (nhãn dùng chữ sẵn có) |
| 3 | **VNTC:** đổi câu hỏi cho khác SIB (ví dụ theo hướng "chuyên mục báo") hoặc bỏ hẳn | 1.000 dòng đổi chữ, hoặc −1.000 | **Có**, nếu đổi chữ (1 chuỗi) |
| 4 | (Tùy chọn, vòng sau) Đa dạng chữ nhãn kiểm chứng trong train: 1–2 cách diễn đạt khác cho mỗi nhãn, áp lên một phần dòng vitaminc | khoảng 300 dòng đổi chữ | **Có**; và **không được dùng** ba chữ của P1′ ("Nhận định đúng / Nhận định sai / Chưa đủ căn cứ để đánh giá") để P1′ còn giá trị |

Sau sửa, C′ khoảng 29.900–30.600 dòng, trần 5% mỗi nguồn vẫn đạt (nguồn lớn nhất VSMRC 1.217 ≈ 4%). Thời gian train thử khoảng 3 giờ GPU; chấm dev + probe khoảng 1,5 giờ.

### 4.3 Tiêu chí đọc kết quả C′ (chốt trước khi train)

- Cổng brief mục 2 giữ nguyên.
- Thêm tiêu chí hành vi, so với C: **P4 có dấu tỉ lệ NEI trở về ≥ 85%** ở cả hai nguồn (mức M1 là 88–91); **Pa ViWikiFC tụt không quá mức M1 (−4,3 [−10,5; +1,2])**; P5 kiểm chứng tụt ≤ 2 ở cả choice và Có/Không; Có/Không không dấu không kém M1 quá 2 điểm.
- Nếu C′ sửa được P4/Pa nhưng điểm chính vẫn dưới 84: báo người dùng rằng mục tiêu không đạt được bằng dữ liệu kiểu này (brief mục 7 đã dự liệu), không tự nới cổng.
- Lưu ý khi dựng cách C ở bước 2: P4 sẽ không còn là phép "chưa từng thấy" nữa; phải ghi rõ trong manifest và vẫn giữ P4 vì nó đo khái quát hoá sang văn bản dev.

## 5. Rủi ro của khuyến nghị

1. **Sửa xong K4/K2 chưa chắc kéo điểm chính lên.** Lợi ích ước: NEI của ViFactCheck choice hiện đúng 20,6% (34 câu); nếu lên 60% được thêm khoảng 13 câu, tức khoảng +8 trên ViFactCheck có dấu, +1,4 trên trung bình đều. SIB trở lại mức M1 cho +0,45. Cộng với +1,77 hiện có mới vào khoảng 83–84, sát ngưỡng, với nhiễu một seed ±1. Phần tăng ViWikiFC +4,2 có thể co lại khi bớt phụ thuộc chữ nhãn.
2. **Tụt Có/Không không dấu có thể do phần A (ViANLI) và sẽ còn sau khi sửa phần B**; khi đó cần A/B hoặc bỏ thử ViANLI.
3. **Độ trễ chưa đo lại**; nếu đo trên GPU rỗi vẫn trên 40 ms thì vấn đề nằm ngoài dữ liệu.
4. **K8 chưa đóng** cho tới khi chấm mốc 75%; nếu 75% cao hơn 100% rõ thì cân nhắc giảm epoch ở C′.
5. Thay đổi 3–4 chỗ cùng lúc làm C′ so với C không tách được từng nguyên nhân; chấp nhận vì mục tiêu là qua cổng, không phải quy nhân quả từng khoản.
6. Mọi chuỗi tiếng Việt mới (câu hỏi VNTC, chữ nhãn đa dạng) phải qua đề xuất và duyệt như quy trình hiện hành; không dùng Opus/Sonnet dịch thêm hàng loạt nếu chưa có quyết định mới của người dùng.

## 6. File đã dùng

- Đọc: `results/BRIEF_train_chinh_thuc.md`, `results/BRIEF_phan_tich_xu_huong.md`, `results/train_v1/phan_tich_C/C-s1_vs_M1.md`, `results/train_v1/trend_C.json`, `results/probe_v1/REPORT_moc.md`, `results/probe_v1/fable_soat/REPORT_fable.md`, `results/train_v1/REPORT_giai_doan_0.md`, `results/train_v1/train/C-Qwen3-1.7B-s1/summary.json`, `results/ft_loss/train/Qwen3-1.7B-s1/summary.json`, `scripts/train_lora.py` (chỉ grep), `config/train_format.json`, `config/prompt_frames.json`.
- Đếm: `results/train_v1/components/C.jsonl`; `data/bench_v1_3/dev_v1_3.jsonl`; `results/probe_v1/{p4_van_ban_khong_lien_quan,pa_lua_chon_khong_dau,p5_doi_cau_hoi}.jsonl`; preds `results/probe_v1/eval/{dev,p4,pa,p5}/{C-s1, C-s1-step484, C-s1-step969, Qwen3-1.7B-fullvocab-s1 (dự án), Qwen3-1.7B}/preds.jsonl`.
- Script đếm (ngoài repo): `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\s2_train_stats.py`, `s3_preds.py`, `s4_extra.py` (cùng thư mục).
- Không đọc test 1.3; không chạy model; không sửa file khác.
