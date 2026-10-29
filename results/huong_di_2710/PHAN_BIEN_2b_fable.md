# Phản biện phân tích trước train 2b — Fable 5.1, 28/10/2026

Chỉ đọc. Mọi số tự tính bằng script CPU trên file có sẵn, đặt ở `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2b\` (tên script ghi cạnh từng số, mục 7). Không chạy model, không đụng test, không sửa file dự án ngoài file này.

Đối tượng: `PHAN_TICH_TRUOC_TRAIN_2b.md` (bản 1), `BRIEF_2b_du_lieu.md` mục 7b–7c, `REPORT_2b_du_lieu.md` mục 1–10, `DIEU_PHOI_doc_mu_vong1.md`.

## Kết luận trước

**Tiến hành bậc 1, sau một lần tái lấy mẫu KC trong pool v2 (không dựng lại, không đổi luật, khoảng 30 phút CPU) và sau khi sửa luật đọc T3/T4 cùng phương án dự phòng VRAM.** Không cần vòng dữ liệu mới; không giảm tỉ lệ KC.

Lý do ngắn:
- Ba cổng trượt không phải lỗi dữ liệu: **D5** là nhiễu lấy mẫu (trên toàn pool lệch chỉ 0,87 điểm; 3,01 điểm trên mix = 1,4 SE), **D9** cũng do rút 1.800 + 1.200 dòng đầu pool, **D6′** do cơ cấu kiểu NEI. Mô phỏng một lần rút lại NEI (giữ nguyên 1.877 dòng Xác nhận/Bác bỏ) cộng khoảng 20 hoán đổi cân phủ định/biến thể: **182/3.000 dòng đổi, cả D5, D6′, D7, D9 cùng đạt** (`f_joint_redraw.py`).
- **Mẹo trùng từ (D10) là cấu trúc của cả khối kiểm chứng, không riêng KC.** Tỉ số P(NEI | bao phủ < 0,9) / P(NEI | bao phủ ≥ 0,9) ở khối train là 5,8; giảm KC xuống 2.000 hay 1.500 dòng chỉ còn 4,9 / 5,0; dev là 1,6. 2d không phải đối trọng mạnh (cũng theo chiều Xác nhận > Bác bỏ > NEI). Vì vậy không có cách sửa bằng dữ liệu hiện có; phải đo bằng hàng rào đúng, và **T3 gộp ≥ 0,76 không đủ**: ViWikiFC Xác nhận tụt 13 điểm mà ViFactCheck giữ thì T3 gộp vẫn 0,766. Bác bỏ (n = 102) không có hàng rào nào.
- **T1 chắc chắn đạt và không nói gì về "đọc văn bản"**: C đã đạt 87,5 / 89,4% NEI trên P4 **không dấu**; lỗi P4 của C chỉ nằm ở bản có dấu, cùng cơ chế với Pa (chuỗi "Bị văn bản bác bỏ"). 2b cân chuỗi nhãn (D1: 947 → 0) nên T1 và G3 sẽ cùng đạt. Đọc T1 là sàn.
- **Dự phòng VRAM sai chiều**: `--token_budget 16384` làm đỉnh micro-batch **tăng** (13.312 → 16.384 token). Dùng `--token_budget 9216` thì đỉnh đúng bằng C. Gradient tương đương theo code.
- **T4 với 1 seed mỗi bên trượt oan 19–25%** khi hiệu thật bằng 0; luật "T4 trượt → không chạy bậc 2" quá cứng. Ngưỡng sàng 1 seed nên là −2,0; −1,0 giữ cho bậc 2.

**Năm thay đổi đề nghị, theo thứ tự quan trọng:**
1. Sửa T3 thành hai vế: **precision NEI gộp ≥ 0,70** (C 0,862; probe mẹo cho 0,35–0,46) và **recall Xác nhận + Bác bỏ của ViWikiFC ≥ 0,78** (C 105/120 = 0,875). Giữ T3 gộp cũ làm chỉ báo.
2. Đổi dự phòng VRAM thành `--token_budget 9216` (hoặc chạy `--vram_probe 1664` trước). Không dùng 16384.
3. Tái lấy mẫu NEI trong pool theo cơ cấu them 0,29 / ghép 0,20 / bỏ bằng 0,31 / cùng bài 0,10 / lạc đề 0,10, cân phủ định và biến thể bằng hoán đổi cùng ô; chạy lại D3–D5, D6′, D7, D9, D10 **một lần**, rồi đi tiếp bất kể D10 (chỉ báo). Ghi là "tái lấy mẫu", không phải vòng sửa thứ hai.
4. T4: ngưỡng 1 seed −2,0; T2 thêm vùng xám 0,42–0,50 (người dùng quyết chạy seed 2 hay dừng). T1: sàn, báo thêm P4 theo nhãn gốc.
5. Giữ `--save_fracs 0.5` (rẻ), ghi G1 là hàng rào có nhiễu (591 dòng KC không dấu, C có 0), thêm pb / p1b chỉ báo vì Pa cũng đã vào phân bố train.

Mọi thay đổi ở 1, 3, 4 (ngưỡng, cơ cấu NEI) phải được viết vào brief hoặc bản phân tích **có ngày, trước khi train**, kèm lý do; nếu không thì không còn là đăng ký trước.

## 1. Số đã kiểm khớp với bản phân tích

| Mục trong phân tích | Số | Kết quả | Script |
|---|---|---|---|
| §4 recall C-s1 ViFactCheck NEI / Xác nhận / Bác bỏ | 0,206 / 0,833 / 0,857 (n 34 / 18 / 28) | khớp | `a_cs1_baselines.py` |
| §4 recall ViWikiFC | 0,463 / 0,870 / 0,878 (n 41 / 46 / 74) | khớp | — |
| §4 gộp | NEI 0,347 (75), Xác nhận 0,859 (64), Bác bỏ 0,873 (102) | khớp | — |
| §4 P4 có dấu | ViFactCheck 62,5 (n 80), ViWikiFC 50,3 (n 161) | khớp | — |
| §4 Pa ViWikiFC choice | −16,8 (77,0 → 60,2); riêng nhãn Bác bỏ 88 → 43 | khớp | — |
| §4 điểm chính | VI 81,65 / không dấu 63,97 / EN 88,20 | khớp | — |
| §2 NEI trong kiểm chứng VI choice | (679 + 170 + 250) / 3.027 = 36,3% | khớp | `b_kc_gates.py` |
| §2 dòng VI không có văn bản | 665 + 265 + 179 = 1.109 | khớp | REPORT mục 2 |
| §3 D5 mix choice có dấu | 7,08 / 10,09 / 7,17 → lệch 3,01 | khớp | `b_kc_gates.py` |
| §3 D9 biến thể | 15,69 / 12,38 / 17,45 → lệch 5,07 (phân tích ghi 5,08, làm tròn) | khớp | — |
| §3 D6′ | NEI trung vị 0,333; ≥ 0,6 27,4%; Xác nhận < 0,9 34,5%; Bác bỏ < 0,9 45,8%; đoạn liền 42,7% | khớp | — |
| §3 D10 độ phủ token theo nhãn KC | 1,00 / 0,97 / 0,58 (phân tích 0,56; khác do tập có dấu choice+noul) | khớp xấp xỉ | `c_shortcut.py` |
| §5 SE hiệu cặp recall NEI | 6,8 điểm với 28 câu đổi chiều, lệch ròng +11 | khớp (6–7) | tính tay |
| §6 thời gian | token sau pad +15,2% → 3,40 h từ 2,95 h (10.625 s trong `summary.json`) | khớp (+20%, 3,5 h) | `d_padded_tokens.py` |

Không có số sai trong bản phân tích. Các vấn đề dưới đây là về cách đọc và cách đặt luật.

## 2. Các vấn đề

### 2.1 T3 không bắt được mẹo trùng từ ở ViWikiFC; Bác bỏ không có hàng rào — **cao**

**Số (`c_shortcut.py`):**
- Độ phủ token của nhận định (tỉ lệ từ của nhận định có trong văn bản), **tỉ lệ dòng ≥ 0,9** theo nhãn Xác nhận / Bác bỏ / NEI:
  - KC mix (choice có dấu): **99 / 95 / 16%**;
  - 2d: 28 / 6 / 5%;
  - dev gộp: 34 / 23 / 16%; **dev ViWikiFC: 11 / 7 / 0%**; dev ViFactCheck: 94 / 64 / 35%.
- Ở dev ViWikiFC, **59% nhận định Xác nhận và 54% Bác bỏ có độ phủ < 0,7**; ở KC, 0% Xác nhận/Bác bỏ dưới 0,7. Ở ViFactCheck nhận định gần như trích nguyên (trung vị 1,00), nên mẹo không hại ở đó.
- Probe LR hai đặc trưng (trùng lớn nhất, độ phủ) học trên KC, áp lên dev choice: dự đoán NEI 182/241 câu; recall Xác nhận 0,22, Bác bỏ 0,13, NEI 0,84; **ViWikiFC Xác nhận 0,04, Bác bỏ 0,04**; precision NEI 63/182 = 0,35. Học trên cả khối KC + 2d + 2e: Xác nhận 0,75, Bác bỏ 0,00, NEI 0,47; precision NEI 35/76 = 0,46.
- C-s1 hiện precision NEI gộp 25/29 ≈ 0,86 (dự đoán NEI 29 lần, gold 75), tức C **thiếu** NEI; mẹo đẩy sang **thừa** NEI. Hai lỗi đối nhau, T2 chỉ đo một chiều.
- T3 gộp: nếu ViWikiFC Xác nhận tụt 0,870 → 0,739 (−13 điểm, 6 câu) mà ViFactCheck giữ 15/18 thì gộp = (15 + 34)/64 = **0,766, vẫn đạt**. Bác bỏ (n = 102; ViWikiFC 74) không có T nào.

**Sửa:** T3 gồm hai vế bắt buộc: (a) **precision NEI gộp ≥ 0,70** (dấu hiệu riêng của mẹo là dự đoán NEI tràn); (b) **recall Xác nhận + Bác bỏ của ViWikiFC ≥ 0,78** (C 105/120 = 0,875; −10 điểm = 12 câu, SE khoảng 3). T3 gộp cũ, recall Xác nhận ViFactCheck (n = 18, quá nhỏ) và proxy kiểm chứng (Xác nhận 110, Bác bỏ 96) ghi chỉ báo. Hai ngưỡng thay một, xác suất trượt oan tăng ít vì (a) rất xa mức C.

Ba ngưỡng T2, T3a, T3b nhất quán với nhau: ở đúng T2 = 0,50 (≈ 38 câu NEI đúng) và precision đúng 0,70 thì model dự đoán NEI ≈ 54 lần, sai ≈ 16; chia theo tỉ lệ câu (ViWikiFC 72%) là ≈ 11–12 câu Xác nhận/Bác bỏ ViWikiFC bị đổi thành NEI → 93–94/120 ≈ 0,78, vừa chạm T3b. Tức model đổi Xác nhận/Bác bỏ lấy NEI theo tỉ lệ 1:1 đứng ngay biên; đổi tệ hơn thì trượt. Đó là hành vi mong muốn của hàng rào.

### 2.2 Dự phòng VRAM `--token_budget 16384` làm đỉnh tăng, không giảm — **cao**

**Số (`d_padded_tokens.py`, `e_budget_peak.py`; độ dài prompt ước = `n_state_tokens` + 90):**
- C: dòng dài nhất ≈ 1.111 token, micro-batch 8 cố định → đỉnh **9.216 token sau pad**; `summary.json` ghi peak VRAM 10,25 GB (reserved 12,9 GB).
- 2b: 468 dòng > 1.024, dài nhất ≈ 1.650 → đỉnh **13.312 token (+44%)**. Batch đầu vượt mức C xuất hiện ở bước thứ 3 trong mô phỏng, nên nếu tràn thì tràn ngay, không mất giờ.
- `token_groups` chia theo ngân sách: với 16384 đỉnh là **16.384**; 12288 → 12.288; **9216 → 9.216 = đúng mức C**.
- Gradient: `(rl.sum() / len(rows)).backward()` = Σ loss / 32, bằng đường cũ `mean()/accum`. Tương đương, chỉ khác thứ tự cộng bf16.

**Sửa:** ghi `--token_budget 9216` làm phương án dự phòng (hay dùng ngay từ đầu). Mô phỏng (`d_padded_tokens.py`, cùng cách chia của `token_groups`): trung bình 2,15 nhóm mỗi bước thay cho 4, nhưng token sau pad **+12,9%** vì nhóm gộp nhiều câu ngắn rồi pad theo câu dài nhất; thời gian ước +0 đến +13% (tới khoảng +0,4 h), đổi lấy đỉnh VRAM bằng C. Hoặc chạy `--vram_probe 1664` vài phút trước khi train để biết micro-batch 8 có vừa 16 GB không. Bỏ con số 16384 khỏi kế hoạch.

### 2.3 Ba cổng trượt là nhiễu lấy mẫu và cơ cấu đích; một lần tái lấy mẫu sửa cả ba — **trung bình**

**Số (`b_kc_gates.py`, `f_joint_redraw.py`):**
- D5 phủ định trên **toàn pool có dấu**: 9,01 / 9,04 / 8,17 → lệch **0,87**. Trên mix 3,01. SE của hiệu hai tỉ lệ (n 337 và 424, p ≈ 0,10) là 2,2 điểm → 3,01 = **1,4 SE**; ngưỡng 2 điểm nằm dưới 1 SE của phép đo. Theo kiểu: S-lược 3,9%, S-nguyên 11,3%, S-ghép 12,3%, R-số 10,1%, R-tên 10,4%, N-thêm 11,1%, N-bỏ bằng 6,5%, N-lạc đề 5,8% — phủ định nằm trong câu gốc Wikipedia (Bác bỏ: 65 nhận định có phủ định, 77 câu gốc có), không do thao tác dựng.
- D9 biến thể lệch 5,07 = 3,1 SE, nhưng mix trước lọc đã 5,08 (REPORT mục 10): cũng do rút 1.800 + 1.200 dòng đầu pool.
- D6′: trung vị trùng lớn nhất theo kiểu NEI: **N-thêm 0,88**, N-ghép 0,50, N-bỏ bằng 0,32, N-cùng bài 0,24, N-lạc đề 0,13. Chỉ N-thêm có dòng ≥ 0,9 (41%). Cơ cấu mix hiện tại: bỏ bằng 29,5 / thêm 22,1 / ghép 15,1 / cùng bài 14,0 / lạc đề 19,3%.
- **Mô phỏng tái lấy mẫu** (giữ nguyên 1.877 dòng Xác nhận/Bác bỏ của mix, rút lại 1.123 dòng NEI theo cơ cấu them 0,29 / ghép 0,20 / bỏ bằng 0,31 / cùng bài 0,10 / lạc đề 0,10 trong cùng (dải, kind); rồi hoán đổi cùng ô (dải, kind, nhãn, kiểu, có/không dấu) để cân phủ định và biến thể): **182 dòng đổi**. Kết quả: D5 choice có dấu 7,08 / 8,31 / 8,30 (lệch 1,23); D9 biến thể 15,69 / 14,12 / 15,67 (1,57), ghép lệch 4,0; **D6′ NEI trung vị 0,414, ≥ 0,6 35,3%** (ViWikiFC train 37%, proxy 34%); Xác nhận < 0,9 34,5%, Bác bỏ 46,2%, đoạn liền 42,7%; D7 NEI 36,3%, noul "Có" 40,0%. Pool N-thêm đủ (326, dùng 317).
- Lọc của `loc_kc_v2.py` phải áp lại cho dòng mới (mô phỏng chỉ áp luật a và c; luật b cần danh sách từ nối của điều phối). D3 đã đạt trên toàn pool; D10 chạy lại vài phút. **D4 là cổng cần nhìn sau tái mẫu:** REPORT mục 5 ghi đặc trưng nặng nhất của probe chỉ đọc nhận định là các cụm "vào năm", "tại"…, đúng loại cụm mà N-thêm chèn vào, và tái mẫu nâng N-thêm từ 22 lên 29% NEI. Dư địa còn khoảng 13 điểm (0,376 so với 0,404 + 10) nên nhiều khả năng vẫn đạt, nhưng không phải thủ tục.

**Sửa:** làm đúng một lần, ghi trước cơ cấu và luật cân, chạy cổng một lần, đi tiếp dù D10 ra sao. Đây là lấy mẫu phân tầng trong pool đã đọc mù (N-thêm 34/34, N-ghép 28/31 trùng nhãn, 0 lỗi rõ), không phải vòng sửa luật; vẫn giữ lời "chỉ một vòng". **Nhưng cơ cấu kiểu NEI là số đã đăng ký trước** (brief 7c: 30/25/15/15/15 và "tỉ lệ kiểu NEI giữ nguyên"); đổi sang 31/29/20/10/10 phải ghi vào brief như một **sai lệch có ngày và lý do** (N-thêm là kiểu NEI duy nhất giống dev về độ phủ; 2e đã cung cấp 250 dòng lạc đề nên N-lạc đề bớt được), **trước** khi train. Các sửa T3/T4 ở 2.1 và 2.5 cũng vậy: viết và ghi ngày trước lúc bấm train thì mới còn là luật đọc chốt trước. Nếu điều phối muốn giữ nguyên mix, phương án chấp nhận được là ghi D5 là nhiễu (1,4 SE) và D6′ là hệ quả cơ cấu đích; nhưng khi đó mất lợi ích phụ là tăng NEI kiểu dev (N-thêm) từ 22 lên 29%.

Ghi chú cho lần sau: ngưỡng D5 "≤ 2 điểm" và D9 "≤ 5 điểm" phải đi kèm n; với n ≈ 340 mỗi nhãn, cổng D5 trượt ngẫu nhiên khoảng một phần ba số lần dù pool cân. Nên tính trên toàn pool hoặc cộng 1,5 SE.

### 2.4 Mẹo trùng từ là của cả khối; giảm KC không giúp; 2d yếu hơn phân tích nghĩ — **trung bình**

**Số (`b_kc_gates.py`, `c_shortcut.py`):**
- Khối kiểm chứng VI choice train (KC có dấu + 2d + 2e, n = 2.673): P(NEI | độ phủ ≥ 0,9) = **0,09**, P(NEI | độ phủ < 0,9) = **0,52**, tỉ số **5,8**. Dev: 0,21 / 0,34, tỉ số **1,6**.
- Giữ cơ cấu NEI mới và giảm KC: 3.000 dòng → tỉ số 4,5; 2.000 → 4,9; 1.500 → 5,0. Giảm KC chỉ làm mất NEI và mất dòng văn bản dài, không đổi tỉ số, vì 2d và 2e mang cùng chiều.
- 2d (1.132 choice): nhãn 643 / 319 / 170 (NEI 15%); độ phủ trung vị 0,78 / 0,60 / 0,54, cùng chiều Xác nhận > Bác bỏ > NEI. Probe học trên 2d riêng, áp lên dev: Xác nhận 0,91, Bác bỏ 0,06, NEI 0,00 — 2d dạy "diễn đạt lại → Xác nhận", không dạy Bác bỏ hay NEI ở độ phủ thấp. Dòng Bác bỏ độ phủ < 0,7 của cả khối ≈ 214, toàn bộ từ 2d, với nhãn nhiễu (đọc mù trùng 18/30).
- C-s1 trên dev NEI đúng 0,42 / 0,35 / 0,00 ở ba mức trùng từ thấp / vừa / cao: lỗi NEI của C không phụ thuộc trùng từ, nên tăng NEI recall bằng mẹo vẫn là tăng thật trên ViFactCheck, nhưng trả giá ở Xác nhận/Bác bỏ ViWikiFC.

**Đọc:** 2b không thể dạy "diễn đạt lại nhưng vẫn Xác nhận/Bác bỏ" bằng dữ liệu hiện có (không có chữ mới). Kỳ vọng thực tế của bậc 1: NEI recall tăng rõ (T2 đạt), Xác nhận/Bác bỏ ViWikiFC có thể tụt vài điểm. Bậc 1 đáng chạy chính vì đo được điều này, với hàng rào ở 2.1. Không đổi tỉ lệ KC/2d; không nhân đôi 2d (nhãn nhiễu).

### 2.5 T4 một seed trượt oan 19–25%; luật dừng theo T4 quá cứng — **trung bình**

**Số:** SD hiệu hai lần chạy 1 seed từ nhiễu seed là 0,79·√2 = 1,12; cộng nhiễu mẫu câu (SE cặp 1,0) là 1,5. P(hiệu < −1,0 | hiệu thật 0) = Φ(−1/1,12) … Φ(−1/1,5) = **0,19–0,25**. Với ngưỡng −2,0: 0,04–0,09. G2 (EN, −1,0, SE 1,06) tương tự khoảng 0,17.

**Sửa:** T4 sàng 1 seed ≥ C − 2,0; T4 trượt một mình → "chưa kết luận, cần bậc 2", không phải "không chạy bậc 2". Ngưỡng −1,0 với bootstrap cụm giữ cho bậc 2 (2 seed). T2: thêm vùng xám 0,42–0,50 (SE 6,8 + nhiễu seed) do người dùng quyết; < 0,42 mới là "không chuyển".

### 2.6 T1 là sàn, không phải bằng chứng; cơ chế lỗi P4 của C trùng Pa — **trung bình**

**Số (`a_cs1_baselines.py`):**
- P4 **không dấu** của C: NEI 87,5% (ViFactCheck) / 89,4% (ViWikiFC). P4 có dấu: 62,5 / 50,3, phần hụt dồn hết vào "Bị văn bản bác bỏ" (37,5 / 49,7%), "Được văn bản xác nhận" 0%. P4 noul: "Không" 100%.
- Pa ViWikiFC: nhãn Bác bỏ đúng 88 → 43 khi bỏ dấu lựa chọn. Hai phép cùng chỉ một thứ: chuỗi có dấu "Bị văn bản bác bỏ" hút xác suất.
- 2b: D1 đưa 947 lần lệch chuỗi về 0; 2e (250) + N-lạc đề (217, sau tái mẫu 112) dạy thẳng lạc đề → NEI; 20% KC không dấu có cả chuỗi nhãn không dấu.

**Đọc:** T1 ≥ 85% gần chắc đạt (SE ở p = 0,85: 4,0 / 2,8 điểm) và G3 cũng vậy, cùng vì cân chuỗi nhãn. Giữ T1 làm sàn (trượt nghĩa là lỗi khung/prompt); báo thêm P4 theo nhãn gốc (C: ViFactCheck chọn NEI 14/28 ở nhận định gốc Bác bỏ, 11/18 Xác nhận, 25/34 NEI; ViWikiFC 43/74, 21/46, 17/41): model đọc văn bản thì tỉ lệ này phải không phụ thuộc nhãn gốc. Thành công của 2b đo bằng T2 và T3 mới, không bằng T1/G3. Pa vào phân bố train nên thêm pb (nhãn tiếng Anh) hoặc p1b chỉ báo cho học thuộc chữ nhãn.

### 2.7 G1 không dấu có nhiễu — **thấp**

Mix có 591 dòng KC không dấu toàn bộ (595 sau tái mẫu), kể cả chuỗi nhãn; C và M1 có 0. G1 tăng không quy được cho "sửa kiểm chứng". Ghi G1 là sàn không tụt; nếu tăng thì ghi là do thêm dữ liệu không dấu. (Kiểm `b_kc_gates.py`.)

### 2.8 N-bỏ bằng còn "văn bản gãy" — **thấp**

Đọc mù vòng 2 gắn cờ `van_ban_gay` 6/34 = 18%; lọc theo danh sách từ nối bắt 20/331 = 6% trong mix. Khoảng 40 dòng N-bỏ bằng còn lại có thể mang dấu riêng của NEI (câu kế mở đầu bằng từ chỉ ngược). Tái lấy mẫu ở 2.3 nâng N-bỏ bằng 331 → 360; nếu muốn tránh thì dùng cơ cấu them 0,29 / ghép 0,20 / bỏ bằng 0,27 / cùng bài 0,12 / lạc đề 0,12 (chưa mô phỏng; chênh nhỏ). Chấp nhận và ghi là đủ.

### 2.9 Chi tiết cấu hình bậc 1 — **thấp**

- Số bước 2 × 987 = 1.974 (C 1.938); `gate_step_frac 0.129` → bước 255 (C 250), `monitor_frac 0.0645` → 127. Tương đương; ghi số tuyệt đối vào manifest.
- Giữ `--save_fracs 0.5` (hoặc 0.25,0.5,0.75 như C): chỉ tốn đĩa, cho phép đọc xu hướng nếu adapter cuối trượt.
- `val_v0` giữ như C là đúng để so val loss và cổng letter_mass; `val_kc_v2` (300 dòng: 104 / 88 / 108) chỉ chấm theo dõi, không chọn checkpoint — đúng như kế hoạch.
- Trần state 2.048: dòng dài nhất 1.560 token state (không dấu); không dòng nào bị cắt. Phân bố dài hơn C (p99 1.000 so với 822) là chủ ý (KC-dài).
- Ước thời gian 3,4–3,5 h hợp lý; attention siêu tuyến tính nên lấy 3,5 h.

## 3. Trả lời năm câu hỏi

1. **Tiến hành hay sửa thêm?** Tiến hành sau một lần tái lấy mẫu trong pool (2.3): không dựng lại, không luật mới, không đọc mù thêm; 182 dòng đổi; cả ba cổng đạt. Không giảm KC (2.4). Nếu từ chối tái mẫu thì vẫn tiến hành, ghi D5 là nhiễu 1,4 SE.
2. **D10 có hại dev không?** Có khả năng, **ở ViWikiFC Xác nhận/Bác bỏ** (59 / 54% nhận định độ phủ < 0,7), không ở ViFactCheck. T3 gộp 0,76 không đủ nhạy; thay bằng precision NEI ≥ 0,70 và recall Xác nhận + Bác bỏ ViWikiFC ≥ 0,78 (2.1). Không đổi tỉ lệ KC/2d: tỉ số mẹo ở mức khối không đổi khi giảm KC; 2d cùng chiều với KC và nhãn nhiễu.
3. **T1–T4, G1–G5:** mốc C-s1 khớp hết. Công thức độ nhạy §5 đúng (SE 6,8; P4 2,8–4,0). Sai ở cách đặt: T4 trượt oan 19–25% với 1 seed; T3 thiếu Bác bỏ và thiếu precision; T1 là sàn vì P4 đã vào phân bố và C đã đạt 87–89% trên bản không dấu. T1 đọc là "khung và chuỗi nhãn ổn", không phải "đọc văn bản".
4. **Lệch cấu hình:** chỉ một chỗ: dự phòng `--token_budget 16384` sai chiều; dùng 9216. Còn lại tương đương C.
5. **Khác:** không có gì khiến bậc 1 vô ích nếu sửa 2.1 và 2.2. Rủi ro còn lại đã biết: nhãn 2d nhiễu (không sửa được không có chữ mới), N-bỏ bằng gãy (thấp), G1 nhiễu (ghi).

## 4. Việc không cần làm

- Không dựng lại KC, không thêm kiểu dựng, không đọc mù vòng 3.
- Không giảm KC xuống 2.000 (2.4).
- Không nhân đôi 2d.
- Không đổi siêu tham số, không thêm 2c, không dùng teacher.

## 5. Script và file đã dùng

Thư mục: `C:\Users\dungbt06\AppData\Local\Temp\claude\C--Users-dungbt06-Fine-tune\37a42dde-25b8-49c2-a6df-b7c150c1a0b0\scratchpad\fable_2b\`

| Script | Việc | Đọc |
|---|---|---|
| `a_cs1_baselines.py` | recall theo nguồn/nhãn, macro, P4 (có dấu, không dấu, noul, theo nhãn gốc), Pa theo nhãn | `data/bench_v1_3/dev_v1_3.jsonl`, `results/probe_v1/{p4,pa}*.jsonl`, `results/probe_v1/eval/{dev,p4,pa}/C-s1/preds.jsonl` |
| `b_kc_gates.py` (+ `kcf.py`) | D5, D9, D6′ trên mix KC và pool; phân rã theo kiểu; tỉ số mẹo mức khối theo phương án mix | `results/train_2b/train_2b_v2_loc.jsonl`, `kc_pool_{ngan,dai}_v2.jsonl` |
| `f_joint_redraw.py` | một lần tái lấy mẫu NEI + hoán đổi cân; chạy lại D5, D6′, D7, D9, D10 | như trên |
| `c_shortcut.py` | độ phủ theo nhãn trên KC, 2d, 2e, dev, proxy, ViWikiFC train; probe LR 2 đặc trưng (sklearn 1.9, Python Windows) | thêm `data/proxy_v1/proxy_v1.jsonl`, `raw_refs/High-Will__ViWikiFC/data/train.csv` |
| `d_padded_tokens.py`, `e_budget_peak.py` | token sau pad, đỉnh micro-batch, ảnh hưởng `--token_budget` | `components/C.jsonl`, `train_2b_v2_loc.jsonl`, `scripts/train_lora.py` (đọc), `results/train_v1/train/C-Qwen3-1.7B-s1/summary.json` |

Phép đo trùng từ chép đúng từ `scripts/train_2b/kc_util.py` và `kiem_kc_v2.py` (normalize của `decontam.py`, tách câu, `max_ov`, `cover`). Một mô phỏng trung gian trong `b_kc_gates.py` ("Mô phỏng D6′… dồn N-them") sắp theo giá trị tăng dần rồi cắt, bị lệch thấp; số dùng trong bản này lấy từ `build()` và `f_joint_redraw.py`. Không đọc `test_v1.3.jsonl`. Không chạy model.
