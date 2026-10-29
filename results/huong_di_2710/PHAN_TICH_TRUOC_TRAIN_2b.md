# Phân tích trước train 2b (28/10/2026, bản 1, chờ Fable phản biện)

Theo quy tắc "phân tích trước train": dữ liệu, dư địa, độ nhạy, ngân sách, rủi ro và luật đọc kết quả phải chốt **trước** lần train đầu tiên.

**Nguồn:**
- `results/BRIEF_2b_du_lieu.md` (mục 7b, 7c);
- `results/train_2b/REPORT_2b_du_lieu.md` (mục 1–9);
- `results/train_2b/DIEU_PHOI_doc_mu_vong1.md` (đọc mù vòng 1 và 2);
- số của C lấy từ `results/probe_v1/eval/dev/C-s1/preds.jsonl` và `results/train_v1/phan_tich_C/FABLE_doc_ket_qua.md`.

## 1. Câu hỏi của lần train

**2b có sửa đúng lỗi kiểm chứng của C mà không làm hỏng phần khác không?** Lỗi kiểm chứng của C gồm:
- P4: gặp văn bản lạc đề thì chọn "bác bỏ";
- NEI ít được chọn: recall thấp;
- Pa: học thuộc chuỗi có dấu "Bị văn bản bác bỏ".

2b là **một gói** thay đổi dữ liệu, không tách được từng phần:
- bỏ nguồn không có văn bản và VNTC;
- đưa vitaminc/nli_fever về khung bench;
- ghép văn bản lạc đề → NEI;
- tăng cường "từ chối";
- sửa đọc hiểu;
- thêm khối KC từ Wikipedia.

Kết quả chỉ trả lời "gói có tác dụng không", không trả lời "phần nào có tác dụng".

## 2. Dữ liệu (C → train_2b_v2_loc)

| | C | 2b |
|---|---|---|
| Tổng dòng | 31.034 | 31.596 |
| VI | 15.075 | 15.637 |
| EN + đa ngữ | 15.959 | 15.959 (giữ nguyên) |
| Dòng VI không có văn bản (health_fact, boolq không đoạn, EXAMS) | 1.109 | 0 |
| VNTC | 1.000 | 0 |
| Kiểm chứng VI choice, khung bench | 0 | 3.050 (2d + 2e + KC) |
| Tỉ lệ NEI trong kiểm chứng VI choice | 12,8% | 36,3% (dev 31,1%) |
| Văn bản lạc đề → NEI | 0 | 250 (2e) + N-lạc đề trong KC |
| State ≥ 500 token (VI) | 846 | thêm khoảng 1.200 dòng KC-dài (trung vị 826 token) |

## 3. Cổng dữ liệu: đạt và không đạt

**File train:** `results/train_2b/train_2b_v2_loc.jsonl`, 31.596 dòng, sha `ecaf2c68…`. Bước lọc bỏ 29 dòng KC và bù lại từ pool:
- N-bỏ bằng còn trùng câu: 2;
- N-bỏ bằng có văn bản gãy: 20;
- state còn cú pháp wiki: 7.

**Tính trên mix 3.000 dòng KC sau lọc** (`REPORT_2b_du_lieu.md` mục 10). Các số này khác mục 9 vì mục 9 tính trên cả pool:
- D5 phủ định: Xác nhận 7,1 / Bác bỏ 10,1 / NEI 7,2% → lệch **3,01 điểm, không đạt**.
- D9 biến thể định dạng: 15,7 / 12,4 / 17,5% → lệch **5,08 điểm, không đạt sát ngưỡng**.
- D6′ trung vị NEI: 0,333, **không đạt**.
- D7 đạt.

Bảng dưới là số trên pool, giữ để đối chiếu.

| Cổng | Kết quả |
|---|---|
| D1 chữ nhãn, D2 không có chuỗi khung mới, D3 khử trùng (0 chạm), D4 probe chỉ đọc nhận định (0,376 < đa số 0,404), D7 tỉ lệ NEI, D9 cân hình dạng | **đạt** |
| D5: độ dài nhận định | đạt |
| D5: tỉ lệ phủ định (Xác nhận 7,2 / Bác bỏ 9,2 / NEI 7,4%) | **không đạt**: lệch 2,02 điểm, ngưỡng 2. Sát ngưỡng; D4 không dùng được tín hiệu này |
| D6′: trung vị bao phủ của NEI | **không đạt**: 0,333, ngưỡng 0,40. Nguyên nhân là **điều phối đặt cơ cấu đích mâu thuẫn với ngưỡng**: bỏ bằng + cùng bài + lạc đề chiếm 64% NEI và cả ba kiểu đều có bao phủ dưới 0,40. Các vế khác của D6′ đạt: NEI bao phủ ≥ 0,6 chiếm 27%; Xác nhận dưới 0,9 chiếm 35,8%; Bác bỏ dưới 0,9 chiếm 46,7%; Xác nhận là đoạn liền chiếm 41,9% |
| D10 (chỉ báo): probe chỉ dùng 2 đặc trưng trùng từ | **KC 0,701** (đa số 0,407), **proxy 0,437** (đa số 0,367). Bao phủ token theo nhãn: KC 1,00 / 0,97 / 0,56, proxy 0,88 / 0,85 / 0,78 |
| Đọc mù vòng 1 (310 dòng) và vòng 2 (220 dòng) | **mọi nhóm đạt**. Lỗi đã lọc trong `train_2b_v2_loc`: dòng trùng câu, văn bản gãy, cú pháp wiki |

**Giới hạn lớn nhất là D10.** Không có diễn đạt lại thì nhận định "Xác nhận" luôn được văn bản bao phủ gần đủ. Vì vậy trên KC, độ trùng từ vẫn tách nhãn tốt hơn hẳn dữ liệu thật: 0,70 so với 0,44. Đã sửa một phần (vòng v2), nhưng không thể sửa hết bằng chương trình.
- **Rủi ro:** model học "chữ của nhận định không có trong văn bản → NEI", nên chọn NEI cho câu "Xác nhận" có diễn đạt lại ở dev.
- **Đối trọng trong dữ liệu:**
  - 2d: 1.160 dòng vitaminc/nli_fever dịch, có diễn đạt lại thật, nhưng nhãn nhiễu (đọc mù trùng 18/30);
  - khối NLI tiếng Việt và tiếng Anh sẵn có trong C.
- **Đo được** bằng recall Xác nhận trên dev (mục 5).

## 4. Mốc của C-s1 (dev 1.3, VI có dấu, kiểm chứng choice)

| Nguồn | Nhãn | n | Recall C-s1 |
|---|---|---|---|
| ViFactCheck | NEI | 34 | 0,206 |
| ViFactCheck | Xác nhận | 18 | 0,833 |
| ViFactCheck | Bác bỏ | 28 | 0,857 |
| ViWikiFC | NEI | 41 | 0,463 |
| ViWikiFC | Xác nhận | 46 | 0,870 |
| ViWikiFC | Bác bỏ | 74 | 0,878 |
| **Gộp hai nguồn** | NEI | 75 | **0,347** |
| | Xác nhận | 64 | **0,859** |
| | Bác bỏ | 102 | 0,873 |

**P4** (có dấu, tỉ lệ chọn NEI khi văn bản lạc đề):
- C: ViFactCheck 62,5 (n = 80), ViWikiFC 50,3 (n = 161);
- M1: 91,2 / 88,2.

**Pa ViWikiFC choice:** C −16,8 [−24,5; −9,9]; M1 −4,3.

**Điểm chính C-s1:** VI 81,65; không dấu 63,97; EN 88,20.

## 5. Độ nhạy

Các thước đo chính đều so theo cặp trên cùng câu, giữa 2b-s1 và C-s1.

| Thước đo | n | SE hiệu cặp, ước | Mức thay đổi bắt chắc |
|---|---|---|---|
| Recall NEI gộp | 75 | khoảng 6–7 điểm (nếu khoảng 25–30 câu đổi chiều) | +15 điểm ≈ 2,2 SE |
| Recall Xác nhận gộp | 64 | khoảng 5 điểm | tụt 10 điểm ≈ 2 SE |
| P4 ViWikiFC | 161 | khoảng 4 điểm | +20 điểm, rất chắc |
| P4 ViFactCheck | 80 | khoảng 5–6 điểm | |
| VI macro | — | SE hiệu cặp khoảng 1,0; thêm nhiễu seed (SD trong một lần chạy 0,79) | 1 seed mỗi bên chỉ bắt được thay đổi ≥ 2–3 điểm |

**Hệ quả:**
- Một seed 2b là đủ để đọc các thước đo chính (P4, recall NEI, recall Xác nhận).
- Kết luận về VI macro cần 2 seed mỗi bên.
- Vì vậy chia train thành 2 bậc.

## 6. Kế hoạch train theo bậc

**Bậc 1** (khoảng 3,5 giờ train + 0,5 giờ chấm; cần người dùng duyệt):
- 2b-s1 trên `train_2b_v2_loc.jsonl`, siêu tham số **giống C**: Qwen3-1.7B, LoRA r16/α32, lr 2e-4, batch hiệu dụng 32, 2 epoch, full_vocab, seed 1, warm-up 3%, `strip_p` 0, val_v0, cổng letter_mass ở bước 250, adapter cuối. Cấu hình lấy theo `scripts/train_v1/run_train_C.sh`. Riêng trần state là 2.048. `val_kc_v2` chỉ chấm riêng để theo dõi.
  - Thời gian ước theo C (2,95 giờ) cộng khoảng 20% token do KC-dài.
  - Nếu VRAM vượt thì dùng `--token_budget 16384`. Gradient không đổi (Σ loss / 32); ghi vào báo cáo.
- Chấm, không đụng test:
  - dev 1.3 (VI có dấu, không dấu, EN);
  - probe P4, Pa, P5;
  - proxy_v1, phần kiểm chứng.

**Luật đọc bậc 1, chốt trước** (2b-s1 so với C-s1, VI có dấu):

| Mã | Điều kiện | Ngưỡng |
|---|---|---|
| T1 | P4 NEI | ≥ 85% ở **cả hai** nguồn (mức M1; tiêu chí cũ của Fable) |
| T2 | Recall NEI gộp | ≥ 0,50 (C 0,347; tức +15 điểm) |
| T3 | Recall Xác nhận gộp | ≥ 0,76 (C 0,859; tụt ≤ 10 điểm). Đây là hàng rào cho rủi ro D10 |
| T4 | VI macro | ≥ 80,65 (C-s1 − 1,0) |
| G1 | Không dấu | ≥ 62,47 (C − 1,5) |
| G2 | EN | ≥ 87,20 (C − 1,0) |
| G3 | Pa ViWikiFC choice | tụt ≤ 10 điểm (C −16,8; M1 −4,3) |
| G4 | P5 kiểm chứng (choice và Có/Không) | không kém C quá 3 điểm |
| G5 | SIB-200 và Belebele có dấu | không kém C quá 5 điểm mỗi nguồn (hàng rào cho việc bỏ VNTC và sửa đọc hiểu; n nhỏ, chỉ bắt mức tụt lớn) |

**Đọc kết quả:**
- **T1–T4 và G1–G5 đều đạt:** "2b sửa được lỗi kiểm chứng" → đề xuất bậc 2.
- **T1–T2 đạt, nhưng T3 hoặc T4 trượt:** "sửa P4/NEI nhưng đổi bằng mẹo trùng từ, hoặc hỏng phần khác".
  - Không chạy bậc 2.
  - Phân tích lỗi theo dòng, rồi đề xuất giảm tỉ lệ KC hoặc bỏ kiểu dựng gây lỗi. Lần train sau phải xin duyệt lại.
- **T1–T2 trượt:** "dữ liệu dựng không chuyển sang dev". Dừng 2b và báo người dùng.
- **G nào trượt:** báo rõ nguồn trượt. Người dùng quyết có sang bậc 2 hay không.

**Bậc 2** (chỉ khi bậc 1 đạt, khoảng 7,5 giờ GPU; **xin duyệt riêng**): 2b-s2 và C-s2. Kết luận VI macro trên trung bình 2 seed, bootstrap theo cụm, như cổng cũ.

**Không làm trong 2b:**
- đích mềm từ teacher;
- thêm tiếng Anh (2c);
- đổi siêu tham số;
- train lại 2a.

## 7. Ngân sách và điều kiện dừng

- **Bậc 1:** khoảng 4 giờ GPU. Báo bên Ún nhường GPU trước, trả lại khi xong.
- **Dừng khi:** nhiệt độ GPU > 85°C; mức dùng tuần > 90% (từ 88% không bắt đầu việc mới); Fable > 90%.
- **Mức dùng lúc viết:** tuần 51%, Fable 33%.

## 8. Rủi ro đã biết, ghi trước

1. **Mẹo trùng từ của KC** (D10). Đo bằng T3.
2. **Nhãn 2d nhiễu.** Nhiễu có sẵn từ FEVER/vitaminc, không do bước dựng. Có thể làm T2 và T3 kém đi.
3. **KC-dài là Wikipedia, khác thể loại văn bản báo của ViFactCheck.** Mức chuyển sang ViFactCheck có thể thấp hơn ViWikiFC.
4. **Khối mới dùng đúng khung bench**, nên điểm dev đo khả năng trong khung. Pa và P5 là hàng rào khái quát hoá (G3, G4).
5. **Tình trạng có sẵn từ C, không do 2b:** 198 dòng C′ (2d, 2f, 2h) chứa trọn một câu 4–12 âm tiết của văn bản tham chiếu, chủ yếu là cụm thông dụng; 28 dòng chạm test. Ghi để minh bạch.
6. **Ngưỡng D6′ và D10 dùng số tổng hợp của proxy_v1 và ViWikiFC train.** Đây là liên hệ nhẹ với tập chọn model. Không có checkpoint nào được chọn theo proxy trong bậc 1, vì chỉ dùng adapter cuối như C.

## 9. Bản 2: sửa sau phản biện của Fable (28/10, trước khi train)

**Nguồn:** `results/huong_di_2710/PHAN_BIEN_2b_fable.md`. Điều phối đã tự kiểm hai số then chốt:
- precision NEI gộp của C-s1 = 25/29 = 0,862;
- tỉ lệ nhận định có độ phủ < 0,7 ở dev ViWikiFC: Xác nhận 54%, Bác bỏ 49% (Fable đo 59 / 54%; khác nhau do cách chuẩn hoá). Ở KC tỉ lệ này là 0%.

Điều phối chấp nhận cả 5 sửa. **Các luật dưới đây thay luật tương ứng ở mục 6**, và được ghi trước lần train đầu tiên.

**1. Hàng rào cho mẹo trùng từ** (thay T3):
- **T3a:** precision NEI gộp hai nguồn ≥ 0,70. C 0,862; probe mẹo trùng từ cho 0,35–0,46.
- **T3b:** recall Xác nhận + Bác bỏ của ViWikiFC ≥ 0,78. C 105/120 = 0,875.
- T3 gộp cũ (recall Xác nhận ≥ 0,76), recall Xác nhận ViFactCheck (n = 18) và proxy kiểm chứng: chỉ báo.

**2. T4, một seed:** sàng ≥ C-s1 − 2,0 = **79,65**.
- T4 trượt một mình → "chưa kết luận, cần bậc 2"; người dùng quyết.
- Ngưỡng −1,0 với bootstrap theo cụm giữ cho bậc 2 (2 seed mỗi bên).

**3. T2:**
- ≥ 0,50 là đạt;
- 0,42–0,50 là vùng xám, người dùng quyết chạy seed 2 hay dừng;
- < 0,42 là "không chuyển sang dev".

**4. T1 là sàn, không phải bằng chứng "đọc văn bản".** C đã chọn NEI 87,5 / 89,4% trên P4 không dấu; lỗi P4 có dấu cùng cơ chế với Pa, mà 2b đã cân chuỗi nhãn.
- T1 trượt nghĩa là lỗi khung hoặc chuỗi nhãn.
- Báo thêm P4 theo nhãn gốc của nhận định. Nếu model đọc văn bản thì tỉ lệ chọn NEI không được phụ thuộc nhãn gốc.

**5. Hàng rào phụ:**
- G1 (không dấu) có nhiễu, vì 2b thêm 591 dòng KC không dấu còn C có 0. Đọc G1 là sàn "không tụt"; nếu tăng thì ghi do thêm dữ liệu không dấu.
- Thêm pb và p1b (chỉ báo), vì Pa đã vào phân bố train.

**6. VRAM:**
- Bỏ phương án `--token_budget 16384`, vì nó **tăng** đỉnh micro-batch: 2b có đỉnh 13.312 token, C 9.216.
- Trước khi train chạy `--vram_probe 1664`:
  - reserved ≤ 14 GB → dùng đường micro-batch 8 như C;
  - ngược lại → `--token_budget 9216`, tức đỉnh bằng C. Gradient tương đương (Σ loss / 32). Ghi vào manifest.
- Giữ `--save_fracs 0.25,0.5,0.75` như C, chỉ để đọc xu hướng; vẫn dùng adapter cuối.

**7. Tái lấy mẫu KC trong pool v2. Đây là sai lệch có ngày so với brief 7c.**
- **Đổi cơ cấu kiểu NEI** từ 30/25/15/15/15 sang **bỏ bằng 0,31 / thêm 0,29 / ghép 0,20 / cùng bài 0,10 / lạc đề 0,10**. Dòng Xác nhận và Bác bỏ giữ nguyên.
  - Lý do: N-thêm là kiểu NEI duy nhất giống dev về độ phủ. 2e đã có 250 dòng lạc đề nên N-lạc đề bớt được.
- **Cân phủ định và biến thể định dạng** bằng hoán đổi trong cùng ô (dải, kind, nhãn, kiểu, có/không dấu). Áp lại bộ lọc `loc_kc_v2` cho dòng mới.
- Mô phỏng của Fable: 182/3.000 dòng đổi; D5 lệch 1,23; D9 lệch 1,57; D6′ NEI trung vị 0,414 (≥ 0,6 chiếm 35,3%); D7 giữ.
- **Chạy cổng một lần:**
  - **D4 phải đạt.** N-thêm chèn các cụm "vào năm…", vốn là đặc trưng nặng nhất của probe chỉ đọc nhận định. Nếu D4 trượt thì dừng và báo người dùng, không train.
  - D10 chỉ báo. Các cổng khác báo số và đi tiếp.
- Không dựng lại KC, không đọc mù thêm. Lời "chỉ một vòng sửa luật" vẫn giữ: đây là lấy mẫu phân tầng trong pool đã đọc mù.

**Kỳ vọng thực tế của bậc 1** (theo Fable, mục 2.4): recall NEI tăng rõ; Xác nhận và Bác bỏ ở ViWikiFC có thể tụt vài điểm. Bậc 1 đáng chạy chính vì đo được sự đánh đổi này, với hàng rào T3a/T3b.

**Không đổi:** tỉ lệ KC so với 2d (giảm KC không làm giảm tỉ số mẹo ở mức khối, vì 2d và 2e cũng mang cùng chiều), siêu tham số, phạm vi 2b.

## 10. Sai lệch có ngày cho bậc 2 (28/10, sau khi đọc 2b-s1, trước khi train 2b-s2)

**Người dùng duyệt** (chat 28/10, khoảng 15:50): "được, duyệt 2b-s2, nhớ giữ nhiệt độ dưới ngưỡng".

**Thay đổi so với mục 6, bậc 2 (2 × 2):**
- **Chỉ chạy 2b-s2**, hoãn C-s2.
- **Lý do** (`results/huong_di_2710/FABLE_doc_ket_qua_2b_s1.md` mục 8):
  - Cổng T4 của bậc 2 (trung bình 2 seed mỗi bên ≥ C − 1,0) chỉ trượt nếu 2b-s2 − C-s2 < −5,2. Như vậy 2b-s2 phải lệch seed 1 tới 8,4 điểm, trong khi dải checkpoint của C chỉ 1,5. C-s2 vì thế không đổi được kết luận đi hay dừng.
  - 2b-s2 trả lời câu hỏi còn mở: nhiễu seed, và các chi phí của s1 có lặp lại không.
- **C-s2 chạy sau** nếu một trong hai điều kiện xảy ra:
  - 2b-s2 lệch 2b-s1 quá 2 điểm VI macro;
  - người dùng muốn giữ dạng 2 × 2 cho báo cáo cuối.

**Luật đọc 2b-s2:**
- **Giữ nguyên T1–T4, G1–G5** (mục 9), so 2b-s2 với C-s1. Báo thêm trung bình 2 seed của 2b so với C-s1.
- **Chỉ báo mới**, đặt sau khi đã thấy số của s1, nên **không phải cổng**:
  - (a) recall "Có" của câu Có/Không, theo từng nguồn, có dấu và không dấu;
  - (b) tỉ lệ đoán NEI trên ViWikiFC không dấu (s1: 91/161, gold 41);
  - (c) recall Bác bỏ ViWikiFC có dấu (s1: 0,770, C 0,878);
  - (d) |2b-s2 − 2b-s1| trên VI macro, không dấu, EN, và từng nguồn kiểm chứng.

**Câu hỏi seed 2 phải trả lời:**
- Bác bỏ ViWikiFC có tụt ≥ 8 điểm nữa không?
- Lỗi "NEI mặc định" khi không dấu có lặp lại không (> 50% câu ViWikiFC không dấu bị đoán NEI)?
- T2 / T3a / T3b có giữ không?
- Macro lệch seed 1 bao nhiêu?

## 11. Dự đoán và cách đọc, ghi trước khi chạy 2b-s2 (28/10, khoảng 19:50; 2b-s2 chưa chạy)

Điều phối tự quyết theo ủy quyền "bạn chủ động quyết". Người dùng có thể đổi. Nhiễu giữa seed chưa đo; dải dưới đây lấy từ checkpoint trong **một** lần chạy C (`fable_2b/g4_out.txt`): macro 80,17–81,65; XN+BB ViWikiFC 95–106/120; precision NEI 0,76–0,89.

**Dự đoán:**
- VI macro khoảng 84–85 (dải 82,5–86). |s2 − s1| ≤ 2 với xác suất khoảng 80%. Mỗi 4 câu NEI choice ViFactCheck ≈ 0,4 điểm macro. Không đạt 86,5 vì seed không đổi dữ liệu.
- Biên của từng luật, tính theo số câu:
  - T3b: 96/120, sàn 94 → biên 2 câu; dễ trượt nhất (khoảng 30%).
  - G1: biên 0,88 ≈ 8 câu choice ViFactCheck không dấu, hoặc 6 câu XCOPA (khoảng 25–35%).
  - T3a: 61/79 → trượt nếu mất 6 lần đoán NEI đúng, hoặc thêm 9 lần đoán NEI sai (khoảng 10%).
  - G3, G4: khoảng 10% mỗi luật. T1, T2, T4, G2, G5: biên rộng.
- Đạt lại đủ 10 luật: khoảng 40–55%.
- Bác bỏ ViWikiFC tụt và "NEI mặc định" khi không dấu: dự đoán lặp lại (khoảng 2/3), vì nguyên nhân nằm ở dữ liệu.

**Cách đọc khi s2 trượt một luật:**
- Quyết đi/dừng theo **trung bình 2 seed của 2b so với C-s1**, đúng ý mục 6 bậc 2 nhưng không có C-s2.
- **s2 trượt nhưng trung bình 2 seed đạt, và mức trượt nằm trong dải checkpoint của C:** ghi là nhiễu seed, không dừng 2b. Đưa luật đó vào mục tiêu sửa dữ liệu.
- **Trung bình 2 seed cũng trượt:** luật trượt thật. Báo người dùng trước khi làm bước tiếp.

**Các nhánh:**
- |Δ| ≤ 2 và giữ đủ luật → 2b thành mốc mới. Bước tiếp là sửa dữ liệu cho lỗi Bác bỏ ở độ trùng từ thấp và lỗi không dấu, chưa sang 2c.
- |Δ| > 2 → chạy C-s2 (mục 10).
- Hai chi phí lặp lại → lỗi có hệ thống, đúng mục tiêu sửa. Không lặp lại → coi là chưa chứng minh.
- Test: giữ kỳ vọng +1,5 đến +3,5 của Fable. s2 chỉ đo nhiễu seed, không đo khả năng chuyển sang dữ liệu khác.

### 11b. Chấm dự đoán mục 11 (29/10, 05:45, sau khi có số 2b-s2)

| Dự đoán | Thực tế | Đúng/sai |
|---|---|---|
| VI 84–85 (dải 82,5–86) | 83,92 | dưới khoảng trung tâm, trong dải |
| \|s2 − s1\| ≤ 2 (khoảng 80%) | 0,92 | đúng |
| Dễ trượt nhất là T3b (khoảng 30%), rồi G1 (25–35%) | T3b 84/120 trượt (TB 90/120 cũng trượt); G1 62,34 trượt 0,13 (TB đạt) | đúng |
| T3a trượt khoảng 10% | 0,679 trượt (TB gộp 0,726 đạt) | sai, mức trượt nằm ngoài dải checkpoint của C (0,76–0,89) nên **không xếp là nhiễu seed** |
| Đạt đủ 10 luật: 40–55% | 7/10 (s2), 9/10 (TB) | không đạt đủ |
| Hai chi phí lặp lại (khoảng 2/3) | Bác bỏ ViWikiFC 0,608; NEI không dấu 97/161 | đúng, và nặng hơn s1 |

**Áp luật mục 11:**
- T3b trượt cả ở trung bình 2 seed → luật trượt thật. 2b không được nhận nguyên trạng, báo người dùng trước bước tiếp.
- T3a: trung bình đạt, nhưng s2 trượt ngoài dải nhiễu đã ghi → ghi là "chưa chắc", không phải nhiễu.
- G1: giữa hai seed, 25–32% câu không dấu ở bốn nguồn ngoài kiểm chứng đổi đáp án → G1 với biên dưới 1 điểm **không đọc được**, cả đạt lẫn trượt.

**Phân rã hiệu TB 2 seed − C-s1 (macro có dấu +2,73):**
- ViFactCheck 88,6 − 72,2 = +16,4 → +2,73 macro, tức **toàn bộ** mức tăng.
- ViWikiFC 79,45 − 80,7 = −1,25 → −0,21 (s1 +2,1; s2 −4,6).
- SIB +0,33; MASSIVE +0,20; Belebele −0,17; XCOPA −0,17.

**Nhiễu seed đo được:** VI macro 0,92; ViWikiFC 6,7 (38/285 câu đổi đáp án); ViFactCheck 2,5. Hệ quả: mọi ứng viên sau mà chênh dưới khoảng 2 điểm macro so với 2b đều phải chạy 2 seed, tức khoảng 8 giờ GPU cho mỗi ứng viên.

**Proxy:** kiểm chứng có dấu s1 80,0, s2 83,3 (C 75,0), ngược chiều với ViWikiFC dev → một phần mức tụt trên dev là tương tác giữa seed và mẫu dev. Nhưng chỉ báo (c) (Bác bỏ 0,608) khớp đúng cơ chế Fable đã chỉ: KC có 0/337 dòng Bác bỏ ở bao phủ < 0,7.

## 12. 2b′: đăng ký trước luật đọc (29/10, khoảng 16:00, trước mọi lần train 2b′)

**Dữ liệu:** biến thể V-e của `results/train_2b_prime/train_2b_prime.jsonl`, sha ghi trong manifest sau khi dựng. Thiết kế ở `results/BRIEF_2b_prime_du_lieu.md` và `results/huong_di_2710/FABLE_truoc_train_2b_prime.md`.

**Sai lệch có ngày so với brief 2b′:**
- C2(b) đổi sang hash mù trên mọi dòng 2d. Lý do: ưu tiên S/R trùng thấp làm mất 117 dòng Bác bỏ trùng thấp có dấu.
- Cổng D11(a) chuyển thành chỉ báo. Cổng đạt/trượt mới: probe riêng ViANLI ≤ đa số + 10. Lý do: một khối ≤ 5% về số học không thể kéo probe của khối gộp xuống, trừ khi tự nó tương quan ngược (đó là lần 2, đã bỏ).
- Lần 1 trượt D11(a) cũ (0,725 > 0,681). Ghi lại, không xóa.

**Luật đọc:** dùng nguyên mục 4 của `FABLE_truoc_train_2b_prime.md`, không sửa.
- Bậc 1, s1 chỉ để dừng: S1 VI ≥ 83,4; S2 NEI ViFactCheck ≥ 0,70; S3 "Có" noul ≥ 0,80; S4 acc ViFactCheck ≥ 85,0. Đạt thì chạy s2.
- Bậc 2, trung bình 2 seed: A1 VI ≥ 84,38 (vùng xám [83,9; 84,38)); A2 "Có" noul ≥ 0,867; A3 T3b ≥ 94/120; A4 NEI ViFactCheck ≥ 0,75 và acc ViFactCheck ≥ 86,0; A5 G1 ≥ 62,85; A6 EN ≥ 87,20. A4′ (T2/T3a) chỉ là chỉ báo.
- Quyết nền (2b hay 2b′) theo bảng "Quyết định" của Fable.
- **Hướng 1 kết thúc sau lần chạy này, dù kết quả thế nào.** Ngoại lệ duy nhất là lỗi khung lặp lại. Không thử 2b″.
- Không đọc "> 86,5" từ dev trừ khi TB ≥ 87,2. Không chấm test cho 2b′.

**Kỳ vọng ghi trước** (Fable, TB 2 seed):
- dev VI 84,6–85,4 (trung tâm 84,9; 2b 84,38);
- Bác bỏ ViWikiFC 0,72–0,80;
- "Có" noul 0,85–0,90;
- NEI ViFactCheck 0,76–0,82;
- không dấu 63,5–65.

86,5 nằm ngoài tầm của hướng 1. Hai seed chỉ phát hiện được chênh ≥ 1,5–2 điểm.

**Điều kiện trước GPU:**
- người dùng duyệt train;
- đọc mù 60 dòng nếu người dùng duyệt (luật dừng ở mục 5 của Fable);
- Ún nhường GPU;
- lính gác nhiệt v2 chạy;
- session dọn ổ đĩa không chạy `wsl --shutdown` trong lúc train.
