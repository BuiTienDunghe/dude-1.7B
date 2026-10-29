# Train thử bỏ dấu ngẫu nhiên (strip_v0), một lần train

Ngày 2026-10-13. Qwen3-1.7B, seed 1. Chỉ chấm trên dev_v1; không có model nào chạy trên test 1.0. Bảng đầy đủ ở `tables_strip_v0.md`, số máy đọc được ở `metrics_strip_v0.json`.

## Trả lời hai câu hỏi

1. **Điểm tiếng Việt không dấu tăng bao nhiêu?** So với adapter đối chứng, trên cùng câu:
   - song song: **+2,0** [−1,5; +5,4];
   - bản địa: **+3,0** [−0,6; +6,4].

   Cả hai đều dưới mốc 5 điểm và CI đều chứa 0. Ở khối bản địa, phần tăng gần như chỉ đến từ một nguồn là ViHSD (xem mục 4.2).
2. **Điểm VI có dấu và tiếng Anh có giảm không?** Không. Chênh lệch là +1,1 (VI có dấu song song), +0,1 (VI có dấu bản địa) và +0,4 (EN), cả ba đều trong vùng nhiễu 1,5 điểm.

**Theo ngưỡng đặt trước: KHÔNG ĐẠT.** Hai tiêu chí chính (không dấu tăng ≥ 5 điểm, CI không chứa 0) đều trượt. Ba tiêu chí giữ (giảm ≤ 2 điểm) đều đạt.

---

## 1. Đã chạy gì

### 1.1 Công thức

Giống hệt lần đối chứng `results/ft_loss/train/Qwen3-1.7B-s1/`. Đã so `summary.json` của hai lần: trùng revision `70d244cc…`, seed 1, lr 2e-4, 2 epoch, 1.250 bước, warmup 37, LoRA r16/α32/dropout 0,05 trên 7 ma trận, batch 32 = 8 × 4, `--loss full_vocab`, train_v0 `d10c9362…`, val_v0 `4377e710…`. Checkpoint là cuối epoch 2, T fit trên val_v0. Lỗi đã biết (xáo thứ tự lựa chọn ở câu score) vẫn giữ nguyên.

**Thay đổi duy nhất:** bỏ dấu khi nạp dữ liệu train (`--strip_p 0.3 --strip_state_only_frac 0.5`).
- Chỉ áp cho dòng có `text_lang == "vi"` (4.002 / 20.000 dòng). Dòng `en` (15.595) và `multi` (403) giữ nguyên.
- Mỗi epoch, mỗi dòng tiếng Việt bị bỏ dấu với xác suất 0,30. Quyết định lấy từ `sha256(seed:epoch:id)`, tách khỏi RNG xáo batch.
- Một nửa số dòng bị bỏ dấu ở cả state, câu hỏi và lựa chọn (chế độ "all"). Nửa còn lại chỉ bỏ dấu ở state ("state_only").
- Hàm bỏ dấu chính là hàm của bench_v1, đã chuyển sang `scripts/common.py`.

| Epoch | Dòng VI | Không đổi | state_only | all | Tỉ lệ bỏ dấu |
|---|---|---|---|---|---|
| 0 | 4.002 | 2.796 | 605 | 601 | 30,1% |
| 1 | 4.002 | 2.787 | 644 | 571 | 30,4% |

Số dòng bỏ dấu toàn bộ (đúng điều kiện mà lát "VI không dấu" kiểm tra) là khoảng 600 mỗi epoch, tức **3% tập train**.

### 1.2 Cổng trước khi train (đều đạt)

- **G0, hàm bỏ dấu sau khi chuyển file.** Dựng lại bản không dấu vào `results/ft_strip/tmp_regen/`. Kết quả trùng SHA-256 với file đã đóng băng: test 1.0 `d92d31f6…0d47`, dev_v1 `89455863…e2b4e2`. Không ghi gì vào `data/bench_v1`.
  - Đây là lần duy nhất file test 1.0 được mở, và chỉ để xử lý chuỗi rồi so hash, không có model nào chạy. Ngoài `regen_nodiac_check.py`, không script nào của lần này nhắc tới test_v1.
- **G1, đường chấm không đổi.** Chấm lại adapter đối chứng trên dev_v1: preds trùng từng byte với lần chấm cũ (sha `1468dcca…`).
- **G2, đường dữ liệu không đổi.**
  - strip_p 0: id và token của 200 bước đầu trùng 200/200 với đường dữ liệu cũ.
  - strip_p 0,3: id theo thứ tự trùng 200/200. Token chỉ khác ở 170 bước có câu bị bỏ dấu (tổng 365 câu).
  - Có một sự cố ở cổng này, xem mục 6.
- **G3, kiểm hàm bỏ dấu.**
  - 0 dòng không phải tiếng Việt bị đổi; 0 dòng còn dấu sau khi bỏ.
  - Các phép thử đ/Đ, chữ hoa và Unicode dạng tổ hợp (NFD) đều đúng. Train có 189 dòng tiếng Việt ở dạng tổ hợp.
  - Đã in 5 ví dụ trước/sau (`g3_strip_check.json`).

### 1.3 Trong khi train

- Thời gian 112,0 phút (đối chứng 110,3). Đỉnh VRAM 10,25 GB, như đối chứng.
- **Cổng letter_mass đạt.** Ở bước 0 có 42/200 câu dưới 0,5 (trung bình 0,786). Từ bước 125 đến 1.250 là 0/200 (trung bình 0,994–0,999). Trên dev_v1 cũng không câu nào dưới 0,5 ở mọi lát.
- **Loss val** (cross-entropy toàn từ vựng):

| Thời điểm | val (strip) | val bỏ dấu toàn bộ dòng VI (strip) | val (đối chứng) |
|---|---|---|---|
| bước 0 | 7,018 | 7,037 | 7,018 |
| hết epoch 1 | 0,756 | 0,795 | 0,756 |
| hết epoch 2 | 0,778 | 0,817 | 0,776 |

  - Loss val gốc của hai lần gần như trùng nhau. Cả hai đều tăng nhẹ từ epoch 1 sang epoch 2, đúng mẫu cũ.
  - Lần đối chứng không ghi val bỏ dấu, vì cờ này mới thêm. Do đó không so được khoảng cách 0,039 giữa val bỏ dấu và val gốc với đối chứng.
- **Nhiệt độ:** T = 1,535 (đối chứng 1,560).
  - Dòng "eval val accuracy 0.0" trong `pipeline.log` không phải lỗi. val_v0 dùng target mềm nên eval_base không tính accuracy; lần đối chứng cũng in đúng như vậy.

### 1.4 Chấm

- Chỉ chấm trên dev_v1 (hash khớp manifest, `894558631429840e…`), batch 1.
- Adapter mới và adapter đối chứng đều được chấm ở hai thứ tự lựa chọn (thường và đảo).
- decider-2b và Qwen3-1.7B gốc dùng preds đã lưu từ phần 5a, chỉ có thứ tự thường, nên không có số nhất quán khi đảo.

---

## 2. Số đo

### 2.1 Cách gộp

Accuracy gộp theo **trung bình đều theo nguồn**, đúng như brief yêu cầu. Vì vậy số của adapter đối chứng ở đây **khác** `results/dev_v1/tables_dev_v1.md`, nơi gộp theo dòng.
- Ví dụ EN của đối chứng: gộp theo nguồn là 86,4, gộp theo dòng là 85,5.
- Số theo từng nguồn thì trùng khớp hai bên, ví dụ ViHSD không dấu 34,3, MASSIVE không dấu 62,7.

CI 95% lấy bằng bootstrap theo state. McNemar tính trên từng dòng, gộp mọi nguồn.

### 2.2 Năm lát chính, accuracy (%)

| Lát | n | strip (mới) | đối chứng | decider-2b | 1.7B gốc |
|---|---|---|---|---|---|
| song song · EN | 1000 | 86,8 | 86,4 | 89,2 | 83,4 |
| song song · VI có dấu | 1000 | 80,8 | 79,8 | 86,0 | 72,9 |
| song song · VI không dấu | 1000 | 60,8 | 58,8 | 49,2 | 48,0 |
| bản địa · có dấu | 763 | 66,3 | 66,2 | 66,1 | 53,9 |
| bản địa · không dấu | 763 | 56,9 | 53,9 | 42,9 | 47,6 |

### 2.3 Chênh lệch theo cặp, cùng câu (điểm, [CI 95%], McNemar p)

| Lát | strip − đối chứng | strip − decider-2b | strip − 1.7B gốc |
|---|---|---|---|
| song song · EN | +0,4 [−0,7; +1,5] (0,75) | −2,3 [−4,4; −0,3] (0,005) | +3,4 [+1,2; +5,6] |
| song song · VI có dấu | +1,1 [−1,1; +3,1] (0,24) | −5,2 [−8,1; −2,2] (5e-5) | +7,9 [+5,0; +10,6] |
| song song · VI không dấu | +2,0 [−1,5; +5,4] (0,13) | +11,7 [+7,6; +15,8] | +12,8 [+8,0; +17,1] |
| bản địa · có dấu | +0,1 [−2,1; +2,3] (0,91) | +0,2 [−3,8; +4,2] (0,33) | +12,4 [+8,5; +16,6] |
| bản địa · không dấu | +3,0 [−0,6; +6,4] (**0,044**) | +14,0 [+9,6; +18,6] | +9,3 [+5,1; +13,5] |

**Về ô "bản địa · không dấu":** McNemar cho p = 0,044, nhưng CI bootstrap chứa 0. Hai phép thử đo khác nhau.
- McNemar tính trên từng dòng, coi mọi dòng độc lập.
- Bootstrap gộp theo nguồn và lấy mẫu theo state.

Tiêu chí đặt trước dùng CI, nên kết luận vẫn là không đạt.

**Kiểm độ vững theo cách gộp:** nếu gộp theo dòng (tính lại từ preds), mức tăng không dấu là +2,2 (song song: 61,6 so với 59,4) và +3,8 (bản địa: 56,1 so với 52,3). Cả hai vẫn dưới 5, nên kết luận không phụ thuộc cách gộp.

### 2.4 Hiệu chỉnh và nhất quán

| Lát | Brier T=1 strip / đối chứng | Brier sau T | ECE T=1 | ECE sau T | Nhất quán khi đảo thứ tự |
|---|---|---|---|---|---|
| song song · EN | 0,203 / 0,213 | 0,201 / 0,208 | 0,054 / 0,063 | 0,043 / 0,045 | 92,4 / 93,0 |
| song song · VI có dấu | 0,284 / 0,304 | 0,278 / 0,290 | 0,067 / 0,093 | 0,061 / 0,036 | 90,1 / 89,4 |
| song song · VI không dấu | 0,540 / 0,568 | 0,524 / 0,538 | 0,108 / 0,147 | 0,041 / 0,053 | 78,6 / 77,1 |
| bản địa · có dấu | 0,456 / 0,500 | 0,429 / 0,455 | 0,125 / 0,180 | 0,054 / 0,095 | 96,3 / 95,8 |
| bản địa · không dấu | 0,592 / 0,629 | 0,550 / 0,570 | 0,155 / 0,230 | 0,098 / 0,149 | 95,3 / 91,2 |

- Brier của adapter mới thấp hơn đối chứng ở cả 5 lát, cả trước lẫn sau T. Chưa kiểm định thống kê cho Brier.
- letter_mass trung bình từ 0,999 đến 1,000 ở cả hai adapter, 0 câu dưới 0,5.

### 2.5 Theo nguồn, lát không dấu (strip − đối chứng, McNemar p)

| Nguồn | strip | đối chứng | 1.7B gốc | Chênh |
|---|---|---|---|---|
| Belebele | 60,7 | 58,3 | 46,0 | +2,3 (0,41) |
| MASSIVE | 66,7 | 62,7 | 43,7 | +4,0 (0,13) |
| SIB-200 | 59,0 | 58,3 | 51,3 | +0,7 (0,89) |
| XCOPA | 57,0 | 56,0 | 51,0 | +1,0 (1) |
| ViFactCheck | 61,3 | 62,6 | 51,5 | −1,2 (0,86) |
| ViWikiFC | 63,3 | 64,7 | 47,7 | −1,3 (0,70) |
| **ViHSD** | **46,0** | **34,3** | **43,7** | **+11,7 (0,0006)** |

Ở bản có dấu, mọi nguồn đều chênh trong khoảng −1,3 đến +3,3 và không nguồn nào có p < 0,05.

### 2.6 Theo kiểu câu hỏi (khối bản địa, trung bình đều theo nguồn)

| Kiểu · dấu | n | strip | đối chứng | strip − đối chứng [CI] (p) |
|---|---|---|---|---|
| choice · có dấu | 256 | 62,7 | 63,3 | −0,6 [−4,3; +3,5] (0,68) |
| choice · không dấu | 256 | 51,0 | 49,0 | +2,1 [−4,4; +8,7] (0,61) |
| noul · có dấu | 363 | 77,5 | 76,8 | +0,6 [−2,7; +4,0] (0,51) |
| noul · không dấu | 363 | 70,0 | 66,3 | +3,7 [−1,1; +8,9] (0,035) |
| score · có dấu | 144 | 42,4 | 42,4 | 0,0 |
| score · không dấu | 144 | 31,9 | 31,2 | +0,7 |

### 2.7 Chẩn đoán phụ: chỉ bỏ dấu ở state

Bản dev_v1 tiếng Việt trong đó chỉ state mất dấu, còn câu hỏi và lựa chọn giữ dấu. File `diag/dev_v1_vi_state_only_nodiac.jsonl` có 1.763 dòng, sha `4b87e329…`. Đây không phải số chính thức.

| Khối | Model | có dấu | chỉ state không dấu | không dấu toàn bộ |
|---|---|---|---|---|
| song song | strip | 80,8 | 63,8 | 60,8 |
| song song | đối chứng | 79,8 | 62,3 | 58,8 |
| song song | decider-2b | 86,0 | 55,6 | 49,2 |
| bản địa | strip | 66,3 | 59,7 | 56,9 |
| bản địa | đối chứng | 66,2 | 55,8 | 53,9 |
| bản địa | decider-2b | 66,1 | 53,6 | 42,9 |

decider-2b có 10 dòng của bản chẩn đoán dài quá 1.536 token (mức cắt mặc định của decider). Giống phần 5a, giới hạn đã được nâng lên 32.768 token, nên không dòng nào bị cắt.

---

## 3. Đánh giá theo ngưỡng đặt trước

| Tiêu chí | Đo được | Kết quả |
|---|---|---|
| Không dấu song song: tăng ≥ 5, CI không chứa 0 | +2,0 [−1,5; +5,4] | **Không đạt** |
| Không dấu bản địa: tăng ≥ 5, CI không chứa 0 | +3,0 [−0,6; +6,4] | **Không đạt** |
| VI có dấu song song: giảm ≤ 2 | +1,1 (nhiễu) | Đạt |
| VI có dấu bản địa: giảm ≤ 2 | +0,1 (nhiễu) | Đạt |
| EN: giảm ≤ 2 | +0,4 (nhiễu) | Đạt |
| **Tổng** | | **KHÔNG ĐẠT** |

---

## 4. Diễn giải

Mọi điểm trong mục này là cách đọc số, không phải số đo.

1. **Bỏ dấu ngẫu nhiên ở mức này không gây hại, nhưng lợi ích nhỏ.**
   - Có dấu và EN không đổi.
   - Ở 6/7 nguồn, không dấu chỉ tăng từ −1,3 đến +4,0 điểm. Hướng chung là dương ở bốn nguồn song song, nhưng không nguồn nào có ý nghĩa thống kê.
2. **Mức tăng ở khối bản địa thực chất là ViHSD.**
   - ViHSD +11,7 (p = 0,0006). ViFactCheck và ViWikiFC giảm nhẹ (−1,2 và −1,3). Bỏ ViHSD thì khối bản địa không có mức tăng nào.
   - Adapter đối chứng ở ViHSD không dấu (34,3) **thấp hơn cả model gốc** (43,7). Adapter mới lên 46,0, chỉ hơn model gốc +2,3.
   - Vì vậy nên đọc là "lấy lại phần đã mất do fine-tune trên một nguồn", không phải "hiểu tiếng Việt bản địa không dấu tốt hơn".
   - ViHSD có 300 dòng mỗi biến thể: 156 câu noul và 144 câu score; toàn bộ câu score của khối bản địa là ViHSD. Tách ra (gộp theo dòng):
     - ViHSD noul không dấu: 59,0 so với 37,2, tức +21,8. Có dấu: +4,5.
     - ViHSD score không dấu: +0,7.
     - Khối bản địa bỏ ViHSD (463 dòng): không dấu −1,3, có dấu −1,1.
   - Như vậy mức +3,7 của kiểu noul (p = 0,035) chủ yếu đến từ phần ViHSD trong noul.
3. **Khoảng cách có dấu / không dấu vẫn còn khoảng 20 điểm ở khối song song**: 80,8 so với 60,8, trước đây là 79,8 so với 58,8.
   - Bản chẩn đoán cho thấy phần lớn khoảng cách đến từ **state**. Chỉ bỏ dấu ở state đã làm mất 17,0 điểm. Bỏ thêm dấu ở câu hỏi và lựa chọn chỉ mất thêm 3,0 điểm.
   - Bản địa: 6,6 điểm do state, thêm 2,8 điểm do câu hỏi và lựa chọn.
   - Nghĩa là điểm nghẽn là hiểu đoạn văn dài không dấu, chứ không phải hiểu câu hỏi.
4. **Giả thuyết vì sao hiệu ứng nhỏ.**
   - Chỉ khoảng 3% tập train (khoảng 600 dòng mỗi epoch) ở đúng điều kiện bỏ dấu toàn bộ. Tính cả state_only thì khoảng 6%.
   - Với tín hiệu chỉ là một chữ cái trả lời, chừng đó có thể không đủ để LoRA học đọc văn bản không dấu.
   - Đây là giả thuyết, chưa kiểm.
5. **Hiệu chỉnh** của adapter mới tốt hơn chút ở mọi lát. Trên bản địa không dấu, tỉ lệ nhất quán khi đảo thứ tự tăng từ 91,2 lên 95,3.
6. **So với decider-2b:**
   - Mô hình mới vẫn kém trên VI có dấu song song (−5,2) và EN (−2,3).
   - Vẫn hơn rõ trên không dấu (+11,7 và +14,0). Lợi thế này đã có từ adapter đối chứng, không phải do bỏ dấu.

## 5. Điều chưa chắc

- **Chỉ có một seed.**
  - Với 1.7B, chênh giữa seed 1 và seed 2 trên tiếng Việt dev_v0 là −0,9 [−3,2; +1,3] (`results/ft/REPORT_ft_v0.md`, lúc đó dùng loss letters_k).
  - Một lần train không tách được +2,0 hay +3,0 khỏi nhiễu giữa các seed. Cũng chưa có seed 2 của adapter đối chứng với loss full_vocab.
- Mức +11,7 ở ViHSD có thể một phần là nhiễu theo seed: ViHSD là nguồn mà adapter đối chứng lệch xa nhất so với model gốc.
- Chênh Brier, ECE và nhất quán chưa được kiểm định thống kê.
- Không đo được val bỏ dấu của adapter đối chứng, vì lần train cũ không có cờ này.
- Kiểu score không kết luận được: lỗi xáo thứ tự lựa chọn câu score cố ý giữ nguyên, và cả ba model (hai adapter và decider-2b) cùng cho 42,4% trên 144 câu.
- decider-2b và model gốc chỉ có thứ tự thường, nên chênh với hai model này có thể lẫn một phần thiên lệch theo vị trí đáp án.

## 6. Sự cố và quyết định trong lúc làm

- **G2 lần đầu trượt.**
  - Công thức cũ sắp 32 câu trong batch theo độ dài token rồi mới chia micro-batch. Câu bỏ dấu có độ dài khác nên thứ tự đổi: tập câu trùng 200/200, nhưng thứ tự chỉ trùng 59/200 và cách chia micro-batch trùng 126/200.
  - Người làm dự án chọn lấy độ dài của câu **trước** khi bỏ dấu làm khóa sắp. Sau khi sửa, G2 đạt (mục 1.2). Đã ghi vào `PLAN_decisions.md`.
- Nhịp đầu train có vẻ chậm hơn (khoảng 6,3 s/bước), nhưng tổng thời gian cuối cùng chỉ hơn đối chứng 1,7 phút.
- Không chạy thêm p hay seed nào khác.

## 7. Đề xuất (chưa chạy, chờ quyết định)

Tham khảo chi phí: một lần train mất khoảng 112 phút. Chấm (thường, đảo, val) mất khoảng 20 phút, thêm khoảng 10 phút nếu chấm bản chẩn đoán.

1. **Đo nhiễu trước khi đổi tham số.**
   - Train seed 2 cho cả adapter đối chứng (full_vocab) và strip p = 0,3: hai lần train, khoảng 4,5 giờ.
   - Chỉ cách này mới cho biết +2/+3 là thật hay nhiễu, và ViHSD +11,7 có lặp lại không.
2. **Tăng phần bỏ dấu toàn bộ**, ví dụ p = 0,5 với `--strip_state_only_frac 0`. Khi đó khoảng 2.000 dòng mỗi epoch (10% tập train) ở đúng điều kiện được đo, so với 600 hiện tại.
   - Đây là cách kiểm trực tiếp giả thuyết ở mục 4.4.
   - Nên làm sau đề xuất 1, hoặc chạy cùng 2 seed.
3. **Ngoài train**: thử khôi phục dấu trước khi đưa vào model.
   - Giới hạn trên là điểm có dấu (80,8 song song, 66,3 bản địa), cách mức hiện tại 20 và 9 điểm.
   - Bản chẩn đoán cho thấy chỗ cần sửa là state dài, nên đáng đo xem một bộ khôi phục dấu nhỏ làm tốt đến đâu.
   - Hướng này nằm ngoài phạm vi brief, chỉ nêu để cân nhắc.

## 8. File

- `train/Qwen3-1.7B-s1/`: adapter, `train_log.jsonl`, `summary.json`.
- `eval/normal/strip-s1`, `eval/reverse/{strip-s1,control-s1}`, `eval/val/strip-s1`, `eval/diag/{strip-s1,control-s1,decider-2b-text}`: preds.
- `temp/strip-s1.json`: nhiệt độ.
- `tables_strip_v0.md`, `metrics_strip_v0.json`: bảng đầy đủ, do `scripts/ft_strip/report_strip.py` sinh ra.
- `gates/`, `gates.log`, `g3_strip_check.json`, `regen_nodiac_check.json`: các cổng.
- `pipeline.log`: nhật ký pipeline.
- `diag/dev_v1_vi_state_only_nodiac.jsonl`: bản chẩn đoán.
- Code: `scripts/train_lora.py` (cờ `--strip_p`, `--strip_state_only_frac`, `--val_nodiac`, `--dump_batches`), `scripts/common.py` (`strip_diacritics`, `strip_decision`, `apply_strip`), `scripts/ft_strip/*`, `setup/ft_strip_*.sh`.
