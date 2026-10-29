# Hai phân tích độ nhạy trên dev (chỉ dùng preds đã lưu)

Ngày 2026-10-13. Không chạy model nào, không mở test. Bảng đầy đủ ở `sens_tables.md`, số máy đọc được ở `metrics_sensitivity.json`. Code: `scripts/bench11/sensitivity.py`.

## Tóm tắt

1. **Câu chữ ("hỗ trợ" ở 1.0 so với "xác nhận" ở 1.1)**, ViFactCheck + ViWikiFC, 463 câu có dấu:
   - **Chỉ decider-2b tăng có ý nghĩa với chữ mới:** +2,4 [+0,8; +4,2] (McNemar p = 0,008). Bốn model còn lại thay đổi từ −1,0 đến +0,8, CI đều chứa 0.
   - Vì vậy khoảng cách "adapter đối chứng − decider-2b" ở khối bản địa có dấu **phụ thuộc câu chữ:**

     | Cách hỏi | Khoảng cách | CI 95% |
     |---|---|---|
     | Chữ cũ | −4,9 | [−9,4; −0,4] |
     | Chữ mới | −8,3 | [−13,0; −3,8] |
     | Trung bình hai cách hỏi | −6,6 | [−11,2; −2,2] |

     Khoảng 3,4 điểm trong mức chênh −8,3 ở bản 1.1 đến từ việc đổi chữ. decider-2b vẫn hơn ở cả ba cách tính.
   - Ở bản không dấu, khoảng cách gần như không đổi: +16,2 / +16,0 / +16,1.
2. **Ngưỡng noul**, chọn trên 174 câu noul của val_v0, không dùng dev:
   - Adapter đối chứng: t* = 0,57. Adapter bỏ dấu: t* = 0,51.
   - Áp lên câu noul bản địa có dấu của dev_v1.1:
     - adapter đối chứng 84,7 → 86,1 (+1,4 [−0,9; +4,0]);
     - adapter bỏ dấu 84,8 → 85,2;
     - decider-2b ở ngưỡng 0,5 vẫn cao hơn (88,8).
   - Ở bản không dấu, chỉnh ngưỡng làm giảm nhẹ (−0,8 và −0,6).
   - **Chỉnh ngưỡng không thu hẹp đáng kể khoảng cách với decider-2b**, dù phép so sánh này nghiêng về dự án: chỉ adapter của dự án được chỉnh ngưỡng.

---

## 1. Độ nhạy theo câu chữ

### 1.1 Cách làm

- **Chữ 1.0:** preds đã lưu trên dev_v1 (`results/dev_v1/eval/*`; strip lấy ở `results/ft_strip/eval/normal`; decider-ft lấy ở `results/ft_decider/eval/dev_v1_ep2`).
- **Chữ 1.1:** preds đã lưu trên dev_v1.1 (`results/dev_v1_1/eval/*`).
- Ghép theo id (id giữ nguyên giữa hai bản). Chỉ ViFactCheck và ViWikiFC, có dấu và không dấu; ViHSD không tham gia. Hai bản chỉ khác ở câu hỏi (choice và noul) và lựa chọn "Được văn bản hỗ trợ / xác nhận" của câu choice.
- **Chỉ số:**
  - Accuracy trung bình đều theo nguồn, CI 95% bootstrap theo state.
  - Chênh theo cặp trên cùng câu kèm McNemar.
  - "Trung bình hai cách hỏi": lấy trung bình 0/1 của hai lượt trên từng câu, rồi trung bình đều theo nguồn.
- decider-2b và decider-ft dùng ánh xạ "text" như báo cáo chính; ánh xạ "keyed" ở bảng phụ (`sens_tables.md`).
- Kiểm thêm: preds 1.1 của adapter bỏ dấu đúng là bản chấm lại. Xác suất đổi ở 168/207 câu noul có dấu (lệch tối đa 0,23), chỉ là không đổi đáp án được chọn.

### 1.2 Có dấu, mọi kiểu câu (463 câu)

| Model | chữ 1.0 | chữ 1.1 | 1.1 − 1.0 [CI] (p) | trung bình hai cách |
|---|---|---|---|---|
| Qwen3-1.7B gốc | 57,5 | 57,4 | −0,1 [−2,3; +2,4] (0,85) | 57,4 |
| adapter đối chứng | 73,9 | 72,9 | −1,0 [−2,5; +0,5] (0,18) | 73,4 |
| strip (Qwen3) | 72,9 | 73,6 | +0,6 [−0,7; +2,1] (0,45) | 73,3 |
| **decider-2b** | 78,8 | 81,2 | **+2,4 [+0,8; +4,2] (0,008)** | 80,0 |
| decider-ft ep2 | 80,2 | 81,0 | +0,8 [−0,6; +2,2] (0,51) | 80,6 |

Tách theo kiểu câu (có dấu):

| Model | choice: 1.0 → 1.1 (chênh) | noul: 1.0 → 1.1 (chênh) |
|---|---|---|
| Qwen3-1.7B gốc | 39,9 → 37,2 (−2,6) | 78,2 → 81,2 (+3,0) |
| adapter đối chứng | 63,3 → 62,7 (−0,6) | 86,1 → 84,7 (−1,4) |
| strip (Qwen3) | 62,7 → 63,9 (+1,2) | 84,8 → 84,8 (0,0) |
| decider-2b | 72,7 → 74,8 (+2,0; p = 0,04) | 86,1 → 88,8 (+2,7 [0,0; +5,6]) |
| decider-ft ep2 | 72,7 → 74,2 (+1,5) | 89,2 → 89,2 (0,0) |

### 1.3 Không dấu, mọi kiểu câu (463 câu)

| Model | chữ 1.0 | chữ 1.1 | 1.1 − 1.0 [CI] | trung bình hai cách |
|---|---|---|---|---|
| Qwen3-1.7B gốc | 49,6 | 46,3 | −3,3 [−6,5; −0,1] | 48,0 |
| adapter đối chứng | 63,6 | 65,4 | +1,8 [−0,3; +4,0] | 64,5 |
| strip (Qwen3) | 62,3 | 64,0 | +1,6 [−1,2; +4,6] | 63,2 |
| decider-2b | 47,4 | 49,4 | +2,0 [−1,3; +5,4] | 48,4 |
| decider-ft ep2 | 56,9 | 57,4 | +0,5 [−2,2; +3,2] | 57,2 |

Ở bản không dấu, câu noul của decider-2b tăng +5,1 [+0,6; +10,3] (p = 0,076); Qwen3 gốc giảm −5,8 [−10,1; −1,9].

### 1.4 Khoảng cách "adapter đối chứng − decider-2b" ở khối bản địa, theo ba cách tính

| Lát | chữ 1.0 | chữ 1.1 | trung bình hai cách hỏi |
|---|---|---|---|
| có dấu · mọi kiểu | −4,9 [−9,4; −0,4] | −8,3 [−13,0; −3,8] | −6,6 [−11,2; −2,2] |
| có dấu · choice | −9,4 [−16,3; −2,4] | −12,0 [−19,1; −5,0] | −10,7 [−17,5; −3,8] |
| có dấu · noul | 0,0 [−4,9; +4,9] | −4,1 [−9,1; +0,6] | −2,1 [−6,6; +2,3] |
| không dấu · mọi kiểu | +16,2 [+10,5; +21,8] | +16,0 [+10,5; +21,5] | +16,1 [+10,8; +21,4] |
| không dấu · choice | +15,2 [+6,1; +23,9] | +18,7 [+10,0; +26,9] | +17,0 [+8,5; +25,2] |
| không dấu · noul | +16,7 [+8,4; +24,3] | +12,4 [+4,4; +19,8] | +14,5 [+6,7; +21,7] |

## 2. Ngưỡng câu noul (chọn trên val_v0, áp lên dev_v1.1)

### 2.1 Cách làm

- **Dữ liệu chọn ngưỡng:** 174 câu noul của val_v0 (106 tiếng Anh, 67 tiếng Việt, 1 đa ngữ). Preds đã lưu:
  - `results/ft_loss/eval/val/Qwen3-1.7B-fullvocab-s1`
  - `results/ft_strip/eval/val/strip-s1`
- **Chiều Yes/No:** thứ tự lựa chọn trong val_v0 đã xáo lúc dựng: 60 câu [Yes, No], 47 [No, Yes], 37 [Có, Không], 30 [Không, Có]. Vì vậy lựa chọn "Yes"/"Có" được xác định theo **chữ**; đã in kiểm vài dòng (`metrics_sensitivity.json`, mục `direction_checks`). Không có câu nào target hòa 0,5/0,5.
- **Tiêu chí:** quét ngưỡng t của P(Yes) từ 0,01 đến 0,99, bước 0,01; tính accuracy so với argmax của target; chọn t có accuracy cao nhất, hòa thì chọn t gần 0,5 nhất.
- **Áp lên dev:** áp t* lên câu noul bản địa của dev_v1.1. Ở đây lựa chọn cố định là [Có, Không], nên P(Có) = probs[0]. Ngưỡng không được chọn trên dev.

### 2.2 Ngưỡng chọn được

| Model | t* | độ lệch logit | acc val ở 0,5 → ở t* | các ngưỡng cùng đạt tối đa | tỉ lệ dự đoán Yes trên val, 0,5 → t* (đáp án Yes: 44,3%) |
|---|---|---|---|---|---|
| adapter đối chứng | 0,57 | +0,28 | 79,3 → 81,0 | 0,57–0,67 | 43,1 → 35,6 |
| strip (Qwen3) | 0,51 | +0,04 | 79,3 → 82,2 | 0,51–0,53 | 45,4 → 41,4 |

- **Hình dạng đường accuracy theo ngưỡng** (bảng đủ ở `sens_tables.md`): **khá phẳng** trong khoảng 0,45–0,70 (77–82%), giảm dần ra hai phía.
- Mức tăng trên val ở t* là +1,7 và +2,9 điểm, tức chỉ 3 và 5 câu trên 174. Đỉnh của adapter bỏ dấu hẹp (0,51–0,53), nên vị trí t* không chắc.

### 2.3 Áp lên câu noul bản địa của dev_v1.1 (207 câu mỗi biến thể; đáp án "Có" chiếm 38,2%; sàn đa số 61,8%)

| Biến thể | Model | acc trước (0,5) | acc sau (t*) | sau − trước [CI] | dự đoán "Có" trước → sau |
|---|---|---|---|---|---|
| có dấu | adapter đối chứng | 84,7 | 86,1 | +1,4 [−0,9; +4,0] | 45,4% → 43,0% |
| có dấu | strip (Qwen3) | 84,8 | 85,2 | +0,4 [0,0; +1,2] | 47,8% → 47,3% |
| có dấu | **decider-2b (0,5, không chỉnh)** | **88,8** | — | — | 42,5% |
| không dấu | adapter đối chứng | 81,6 | 80,8 | −0,8 [−3,2; +1,4] | 43,5% → 40,6% |
| không dấu | strip (Qwen3) | 77,0 | 76,3 | −0,6 [−2,5; +0,8] | 46,4% → 44,9% |
| không dấu | decider-2b (0,5, không chỉnh) | 69,2 | — | — | 43,5% |

- Hai adapter đều dự đoán "Có" nhiều hơn tỉ lệ đáp án "Có" (45–48% so với 38,2%). Ngưỡng chọn trên val chỉ kéo xuống một chút.
- Sau khi chỉnh, adapter đối chứng vẫn thấp hơn decider-2b 2,7 điểm ở câu noul có dấu (trước khi chỉnh là 4,1).

## 3. Diễn giải

Mọi điểm trong mục này là cách đọc số, không phải số đo.

1. **Một phần lợi thế của decider-2b ở khối bản địa bản 1.1 đến từ câu chữ mới.**
   - Đổi "hỗ trợ" thành "xác nhận" có lợi rõ cho decider-2b (+2,4) nhưng gần như không tác động tới các model khác.
   - Con số công bằng nhất cho "adapter đối chứng − decider-2b" ở bản địa có dấu có lẽ là trung bình hai cách hỏi: **−6,6 [−11,2; −2,2]**. decider-2b vẫn hơn rõ, nhưng ít hơn mức −8,3 trong REPORT_dev_v1_1.
   - Riêng câu noul có dấu, hai model ngang nhau ở chữ cũ (0,0) và chỉ lệch khi đổi chữ (−4,1). Như vậy phần lợi thế chắc chắn của decider-2b ở khối bản địa nằm ở câu choice (−10,7 theo trung bình hai cách hỏi).
2. **Ở bản không dấu, khoảng cách khoảng +16 điểm nghiêng về adapter đối chứng ổn định theo câu chữ.** Kết luận "adapter của dự án chịu không dấu tốt hơn" không phụ thuộc cách hỏi.
3. **Chỉnh ngưỡng noul không đổi bức tranh.**
   - Cả hai adapter vốn đã gần ngưỡng tốt nhất trên val (t* 0,51–0,57).
   - Lợi ích trên dev nhỏ và CI chứa 0, ngoại trừ adapter bỏ dấu tăng +0,4 với CI chạm 0. Lợi ích này cũng không chuyển sang bản không dấu.
   - Hai adapter thiên về "Có" hơn tỉ lệ thật một chút, nhưng đó không phải lý do chính khiến chúng thua decider-2b ở câu noul có dấu.

## 4. Điều chưa chắc và giới hạn của so sánh

- **So sánh ngưỡng nghiêng về dự án.** Chỉ hai adapter của dự án được chỉnh ngưỡng. decider-2b giữ ngưỡng 0,5, vì không có preds noul của decider-2b trên val_v0 và brief cấm chạy model. Dù được lợi thế đó, adapter vẫn chưa bằng decider-2b ở câu noul có dấu.
- **val_v0 khác phân phối với dev bản địa.**
  - Khoảng 61% câu noul của val_v0 là tiếng Anh, lấy từ tasksource với nhiều dạng câu hỏi khác nhau.
  - Câu noul của dev bản địa là kiểm chứng tiếng Việt, và "Không" gộp hai nhãn.
  - Ngưỡng chọn trên val có thể không phải ngưỡng tốt nhất cho dev. Brief cấm chọn ngưỡng trên dev nên tôi không thử.
- **Ngưỡng chọn trên rất ít câu:** 174 câu; mức tăng trên val là 3–5 câu. t* của adapter bỏ dấu nằm trên một đỉnh hẹp.
- **Câu chữ chỉ có hai phiên bản.** Một cách hỏi thứ ba có thể cho thứ tự khác. "Trung bình hai cách hỏi" chỉ là trung bình của hai cách cụ thể này, không phải ước lượng cho mọi cách hỏi.
- **Mỗi model một seed**; preds của decider-ft dùng adapter train trên train_v0, chỉ dùng nội bộ.
