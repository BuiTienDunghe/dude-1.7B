# Phân tích trước khi train: chưng cất và độ nhạy của cổng quyết định (28/10/2026, bản 2)

Điều phối tự tính trên preds dev 1.3 đã có (sha `c6a9a75d…`), kết quả G0 và kết quả chẩn đoán teacher. Không đụng test. Script nằm trong scratchpad của phiên điều phối: `pre_kd_analysis.py`, `cham_soat_mu.py`.

## Kết luận bản 2 (đọc trước)

1. **G0 (mục 7):**
   - Giữ 1.7B: 2B-Base không đạt ngưỡng.
   - 4B gốc chưa train đạt 84,07 nhưng chậm (trung vị 82 ms), nên bị loại.
   - Giữ khung prompt VI.
   - Bản C đạt 40 ms ở trung vị, chưa đạt ở p90 và p95.
2. **Chẩn đoán teacher (mục 8) bác dự đoán ở mục 1 bản 1.** Trên prompt train, Quyet bất đồng với 19% nhãn VI. Soát mù cho thấy nhãn gốc ở kiểm chứng/NLI, chủ đề và ý định thường sai hoặc lệch quy ước, còn ở đọc hiểu thì teacher sai. Vì vậy 2a đáng train, với cách dựng đích đã sửa ở mục 8.
3. **Cổng "+2 trên 2 seed" vẫn kém nhạy (mục 3).** Đây là giới hạn chung của mọi lần so sánh.

## Kết luận bản 1 (giữ để đối chiếu; ý 1–2 đã bị mục 8 bác một phần)

1. **Chưng cất trên chính các prompt C (bước 2a cũ) nhiều khả năng chỉ tăng rất ít.** Hai lý do:
   - Quyet gần như trả lời dứt khoát: entropy chuẩn hóa trung bình chỉ 0,10–0,41. Vì vậy đích mềm của nó gần trùng với nhãn cứng ta đã có.
   - C chưa có prompt đúng khung cho các nguồn mà Quyet hơn ta nhiều nhất: ViFactCheck, Belebele, XCOPA.

   Giá trị thật của teacher là **gán nhãn cho prompt mới** đúng các dạng bài đó, tức bước 2b.
2. **Đề xuất thay bước 2a train bằng "2a′: chẩn đoán teacher"**, chỉ chạy teacher, không train:
   - chi phí khoảng 1 giờ GPU;
   - cho Quyet chấm các prompt choice/noul của C (khoảng 29.500 dòng);
   - đo độ đồng ý với nhãn và entropy theo từng họ dữ liệu.

   Nếu độ đồng ý ≥ 95% và entropy thấp thì bỏ lần train 2a (tiết kiệm khoảng 6–9 giờ GPU) và đi thẳng sang 2b.
3. **Cổng "VI tăng ≥ 2 điểm, trung bình 2 seed" có độ nhạy thấp.**
   - Model không thật sự tốt hơn vẫn lọt cổng khoảng 4–8%.
   - Model tốt hơn thật +2 điểm chỉ có **khoảng 50%** cơ hội qua cổng.
   - Muốn chắc 80% thì mức tăng thật phải khoảng **+3**.

   Sai số chủ yếu đến từ số câu dev, không phải từ seed. Vì vậy chạy thêm seed giúp ít; thêm câu chấm mới giúp nhiều.

## 1. Dư địa: chỗ nào Quyet đúng mà C sai (VI có dấu)

| Nguồn | n | C | Quyet | decider | C sai, Quyet đúng | C đúng, Quyet sai | C sai, cả Quyet và decider đúng |
|---|---|---|---|---|---|---|---|
| Belebele | 294 | 84,0 | 94,2 | 88,1 | 36 (12,2%) | 6 | 27 |
| SIB-200 | 299 | 79,9 | 83,6 | 82,3 | 22 (7,4%) | 11 | 13 |
| MASSIVE | 294 | 88,1 | 96,6 | 96,3 | 29 (9,9%) | 4 | 25 |
| XCOPA | 100 | 85,0 | 93,0 | 86,0 | 12 (12,0%) | 4 | 6 |
| ViFactCheck | 158 | 72,2 | 88,6 | 83,5 | 31 (19,6%) | 5 | 23 |
| ViWikiFC | 285 | 80,7 | 81,4 | 82,8 | 28 (9,8%) | 26 | 20 |

- **Trần lý thuyết** nếu student chép được Quyet hoàn toàn: 89,6 VI có dấu. Nhưng VI không dấu chỉ 64,6, ngang C.
- **ViWikiFC: Quyet không hơn C** (81,4 so với 80,7; số câu "Quyet đúng C sai" xấp xỉ "C đúng Quyet sai"), nên chưng cất không giúp được nguồn này.
- **SIB-200:** dư địa nhỏ.
- **Dư địa lớn nhất:** ViFactCheck, rồi Belebele, XCOPA, MASSIVE.
- **Độ phủ của C:** C có 0 dòng dùng đúng câu hỏi của Belebele, XCOPA hay kiểm chứng. Riêng câu hỏi SIB-200 và MASSIVE thì có (VNTC và ý định du lịch, mỗi loại 1.000 dòng, nhưng với bộ nhãn khác). Chưng cất chỉ chuyển kỹ năng qua những prompt ta đưa vào, nên muốn lấy dư địa này phải có prompt đúng dạng. Đó là bước 2b.

## 2. Tính chất của teacher (Quyet, VI có dấu)

| Nguồn | Số câu sai | Độ tự tin TB khi sai | Sai mà tự tin > 0,9 | Xác suất cho đáp án đúng khi sai | Entropy chuẩn hóa TB |
|---|---|---|---|---|---|
| Belebele | 17 | 0,71 | 4 | 0,19 | 0,24 |
| SIB-200 | 49 | 0,71 | 11 | 0,16 | 0,27 |
| MASSIVE | 10 | 0,61 | 1 | 0,17 | 0,10 |
| XCOPA | 7 | 0,74 | 1 | 0,26 | 0,40 |
| ViFactCheck | 18 | **0,93** | **14** | 0,06 | 0,19 |
| ViWikiFC | 53 | 0,78 | 14 | 0,18 | 0,41 |

- **Ở kiểm chứng, Quyet sai thì sai rất tự tin.** Vì vậy phải giữ quy tắc "teacher bất đồng với nhãn thì dùng nhãn". Với prompt chưa có nhãn, cần lọc theo độ tự tin, kèm kiểm tra thủ công một mẫu.
- **Entropy thấp:** "tri thức mềm" mà Quyet chuyển được trên prompt đã có nhãn là ít.

## 3. Sai số và độ nhạy của cổng

**Sai số đo trên dev 1.3** (bootstrap theo cụm, phân tầng theo nguồn, B = 2.000):

| Đại lượng | VI có dấu | VI không dấu | EN |
|---|---|---|---|
| SE điểm của 1 model | 1,05–1,12 | 1,38 | 1,1 |
| SE hiệu theo cặp giữa 2 model | 0,97–1,20 | 1,36 | 0,80 |
| Độ lệch giữa các mốc 50/75/100% của cùng một lần train | 0,79 | 1,00 | 0,29 |

- **Nhiễu do seed:** lần đo 12/10 cho 1.7B chênh 0,9 điểm giữa hai seed, tức SD khoảng 0,6–1,0.
- **Cổng "trung bình 2 seed mới − trung bình 2 seed cũ ≥ +2"**, tính cả sai số do mẫu câu (SD tổng khoảng 1,2–1,4):

| Mức tăng thật | Xác suất qua cổng |
|---|---|
| 0 | 4–8% |
| +2 | khoảng 50% |
| +3 | khoảng 76–80% |

  Nếu chỉ xét "tốt hơn trên đúng bộ dev này" (bỏ sai số do mẫu câu): mức tăng thật 0 thì gần như không lọt cổng; +2,5 thì qua cổng 70–80%.
- **Không dấu ≥ 61:** C đạt 63,97, nhưng SE 1,38 và các mốc của cùng một lần train lệch 61,4–65,0. Một model có không dấu thật khoảng 62 có thể rớt cổng khoảng 25% chỉ vì nhiễu.

**Hệ quả cho thiết kế:**
- Báo cả CI theo cặp, không chỉ điểm.
- Coi +2 là ngưỡng "đáng theo", không phải bằng chứng chắc chắn.
- Nếu muốn quyết chắc hơn: cộng thêm tập chọn model (proxy_v1, khoảng 900 câu VI) vào phép so. Việc này giảm sai số do mẫu câu hiệu quả hơn chạy thêm seed. Người dùng quyết vì proxy còn dùng để chọn mốc.

## 4. Bằng chứng ngoài về phần do cỡ model

Họ decider cùng công thức (README v11):
- decider-2b: 0,802 trong task / 0,752 ngoài task;
- decider-4b: 0,831 / 0,784.

Tức là cùng công thức, lên 4B được khoảng **+3 điểm** trên bộ tiếng Anh của họ. Con số này khớp ước lượng "khoảng 3/8 khoảng cách với Quyet là do cỡ". Đây là bằng chứng trên bài tiếng Anh, không phải tiếng Việt.

## 5. Chi phí

- **Teacher chấm C:** khoảng 29.500 dòng choice/noul. Tốc độ đo được của Quyet là khoảng 0,13 s/dòng ở batch 1 (dev: 497 s cho 3.847 dòng), nên mất khoảng **1 giờ GPU**, ít hơn nếu gom batch.
- **Một lần train trên C:** 2,95 giờ, hoặc khoảng 1,5–2,3 giờ nếu tăng tốc 1,3–2×.

## 6. Bài học từ decider

Chi tiết trong `HOC_TU_DECIDER.md`, do agent Opus đọc README v11 và GitHub. Điểm 86,5 của decider-2b là bản v11 (revision `ff99089b…`), điều phối đã kiểm.

- **Chưng cất của decider (KL về v10 trên 20.100 dòng) không làm tăng độ chính xác.** Hai nhánh hòa nhau trong khoảng 0,002. Giá trị của nó là giữ hiệu chỉnh và chống quên. Điều này khớp với mục 1–2: chưng cất trên dữ liệu đã có nhãn khó cho thêm điểm. Đòn bẩy nằm ở dữ liệu mới.
- **Kiểu tăng cường "từ chối trả lời" nhắm thẳng vào lỗi P4 của ta:**
  - 10% số câu được tăng cường: 75% thêm lựa chọn "abstain" nhưng giữ nguyên đáp án, 25% thay cả bộ lựa chọn bằng nhãn của một task không liên quan.
  - Kết quả: câu lạc đề 0,68 → 0,83, task chung không đổi.
  - Cỡ model không chữa được lỗi này: decider-4b còn kém 2B ở các phép thử abstain.
  - Với ta, cách tương ứng là ghép nhận định với văn bản không liên quan, đáp án "không đủ thông tin". Chỉ cần code, không có chữ mới.
- **Quy mô:** decider dùng khoảng 95 bộ, mỗi bộ tối đa 20.000 dòng, tổng 1,47 triệu dòng × 1 epoch. Mỗi lần gấp đôi cỡ model được khoảng +3 điểm (0,8B → 2B → 4B).
- **RL không cần cho bài của ta.** Reward của họ là kết quả trình duyệt và game, và độ chính xác chung còn giảm 0,4.
- **Khác biệt còn để ngỏ:**
  - decider tính CE chỉ trên logit chữ cái, ta tính trên toàn từ vựng;
  - decider xáo cả câu Có/Không, ta giữ cố định theo quyết định 12/10 vì dev cố định thứ tự.

  Hai điểm này chưa đáng thử trước dữ liệu.

## 7. G0, số sơ bộ

Điều phối tự tính lúc 02:00 ngày 28/10, từ preds của worker, trước khi có REPORT_G0. Đây là điểm zero-shot: model gốc, chưa train, dùng khung prompt của dự án.

| Model | VI có dấu | Không dấu | EN | Belebele / SIB / MASSIVE / XCOPA / ViFactCheck / ViWikiFC |
|---|---|---|---|---|
| Qwen3-1.7B gốc (M0) | 70,65 | 48,45 | 84,59 | — |
| Qwen3.5-2B gốc (post-trained) | 70,6 | 39,0 | 84,5 | — |
| **Qwen3.5-2B-Base** | **68,20** | 37,78 | 83,91 | 75,2 / 67,2 / 71,4 / 76,0 / 63,9 / 55,4 |
| **Qwen3.5-4B gốc** (nền của Quyet) | **84,07** | 60,57 | 90,48 | 83,0 / 83,3 / 87,8 / 96,0 / 77,2 / 77,2 |
| C (1.7B đã train) | 81,65 | 63,97 | 88,20 | — |
| decider-2b (2B-Base đã train) | 86,50 | 49,64 | 90,33 | — |
| Quyet-Medium (4B đã train) | 89,57 | 64,60 | 94,46 | — |

**Đọc theo các ngưỡng đã chốt trong brief:**
- **Nền 2B:** 2B-Base thấp hơn M0 2,4 điểm, nên **không đạt** ngưỡng "+3" → **giữ 1.7B**.
- **Điểm của Quyet đến từ đâu:** 4B gốc đạt 84,07, nằm giữa 75 và 85 → **"trộn lẫn", nhưng sát phía "do cỡ"**.
  - 4B **chưa train** đã hơn C (đã train) 2,4 điểm.
  - Fine-tune của Quyet chỉ thêm +5,5 điểm (84,1 → 89,6).
  - Fine-tune của ta thêm +11 điểm (70,7 → 81,7). Của decider thêm +18,3 (68,2 → 86,5).
- **Hai kết luận rút ra:**
  - **Cỡ model có tác động lớn.** Ngay ở mức zero-shot, 4B hơn 1.7B tới 13,4 điểm.
  - **Công thức train cũng quan trọng.** Cùng cỡ khoảng 2B, decider lên được 86,5, còn ta mới 81,7. Khoảng cách công thức vào cỡ **+5 đến +7 điểm**.
- **Khung prompt** (VI có dấu, khung EN so với khung VI):
  - Qwen3-1.7B +1,44;
  - Qwen3.5-2B +1,14;
  - Qwen3.5-2B-Base −3,10.

  Không model nào đạt ngưỡng "+2 ở 2/3 model" → **giữ khung VI**.
- **Hệ quả:** nếu 4B chạy được trong ngưỡng độ trễ người dùng chọn, thì 4B gốc đã gần chỉ tiêu VI 84 mà chưa cần train. Vì vậy số đo độ trễ là biến quyết định.

**Độ trễ, theo REPORT_G0** (batch 1, ms; số eager là số chính thức):

| Model | Eager: trung vị / p90 / p95 | Graphs, chỉ THAM KHẢO vì chưa qua cổng đúng: trung vị / p90 / p95 |
|---|---|---|
| C (1.7B) | 31,8 / 47,3 / 85,4 | 24,9 / 49,1 / 102,1 |
| Qwen3.5-2B-Base, không FLA | 53,2 / 70,3 / 117,0 | 31,9 / 50,5 / 125,7 |
| Qwen3.5-4B, không FLA (tập con) | 82,0 / 138,0 / 263,8 | 67,9 / 126,5 / 313,5 |

**Đọc kết quả:**
- **4B bị loại** dưới mọi cách hiểu ngưỡng 40 ms: ngay trung vị cũng 68–82 ms. Còn thiếu số đo có FLA (đang chạy), nhưng FLA khó kéo trung vị xuống dưới 40 ms.
- **CUDA graphs chỉ giúp câu ngắn.** Với C, trung vị giảm từ 31,8 xuống 24,9 ms, nhưng p90 và p95 không giảm, vì pad theo nhóm độ dài làm câu dài tốn thêm tính toán.
- Với p90 và p95, phần quyết định là lượng tính toán cho văn bản dài (ViFactCheck). Muốn hạ p95 thì phải cắt độ dài state lúc suy luận, hoặc định nghĩa ngưỡng theo trung vị hay p90.
- **Đường graphs của worker có lỗi riêng:** khác eager có pad ở 5 câu argmax. Việc sửa để sau, không ảnh hưởng các kết luận trên.

## 8. Chẩn đoán teacher trên prompt train C (bản 2, 28/10 khoảng 04:40)

**Nguồn:**
- `results/teacher_diag/REPORT_chan_doan.md`;
- `quyet_C.jsonl` (29.523 dòng có xác suất, 1 dòng bị từ chối). Điều phối đã tự tính lại và số khớp.

**Chi phí:**
- 59 phút GPU, nhiệt độ cao nhất 78°C.
- Đầu vào chỉ là prompt train có dấu, không có prompt dev hay test.

| Phần | n | A (teacher đồng ý nhãn) | S (độ mềm khi đồng ý) | D90 | Entropy |
|---|---|---|---|---|---|
| VI | 15.027 | **0,811** | **0,138** | 0,016 | 0,42 |
| EN | 14.154 | 0,688 | 0,210 | 0,022 | 0,60 |
| VI, bỏ health_fact, boolq không văn bản, EXAMS | 13.931 | 0,830 | 0,128 | — | — |

**Theo dataset VI:**
- A thấp nhất:
  - health_fact 0,52: chỉ có nhận định, không có văn bản. Audit đã gắn cờ.
  - boolq không văn bản 0,63, EXAMS 0,63, headline_cause 0,63, multidogo 0,65, quail 0,67, vitaminc 0,68.
- A cao nhất: VSMRC 0,97, ViSFC-sentiment 0,97, DREAM 0,93.

**Áp bảng đã chốt trước** (A_VI = 0,811 ≤ 0,90): **"Teacher mang thông tin khác nhãn đáng kể: 2a đáng train, nhưng phải soát mù trước."** Điều này **bác bỏ dự đoán ở mục 1** của chính điều phối ("chưng cất trên C chỉ tăng rất ít"). Dự đoán đó dựa trên dev, nơi Quyet trả lời gần như one-hot. Trên prompt train, Quyet kém chắc hơn nhiều và bất đồng với 19% nhãn VI.

**Soát mù 60 dòng VI teacher bất đồng tự tin (p > 0,9).**
- Người soát là agent Opus, đã che nhãn gốc và đáp án teacher. Script: `cham_soat_mu.py` trong scratchpad của điều phối.

| Nhóm (10 dòng mỗi nhóm) | Người soát mù trùng nhãn gốc | Trùng teacher | Khác cả hai |
|---|---|---|---|
| Đọc hiểu | **8** | 2 | 0 |
| Kiểm chứng/NLI | 1 | **9** | 0 |
| Chủ đề (VNTC, ViSFC) | 2 | **7** | 1 |
| Ý định | 3 | **6** | 1 |
| Nhân quả (e-CARE, HeadlineCause) | 5 | 5 | 0 |
| Còn lại | 3 | **7** | 0 |
| **Tổng** | **22** | **36** | 2 |

- Chỉ tính các câu người soát chắc chắn và không gắn cờ (28 câu): teacher 17, nhãn gốc 9, khác 2.
- Người soát gắn cờ 32/60 câu:
  - mơ hồ 25;
  - lỗi câu 6;
  - thiếu thông tin 3;
  - không có đáp án đúng 1.

  Các nguồn nhiễu: e-CARE (dịch sai, câu như tung đồng xu); VNTC (10 chủ đề chồng lấn); quy ước "mâu thuẫn hay trung lập" của FEVER.

**Đọc kết quả** (n = 60 là nhỏ, mỗi nhóm chỉ 10 dòng, nên đây là tín hiệu, chưa phải ước lượng chính xác):
1. **Ở kiểm chứng/NLI, chủ đề và ý định, khi teacher tự tin bất đồng thì nhãn gốc thường sai hoặc lệch quy ước.** Quy tắc cũ "teacher bất đồng thì dùng nhãn" sẽ **giữ nguyên** các nhãn sai này. Điều này khớp với lỗi P4 của bản C và với việc audit thấy khung kiểm chứng lệch.
2. **Ở đọc hiểu, teacher sai, nhãn gốc đúng.** Phải giữ nhãn.
3. **Nhiều dòng tự thân mơ hồ hoặc hỏng.** Bỏ chúng là lựa chọn an toàn hơn là gán lại nhãn.

**Đề xuất sửa thiết kế 2a** (chờ Fable phản biện và người dùng duyệt):
- Dòng **đồng ý**: đích = 0,6·teacher + 0,4·nhãn. Đây là lượng thông tin mới khoảng 0,6·S, tức khoảng 0,08 trên mỗi dòng VI.
- Dòng **bất đồng tự tin** (p > 0,9):
  - ở kiểm chứng/NLI, chủ đề, ý định, nhân quả, còn lại: **bỏ dòng**, khoảng 1,6% VI;
  - ở đọc hiểu: giữ nhãn.
- Dòng **bất đồng nhưng không tự tin**: giữ nhãn.
- **Không** dùng teacher để tạo nhãn mới cho dòng có nhãn, vì Quyet sai ở đọc hiểu và có thể đã nhiễm.
- Bỏ 3 nguồn không có văn bản (health_fact, boolq không văn bản, EXAMS). Việc này audit cũng đã đề xuất.
- Bỏ 3 nguồn ra khỏi 2a là thêm một biến thứ hai. Có thể gộp vào cùng biến "xây đích", hoặc tách ra; người dùng quyết.

**Kỳ vọng:** chưa có số công khai cho đúng bài này, và decider không thấy chưng cất tăng độ chính xác. Lợi thế riêng của ta là **sửa nhãn lệch quy ước ở kiểm chứng**, đúng nguồn đang yếu (ViFactCheck 72,2, P4). Ước lượng thô +0,5 đến +2 điểm VI, chủ yếu ở ViFactCheck và P4. Với độ nhạy ở mục 3, 2 seed mỗi bên khó tách được mức tăng dưới 2 điểm.

## 9. Còn lại

- Fable phản biện bản 2: `PHAN_BIEN_phan_tich_fable.md`.
- Người dùng duyệt thiết kế 2a, gồm C seed 2 và dev + proxy làm phép so phụ, trước khi train bất cứ gì.

## 10. Sau phản biện của Fable (bản 2.1, 28/10 khoảng 05:10)

Nguồn: `PHAN_BIEN_phan_tich_fable.md`. Điều phối đã tự kiểm lại các số then chốt và **chấp nhận phản biện**. Kết luận "2a đáng train" ở mục 8 **được rút lại**.

**Ba điểm đã tự kiểm:**
- **Lượng dòng bị bỏ theo quy tắc mục 8 là 146 dòng, tức 0,97% VI**, không phải 1,6%. Con số 1,6% là toàn bộ D90, gồm cả đọc hiểu.
- **Quyet nhận diện "không đủ thông tin" (NEI) kém trên prompt train, ở các bài cùng chủ đề:**

  | Nguồn | Recall NEI | Precision NEI |
  |---|---|---|
  | vitaminc | 0,29 | 0,30 |
  | nli_fever | 0,43 | 0,78 |
  | ViNLI relation | 0,52 | 0,81 |
  | (đối chiếu) dev | 0,82 | 0,73 |

  Quyet giỏi NEI kiểu "văn bản lạc đề" (P4 97,5%), nhưng kém NEI kiểu "cùng chủ đề mà thiếu dữ kiện". Bench theo quy ước NEI chặt, có 31% câu NEI. Tin teacher ở các dòng này sẽ làm model **bớt** chọn NEI, tức đi ngược hướng cần sửa. Người soát mù (Opus) nghiêng về teacher ở các câu loại này có thể vì áp quy ước "suy ra được" lỏng hơn quy ước của bộ dữ liệu.
- **Mẫu soát mù lệch về dữ liệu dịch:**
  - 41/60 dòng trong mẫu là dòng dịch.
  - Ở dòng dịch Quyet đồng ý với nhãn 0,729, ở dòng VI gốc 0,876.
  - Nếu chỉ tính câu người soát chắc "cao" và không gắn cờ thì có 16 câu: teacher đúng 11, nhãn đúng 5.

**Kết luận bản 2.1:**
1. **Không chạy 2a như một thí nghiệm riêng** (3 lần train). Tác động gồm 146 dòng bị bỏ và lượng làm mềm khoảng 0,08 trên mỗi dòng, quá nhỏ so với độ nhạy của cổng (mục 3). Riêng hướng "tin teacher ở kiểm chứng" là sai hướng.
2. **Bước tiếp nên là 2b** (dữ liệu đúng khung), gộp phần lọc rẻ vào C′:
   - bỏ 3 nguồn không có văn bản (health_fact, boolq không văn bản, EXAMS);
   - bỏ 146 dòng teacher bất đồng tự tin, như một bước làm sạch, kèm soát mẫu dòng dịch;
   - sửa chữ nhãn NEI theo dev;
   - thêm tăng cường "từ chối trả lời" kiểu decider;
   - thêm khung kiểm chứng của bench, có **NEI cùng chủ đề** (câu lấy từ phần khác của cùng bài, dựng bằng chương trình) và NEI lạc đề.
3. **Hoãn đích mềm từ teacher.** Nếu thử thì làm sau 2b, như một biến duy nhất, và đo bằng hiệu chỉnh.
4. **Thước đo chính của bước sau:**
   - P4;
   - recall NEI trên ViFactCheck/ViWikiFC, hiện ở C là 0,21 / 0,44;
   - VI macro.

   proxy_v1 (300 cặp kiểm chứng, 94 NEI) là phép đo phụ. Các thước đo này nhạy hơn VI macro với đúng thay đổi đang thử.
5. **"Giữ 1.7B" ghi là quyết định theo chi phí.** Điểm zero-shot của 2B-Base (dùng chat template mượn) là bằng chứng yếu về khả năng của nó sau khi train.
