# G0: cách đọc kết quả, chốt TRƯỚC khi chấm (28/10/2026)

File này ghi trước khi chạy bất kỳ model nào cho G0. Ngưỡng lấy nguyên từ `results/BRIEF_buoc0_G0.md` (mục "Cách đọc"). Ở đây chỉ chốt thêm những chỗ brief chưa nói rõ. **Không đổi sau khi thấy số.**

## Điểm dùng để so

- **"Điểm VI có dấu"** = trung bình đều 6 nguồn (macro) trên dòng VI có dấu của dev 1.3. Đây là quy ước của dự án từ `BRIEF_train_chinh_thuc.md`.
- Điểm trên mọi câu chỉ báo kèm, không dùng để quyết.
- **CI theo cặp** = bootstrap 95% theo cụm (trường `cluster` của dev), rút lại riêng từng nguồn để giữ cơ cấu macro, 1.000 lần, seed 20261012. Đây là đúng cách của `scripts/train_v1/trend_C.py`.

## Nền 2B

**"Đáng cân nhắc đổi nền"** nếu cả hai điều sau đều đúng:
1. Qwen3.5-2B-Base − Qwen3-1.7B gốc ≥ **3,0 điểm** macro VI có dấu, **và** CI theo cặp không chứa 0;
2. p90 của 2B-Base ở đường CUDA graphs ≤ p90 của C-s1 ở đường CUDA graphs + **5 ms**.

Cách tính p90 cho điều kiện 2:
- trên toàn bộ dev 1.3, batch 1;
- lấy trung bình hai lần đo (lần 3 nếu hai lần lệch > 10%);
- 2B-Base lấy cấu hình nhanh hơn trong hai cấu hình có FLA và không FLA, với điều kiện cấu hình đó qua cổng đúng.

Thiếu một trong hai điều kiện thì **giữ 1.7B**. Quyết định cuối do người dùng.

## Điểm của Quyet đến từ đâu

Đọc theo macro VI có dấu của Qwen3.5-4B gốc (revision `851bf6e806ef`), chấm bằng khung dự án:
- **≥ 85** → lợi thế của Quyet chủ yếu do cỡ model;
- **≤ 75** → chủ yếu do công thức fine-tune;
- ở giữa → trộn lẫn.

## Khung prompt

- **Ghi "khung VI có thể làm mất điểm"** nếu khung EN hơn khung VI ≥ **2,0 điểm** macro trên dòng VI có dấu, CI theo cặp không chứa 0, ở **ít nhất 2 trong 3 model** (Qwen3-1.7B gốc, Qwen3.5-2B gốc, Qwen3.5-2B-Base).
- Ngoài trường hợp đó → giữ khung VI.
- Hai khung của cùng một model chấm trong **cùng một môi trường** (Qwen3-1.7B: `.venv`; hai model Qwen3.5: `.venv-g0`).

## Độ trễ

- Chỉ báo số: trung vị, p90, p95, p99, và p90 theo nhóm độ dài prompt (tính bằng tokenizer của chính model).
- Không kết luận về ngưỡng 40 ms.
- Số của đường CUDA graphs chỉ dùng nếu qua cổng đúng: argmax trùng đường eager 100% trên toàn dev, và xác suất lệch ≤ 1e-3. Không đạt thì báo độ lệch lớn nhất và **không dùng số đo**; không nới cổng.
