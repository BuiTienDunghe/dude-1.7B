# Bảng số bước E (DEV 1.2)

Dev: `data/bench_v1_2/dev_v1_2.jsonl` (sha `ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b`). Accuracy (%), trung bình đều theo nguồn, [CI 95% bootstrap theo state].

## VI có dấu

| Model | Trung bình đều [CI] | Chỉ dòng source | Bỏ MASSIVE | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC | Brier | Câu bị loại |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 70,6 [68,1; 73,2] | 70,5 | 69,7 | 75,2 | 75,6 | 75,2 | 80,0 | 57,6 | 59,9 | 0,543 | 0 |
| Qwen3-1.7B full_vocab (dự án) | 79,7 [77,4; 82,0] | 79,7 | 78,3 | 82,3 | 82,6 | 86,7 | 78,0 | 72,8 | 76,0 | 0,294 | 0 |
| Qwen3.5-2B gốc | 70,6 [68,1; 73,1] | 70,9 | 71,9 | 77,9 | 76,3 | 63,9 | 73,0 | 70,3 | 62,0 | 0,411 | 0 |
| decider-2b (text) | 86,3 [84,4; 88,2] | 86,3 | 84,4 | 87,8 | 82,3 | 96,3 | 86,0 | 83,5 | 82,2 | 0,209 | 0 |
| decider-2b (keyed) | 84,9 [82,9; 86,9] | 84,7 | 83,5 | 88,4 | 81,3 | 92,2 | 83,0 | 82,9 | 81,9 | 0,217 | 0 |
| Quyet-1.0-Medium (text) | 89,4 [87,8; 91,1] | 88,8 | 88,0 | 93,9 | 83,6 | 96,6 | 93,0 | 88,6 | 80,8 | 0,166 | 0 |

## VI không dấu

| Model | Trung bình đều [CI] | Chỉ dòng source | Bỏ MASSIVE | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC | Brier | Câu bị loại |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 48,5 [45,6; 51,3] | 48,7 | 49,0 | 49,3 | 51,2 | 45,6 | 52,0 | 44,9 | 47,7 | 0,935 | 0 |
| Qwen3-1.7B full_vocab (dự án) | 62,3 [59,6; 65,0] | 63,5 | 61,5 | 58,8 | 60,2 | 66,3 | 57,0 | 65,2 | 66,2 | 0,520 | 0 |
| Qwen3.5-2B gốc | 39,0 [36,4; 41,8] | 39,0 | 43,4 | 43,5 | 29,8 | 17,3 | 48,0 | 50,0 | 45,6 | 0,789 | 0 |
| decider-2b (text) | 49,6 [46,9; 52,4] | 50,2 | 48,8 | 53,1 | 35,8 | 53,7 | 54,0 | 53,2 | 48,1 | 0,610 | 0 |
| decider-2b (keyed) | 48,5 [45,7; 51,3] | 49,5 | 48,0 | 52,7 | 36,5 | 51,4 | 52,0 | 51,3 | 47,4 | 0,619 | 0 |
| Quyet-1.0-Medium (text) | 64,5 [61,6; 67,2] | 65,5 | 61,7 | 65,0 | 63,9 | 78,6 | 62,0 | 65,2 | 52,6 | 0,497 | 0 |

## EN

| Model | Trung bình đều [CI] | Chỉ dòng source | Bỏ MASSIVE | Belebele | SIB-200 | MASSIVE | XCOPA | Brier | Câu bị loại |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 84,6 [82,2; 86,9] | 85,7 | 82,6 | 80,6 | 81,3 | 90,5 | 86,0 | 0,288 | 0 |
| Qwen3-1.7B full_vocab (dự án) | 88,0 [85,7; 90,0] | 89,0 | 87,2 | 86,7 | 83,9 | 90,1 | 91,0 | 0,187 | 0 |
| Qwen3.5-2B gốc | 84,5 [81,8; 86,9] | 85,2 | 84,2 | 84,0 | 84,6 | 85,4 | 84,0 | 0,239 | 0 |
| decider-2b (text) | 90,3 [88,4; 92,3] | 90,1 | 88,2 | 88,8 | 83,9 | 96,6 | 92,0 | 0,148 | 0 |
| decider-2b (keyed) | 89,8 [87,9; 91,8] | 89,5 | 88,2 | 88,1 | 83,6 | 94,6 | 93,0 | 0,159 | 0 |
| Quyet-1.0-Medium (text) | 94,5 [93,1; 95,7] | 93,2 | 93,6 | 97,3 | 83,6 | 96,9 | 100,0 | 0,090 | 0 |

## So với Qwen3-1.7B full_vocab (dự án) (A − dự án, điểm %; [CI]; p McNemar gộp)

| Model | VI có dấu | VI không dấu | EN | VI bỏ MASSIVE | EN bỏ MASSIVE |
|---|---|---|---|---|---|
| Qwen3-1.7B gốc | -9,2 [-11,5; -6,7] (p=2e-18) | -13,8 [-17,4; -10,4] (p=2e-21) | -3,4 [-5,7; -1,3] (p=0,004) | -8,7 [-11,3; -6,1] | -4,6 [-7,3; -1,9] |
| Qwen3.5-2B gốc | -9,2 [-12,2; -6,3] (p=3e-14) | -23,2 [-26,9; -19,7] (p=2e-46) | -3,5 [-6,3; -0,8] (p=0,032) | -6,4 [-9,9; -3,0] | -3,0 [-6,7; +0,5] |
| decider-2b (text) | +6,6 [+4,4; +8,9] (p=5e-09) | -12,7 [-16,1; -9,2] (p=4e-18) | +2,4 [+0,2; +4,6] (p=0,017) | +6,0 [+3,3; +8,6] | +1,0 [-1,7; +3,8] |
| decider-2b (keyed) | +5,2 [+2,9; +7,6] (p=9e-06) | -13,8 [-17,2; -10,4] (p=7e-20) | +1,9 [-0,4; +4,1] (p=0,108) | +5,2 [+2,3; +7,8] | +1,0 [-1,6; +3,7] |
| Quyet-1.0-Medium (text) | +9,7 [+7,3; +12,2] (p=1e-15) | +2,2 [-1,2; +5,5] (p=0,160) | +6,5 [+4,4; +8,6] (p=1e-08) | +9,7 [+7,0; +12,6] | +6,4 [+3,9; +9,2] |

### Theo nguồn, VI có dấu (A − dự án; p McNemar chính xác)

| Model | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC |
|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | -7,1 (p=0,003) | -7,0 (p=0,005) | -11,6 (p=5e-06) | +2,0 (p=0,791) | -15,2 (p=7e-05) | -16,0 (p=6e-08) |
| Qwen3.5-2B gốc | -4,4 (p=0,124) | -6,4 (p=0,005) | -22,8 (p=2e-12) | -5,0 (p=0,487) | -2,5 (p=0,683) | -13,9 (p=5e-05) |
| decider-2b (text) | +5,4 (p=0,036) | -0,3 (p=1,000) | +9,5 (p=6e-08) | +8,0 (p=0,096) | +10,8 (p=0,006) | +6,3 (p=0,011) |
| decider-2b (keyed) | +6,1 (p=0,018) | -1,3 (p=0,627) | +5,4 (p=0,014) | +5,0 (p=0,359) | +10,1 (p=0,011) | +5,9 (p=0,021) |
| Quyet-1.0-Medium (text) | +11,6 (p=6e-07) | +1,0 (p=0,701) | +9,9 (p=1e-06) | +15,0 (p=0,001) | +15,8 (p=4e-05) | +4,9 (p=0,098) |

### Theo nguồn, EN (A − dự án; p McNemar chính xác)

| Model | Belebele | SIB-200 | MASSIVE | XCOPA |
|---|---|---|---|---|
| Qwen3-1.7B gốc | -6,1 (p=0,005) | -2,7 (p=0,200) | +0,3 (p=1,000) | -5,0 (p=0,180) |
| Qwen3.5-2B gốc | -2,7 (p=0,332) | +0,7 (p=0,868) | -4,8 (p=0,038) | -7,0 (p=0,167) |
| decider-2b (text) | +2,0 (p=0,441) | +0,0 (p=1,000) | +6,5 (p=7e-05) | +1,0 (p=1,000) |
| decider-2b (keyed) | +1,4 (p=0,627) | -0,3 (p=1,000) | +4,4 (p=0,024) | +2,0 (p=0,754) |
| Quyet-1.0-Medium (text) | +10,5 (p=2e-06) | -0,3 (p=1,000) | +6,8 (p=2e-04) | +9,0 (p=0,004) |

### Theo nguồn, VI không dấu (A − dự án; p McNemar chính xác)

| Model | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC |
|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | -9,5 (p=0,004) | -9,0 (p=0,002) | -20,7 (p=2e-09) | -5,0 (p=0,542) | -20,3 (p=2e-05) | -18,5 (p=6e-07) |
| Qwen3.5-2B gốc | -15,3 (p=7e-05) | -30,4 (p=6e-15) | -49,0 (p=5e-32) | -9,0 (p=0,272) | -15,2 (p=0,006) | -20,6 (p=9e-07) |
| decider-2b (text) | -5,8 (p=0,125) | -24,4 (p=7e-11) | -12,6 (p=1e-04) | -3,0 (p=0,755) | -12,0 (p=0,011) | -18,1 (p=4e-06) |
| decider-2b (keyed) | -6,1 (p=0,111) | -23,7 (p=2e-10) | -15,0 (p=2e-05) | -5,0 (p=0,522) | -13,9 (p=0,003) | -18,8 (p=1e-06) |
| Quyet-1.0-Medium (text) | +6,1 (p=0,085) | +3,7 (p=0,289) | +12,2 (p=5e-06) | +5,0 (p=0,542) | +0,0 (p=1,000) | -13,6 (p=4e-04) |

### Brier theo nguồn, VI có dấu (thấp hơn là tốt hơn) [CI]

| Model | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC | Trung bình đều |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 0,434 | 0,455 | 0,442 | 0,331 | 0,832 | 0,765 | 0,543 [0,497; 0,587] |
| Qwen3-1.7B full_vocab (dự án) | 0,293 | 0,264 | 0,195 | 0,308 | 0,359 | 0,347 | 0,294 [0,266; 0,326] |
| Qwen3.5-2B gốc | 0,323 | 0,344 | 0,512 | 0,364 | 0,421 | 0,502 | 0,411 [0,383; 0,442] |
| decider-2b (text) | 0,189 | 0,260 | 0,062 | 0,207 | 0,243 | 0,290 | 0,209 [0,188; 0,232] |
| decider-2b (keyed) | 0,184 | 0,256 | 0,117 | 0,214 | 0,244 | 0,290 | 0,217 [0,196; 0,241] |
| Quyet-1.0-Medium (text) | 0,094 | 0,252 | 0,046 | 0,109 | 0,210 | 0,286 | 0,166 [0,144; 0,189] |

### Brier theo nguồn, VI không dấu (thấp hơn là tốt hơn) [CI]

| Model | Belebele | SIB-200 | MASSIVE | XCOPA | ViFactCheck | ViWikiFC | Trung bình đều |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 0,892 | 0,906 | 0,982 | 0,889 | 0,976 | 0,965 | 0,935 [0,883; 0,985] |
| Qwen3-1.7B full_vocab (dự án) | 0,564 | 0,568 | 0,468 | 0,528 | 0,490 | 0,499 | 0,520 [0,489; 0,552] |
| Qwen3.5-2B gốc | 0,712 | 0,889 | 1,088 | 0,673 | 0,641 | 0,731 | 0,789 [0,759; 0,821] |
| decider-2b (text) | 0,595 | 0,751 | 0,606 | 0,508 | 0,600 | 0,603 | 0,610 [0,590; 0,631] |
| decider-2b (keyed) | 0,591 | 0,762 | 0,651 | 0,507 | 0,599 | 0,604 | 0,619 [0,599; 0,638] |
| Quyet-1.0-Medium (text) | 0,453 | 0,485 | 0,303 | 0,488 | 0,597 | 0,657 | 0,497 [0,466; 0,527] |

### Brier theo nguồn, EN (thấp hơn là tốt hơn) [CI]

| Model | Belebele | SIB-200 | MASSIVE | XCOPA | Trung bình đều |
|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 0,364 | 0,354 | 0,185 | 0,249 | 0,288 [0,243; 0,336] |
| Qwen3-1.7B full_vocab (dự án) | 0,222 | 0,262 | 0,133 | 0,130 | 0,187 [0,158; 0,219] |
| Qwen3.5-2B gốc | 0,230 | 0,268 | 0,226 | 0,233 | 0,239 [0,209; 0,271] |
| decider-2b (text) | 0,166 | 0,238 | 0,057 | 0,129 | 0,148 [0,126; 0,171] |
| decider-2b (keyed) | 0,167 | 0,233 | 0,100 | 0,134 | 0,159 [0,137; 0,183] |
| Quyet-1.0-Medium (text) | 0,054 | 0,241 | 0,044 | 0,023 | 0,090 [0,075; 0,107] |

## Tụt Anh → Việt, nhất quán khi đảo thứ tự, độ trễ

Nhất quán khi đảo tính trên câu KHÔNG phải Có/Không (noul) cho MỌI model, kể cả mốc dự án, để cùng một tập câu. Độ trễ: trung vị batch 1 trên toàn bộ 3.851 dòng dev, riêng decider-2b đo ở lượt đảo trên 3.445 dòng không phải noul (khác tập dòng, cùng cách đo).

| Model | Tham số | EN − VI có dấu [CI] | Nhất quán khi đảo VI / VI không dấu / EN | Accuracy trung bình hai thứ tự VI | Độ trễ trung vị batch 1 (ms) |
|---|---|---|---|---|---|
| Qwen3-1.7B gốc | 2,03 B | +8,1 [+4,9; +11,4] | 70,0 / 47,2 / 86,3 | 63,2 | 32,3 |
| Qwen3-1.7B full_vocab (dự án) | 2,03 B + LoRA 17,4 M (đã merge) | +5,5 [+2,9; +8,1] | 90,4 / 82,0 / 93,7 | 76,6 | 32,9 |
| Qwen3.5-2B gốc | 2,27 B trong repo (gồm bộ mã hóa ảnh); chỉ nạp phần văn bản | +11,7 [+8,4; +15,1] | 71,3 / 39,1 / 85,1 | 66,4 | 57,7 |
| decider-2b (text) | 1,88 B | +2,3 [+0,0; +4,5] | 93,9 / 81,9 / 96,6 | 84,0 | 52,3 |
| decider-2b (keyed) | 1,88 B | +3,6 [+1,2; +6,1] | 92,7 / 75,4 / 95,4 | 82,7 | 54,3 |
| Quyet-1.0-Medium (text) | 4,66 B (Qwen3.5-4B + LoRA đã merge) | +2,7 [+1,0; +4,5] | 97,9 / 81,7 / 99,2 | 89,4 | 87,5 |

Tái lập (lượt chạy lại khi GPU rảnh so với lượt đầu): Qwen3.5-2B gốc normal: cùng dự đoán 100,0%, xác suất lệch tối đa 0.00e+00; Qwen3-1.7B full_vocab (dự án) reverse: cùng dự đoán 100,0%, xác suất lệch tối đa 0.00e+00; decider-2b (text) reverse: cùng dự đoán 100,0%, xác suất lệch tối đa 0.00e+00; decider-2b (keyed) reverse: cùng dự đoán 100,0%, xác suất lệch tối đa 0.00e+00

## Kiểm adapter model ngoài

| Model | README | Câu hiển nhiên sai, text / keyed (trên 40 câu × 2 thứ tự) | Khớp predict() công khai (4 chữ số) | Câu vượt giới hạn mặc định | Câu bị cắt / từ chối ở giới hạn đã nâng | Vào bảng chính |
|---|---|---|---|---|---|---|
| Quyet-1.0-Tiny | đạt | 6 / 2 | có (lệch tối đa 5.0e-05) | 0 | 0 | không (chưa qua kiểm) |
| Quyet-1.0-Small | đạt | 6 / 2 | có (lệch tối đa 5.0e-05) | 0 | 0 | không (chưa qua kiểm) |
| laya-multilingual | KHÔNG đạt | 12 / 12 | có (lệch tối đa 5.0e-05) | 101 | 0 | không (chưa qua kiểm) |
| laya-vi | đạt | 2 / 3 | có (lệch tối đa 5.0e-05) | 101 | 0 | không (chưa qua kiểm) |
| Quyet-1.0-Medium | đạt | 0 / 0 | có (lệch tối đa 4.9e-05) | 0 | 0 | có |

## Bảng phụ: CHƯA QUA KIỂM CÂU HIỂN NHIÊN – chỉ tham khảo

Bốn model dưới đây trượt câu hiển nhiên; adapter khớp predict() công khai đến 4 chữ số nên số phản ánh model, nhưng KHÔNG thuộc bảng chính. Dùng cách ánh xạ cho VI có dấu cao hơn (quy tắc brief).

| Model | Ánh xạ dùng | VI có dấu [CI] | VI không dấu | EN [CI] | Brier VI / EN | Nhất quán khi đảo VI / VI không dấu / EN | Độ trễ (ms) | Câu hiển nhiên sai |
|---|---|---|---|---|---|---|---|---|
| Quyet-1.0-Tiny | keyed | 65,4 [62,8; 68,1] | 42,6 | 73,3 [70,2; 76,3] | 0,475 / 0,365 | 85,2 / 71,1 / 87,8 | 13,5 | keyed sai câu Có/Không số 7 (VI, EN; hai thứ tự); text sai thêm câu chọn đáp án 2 (VI), 5 (EN), 6 (VI, EN) |
| Quyet-1.0-Small | keyed | 66,6 [63,9; 69,3] | 44,5 | 76,7 [73,7; 79,7] | 0,459 / 0,323 | 81,6 / 69,2 / 90,5 | 15,7 | keyed sai câu Có/Không số 7 (VI, EN; hai thứ tự); text sai thêm câu chọn đáp án 1 (VI, EN), 2 (EN), 5 (EN) |
| laya-multilingual | keyed | 52,3 [49,8; 54,9] | 33,8 | 63,6 [60,4; 66,9] | 0,644 / 0,487 | 69,8 / 40,2 / 74,6 | 16,5 | sai mọi câu Có/Không có đáp án 'Có' (0, 2, 4, 6, 8; VI và EN) và câu 7 EN; câu chọn đáp án 5 (VI); ví dụ README trượt cả ở fp32 (chọn 'sales' thay vì 'billing'); gói laya tự ghi revision 'đã duyệt' e4e9ddf2…, khác pin 1720e3e3… của dự án |
| laya-vi | text | 55,5 [52,8; 58,2] | 40,4 | 62,5 [59,3; 65,7] | 0,584 / 0,488 | 82,4 / 65,7 / 81,4 | 16,2 | sai câu Có/Không số 4 (VI) và số 6 (EN), hai thứ tự; keyed sai thêm câu chọn đáp án 5 (VI, một thứ tự) |

Quyet-1.0-Medium: hai cách ánh xạ cho cùng prompt (gói ẩn khóa khi có mô tả; đã kiểm xác suất trùng trên câu hiển nhiên), nên chỉ chấm và ghi một lần.

