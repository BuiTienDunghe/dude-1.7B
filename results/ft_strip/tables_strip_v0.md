dev_v1 SHA-256 khớp manifest: `894558631429840e…`. Nhiệt độ (fit trên val_v0): strip 1.535, control 1.560.

## 1. Năm lát chính — accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]

| Lát | n | strip (mới) | control (đối chứng) | decider-2b | Qwen3-1.7B gốc |
|---|---|---|---|---|---|
| song song · EN | 1000 | 86.8 [84.5; 89.0] | 86.4 [84.1; 88.6] | 89.2 [87.3; 91.1] | 83.4 [80.9; 85.8] |
| song song · VI có dấu | 1000 | 80.8 [78.0; 83.5] | 79.8 [76.9; 82.4] | 86.0 [83.5; 88.3] | 72.9 [69.8; 75.9] |
| song song · VI không dấu | 1000 | 60.8 [57.4; 64.2] | 58.8 [55.6; 62.3] | 49.2 [45.9; 52.6] | 48.0 [44.5; 51.4] |
| bản địa · có dấu | 763 | 66.3 [63.0; 69.7] | 66.2 [62.9; 69.4] | 66.1 [62.9; 69.3] | 53.9 [50.1; 57.5] |
| bản địa · không dấu | 763 | 56.9 [53.4; 60.4] | 53.9 [50.5; 57.2] | 42.9 [39.4; 46.6] | 47.6 [44.1; 51.2] |

## 2. Chênh lệch theo cặp (cùng câu): strip − model khác (điểm %, [CI 95%], McNemar p gộp)

| Lát | strip − control (đối chứng) | strip − decider-2b | strip − Qwen3-1.7B gốc |
|---|---|---|---|
| song song · EN | +0.4 [-0.7; +1.5] (p=0.75) | -2.3 [-4.4; -0.3] (p=0.0051) | +3.4 [+1.2; +5.6] (p=0.0087) |
| song song · VI có dấu | +1.1 [-1.1; +3.1] (p=0.24) | -5.2 [-8.1; -2.2] (p=4.8e-05) | +7.9 [+5.0; +10.6] (p=2.6e-12) |
| song song · VI không dấu | +2.0 [-1.5; +5.4] (p=0.13) | +11.7 [+7.6; +15.8] (p=4.4e-12) | +12.8 [+8.0; +17.1] (p=1.1e-15) |
| bản địa · có dấu | +0.1 [-2.1; +2.3] (p=0.91) | +0.2 [-3.8; +4.2] (p=0.33) | +12.4 [+8.5; +16.6] (p=8e-10) |
| bản địa · không dấu | +3.0 [-0.6; +6.4] (p=0.044) | +14.0 [+9.6; +18.6] (p=2.1e-11) | +9.3 [+5.1; +13.5] (p=3.9e-05) |

## 3. Brier, ECE, letter_mass, nhất quán khi đảo thứ tự

| Lát | Model | Brier T=1 | Brier sau T | ECE T=1 | ECE sau T | letter_mass TB / số < 0,5 | nhất quán |
|---|---|---|---|---|---|---|---|
| song song · EN | strip (mới) | 0.203 | 0.201 | 0.054 | 0.043 | 1.000 / 0 | 92.4 |
| song song · EN | control (đối chứng) | 0.213 | 0.208 | 0.063 | 0.045 | 1.000 / 0 | 93.0 |
| song song · EN | decider-2b | 0.164 | — | 0.022 | — | — | — |
| song song · EN | Qwen3-1.7B gốc | 0.310 | — | 0.156 | — | 0.982 / 14 | — |
| song song · VI có dấu | strip (mới) | 0.284 | 0.278 | 0.067 | 0.061 | 0.999 / 0 | 90.1 |
| song song · VI có dấu | control (đối chứng) | 0.304 | 0.290 | 0.093 | 0.036 | 0.999 / 0 | 89.4 |
| song song · VI có dấu | decider-2b | 0.211 | — | 0.023 | — | — | — |
| song song · VI có dấu | Qwen3-1.7B gốc | 0.474 | — | 0.235 | — | 0.988 / 4 | — |
| song song · VI không dấu | strip (mới) | 0.540 | 0.524 | 0.108 | 0.041 | 0.999 / 0 | 78.6 |
| song song · VI không dấu | control (đối chứng) | 0.568 | 0.538 | 0.147 | 0.053 | 0.999 / 0 | 77.1 |
| song song · VI không dấu | decider-2b | 0.626 | — | 0.044 | — | — | — |
| song song · VI không dấu | Qwen3-1.7B gốc | 0.942 | — | 0.452 | — | 0.981 / 5 | — |
| bản địa · có dấu | strip (mới) | 0.456 | 0.429 | 0.125 | 0.054 | 1.000 / 0 | 96.3 |
| bản địa · có dấu | control (đối chứng) | 0.500 | 0.455 | 0.180 | 0.095 | 1.000 / 0 | 95.8 |
| bản địa · có dấu | decider-2b | 0.436 | — | 0.126 | — | — | — |
| bản địa · có dấu | Qwen3-1.7B gốc | 0.880 | — | 0.429 | — | 0.999 / 0 | — |
| bản địa · không dấu | strip (mới) | 0.592 | 0.550 | 0.155 | 0.098 | 0.999 / 0 | 95.3 |
| bản địa · không dấu | control (đối chứng) | 0.629 | 0.570 | 0.230 | 0.149 | 0.999 / 0 | 91.2 |
| bản địa · không dấu | decider-2b | 0.611 | — | 0.135 | — | — | — |
| bản địa · không dấu | Qwen3-1.7B gốc | 0.929 | — | 0.440 | — | 0.999 / 0 | — |

## 4. Theo nguồn — accuracy (%) và strip − control (McNemar p)

| Lát · nguồn | strip (mới) | control (đối chứng) | decider-2b | Qwen3-1.7B gốc | strip − control (p) |
|---|---|---|---|---|---|
| song song · EN · belebele | 84.3 | 84.7 | 87.3 | 79.0 | -0.3 (p=1) |
| song song · EN · massive | 88.0 | 86.7 | 94.0 | 88.0 | +1.3 (p=0.22) |
| song song · EN · sib200 | 83.0 | 83.3 | 83.3 | 80.7 | -0.3 (p=1) |
| song song · EN · xcopa | 92.0 | 91.0 | 92.0 | 86.0 | +1.0 (p=1) |
| song song · VI có dấu · belebele | 78.3 | 77.3 | 83.0 | 67.3 | +1.0 (p=0.7) |
| song song · VI có dấu · massive | 84.3 | 81.0 | 92.7 | 71.3 | +3.3 (p=0.064) |
| song song · VI có dấu · sib200 | 80.7 | 81.7 | 82.3 | 74.0 | -1.0 (p=0.55) |
| song song · VI có dấu · xcopa | 80.0 | 79.0 | 86.0 | 79.0 | +1.0 (p=1) |
| song song · VI không dấu · belebele | 60.7 | 58.3 | 49.0 | 46.0 | +2.3 (p=0.41) |
| song song · VI không dấu · massive | 66.7 | 62.7 | 58.7 | 43.7 | +4.0 (p=0.13) |
| song song · VI không dấu · sib200 | 59.0 | 58.3 | 36.0 | 51.3 | +0.7 (p=0.89) |
| song song · VI không dấu · xcopa | 57.0 | 56.0 | 53.0 | 51.0 | +1.0 (p=1) |
| bản địa · có dấu · vifactcheck | 70.6 | 71.2 | 81.0 | 54.6 | -0.6 (p=1) |
| bản địa · có dấu · vihsd | 53.0 | 50.7 | 40.7 | 46.7 | +2.3 (p=0.32) |
| bản địa · có dấu · viwikifc | 75.3 | 76.7 | 76.7 | 60.3 | -1.3 (p=0.52) |
| bản địa · không dấu · vifactcheck | 61.3 | 62.6 | 52.8 | 51.5 | -1.2 (p=0.86) |
| bản địa · không dấu · vihsd | 46.0 | 34.3 | 34.0 | 43.7 | +11.7 (p=0.00064) |
| bản địa · không dấu · viwikifc | 63.3 | 64.7 | 42.0 | 47.7 | -1.3 (p=0.7) |

## 5. Theo kiểu câu hỏi (khối bản địa) — accuracy (%, trung bình đều theo nguồn), strip − control và strip − decider; McNemar p gộp theo dòng

| Lát | n | strip (mới) | control (đối chứng) | decider-2b | Qwen3-1.7B gốc | strip − control [CI] (p) | strip − decider [CI] (p) |
|---|---|---|---|---|---|---|---|
| bản địa · choice · có dấu | 256 | 62.7 | 63.3 | 72.7 | 39.9 | -0.6 [-4.3; +3.5] (0.68) | -10.0 [-17.0; -3.1] (0.034) |
| bản địa · choice · không dấu | 256 | 51.0 | 49.0 | 33.7 | 32.0 | +2.1 [-4.4; +8.7] (0.61) | +17.3 [+8.4; +25.8] (2.2e-07) |
| bản địa · noul · có dấu | 363 | 77.5 | 76.8 | 70.4 | 71.8 | +0.6 [-2.7; +4.0] (0.51) | +7.1 [+1.9; +11.9] (0.00082) |
| bản địa · noul · không dấu | 363 | 70.0 | 66.3 | 54.9 | 65.0 | +3.7 [-1.1; +8.9] (0.035) | +15.1 [+8.6; +21.4] (5.3e-06) |
| bản địa · score · có dấu | 144 | 42.4 | 42.4 | 42.4 | 33.3 | +0.0 [-5.6; +5.6] (1) | +0.0 [-10.4; +10.4] (1) |
| bản địa · score · không dấu | 144 | 31.9 | 31.2 | 31.2 | 32.6 | +0.7 [-5.6; +6.9] (1) | +0.7 [-4.9; +6.2] (1) |

## 6. Chẩn đoán phụ — accuracy tiếng Việt (%) theo phần bị bỏ dấu (không phải số chính thức)

| Khối | Model | có dấu | chỉ state không dấu | không dấu toàn bộ |
|---|---|---|---|---|
| song song | strip (mới) | 80.8 | 63.8 | 60.8 |
| song song | control (đối chứng) | 79.8 | 62.3 | 58.8 |
| song song | decider-2b | 86.0 | 55.6 | 49.2 |
| bản địa | strip (mới) | 66.3 | 59.7 | 56.9 |
| bản địa | control (đối chứng) | 66.2 | 55.8 | 53.9 |
| bản địa | decider-2b | 66.1 | 53.6 | 42.9 |

## 7. Ngưỡng đã định trước

- Không dấu, song song: strip − control +2.0 [-1.5; +5.4] (cần ≥ +5,0 và CI không chứa 0) → **KHÔNG ĐẠT**
- Không dấu, bản địa: +3.0 [-0.6; +6.4] → **KHÔNG ĐẠT**
- VI có dấu, song song: +1.1 [-1.1; +3.1] (cần giảm không quá 2,0) → **ĐẠT** (dưới 1,5 điểm: coi là nhiễu)
- VI có dấu, bản địa: +0.1 [-2.1; +2.3] (cần giảm không quá 2,0) → **ĐẠT** (dưới 1,5 điểm: coi là nhiễu)
- EN: +0.4 [-0.7; +1.5] (cần giảm không quá 2,0) → **ĐẠT** (dưới 1,5 điểm: coi là nhiễu)
- Tổng: **KHÔNG ĐẠT**
