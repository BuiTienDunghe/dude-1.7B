# xcopa:22: chấm thử bản dịch của người dùng (DEV, một câu; đáp án B)

Ô = 1.1 → bản thử 1.2 → bản người dùng. Mỗi mốc: ✓/✗, chữ cái dự đoán, P(đáp án đúng). EN giữ nguyên bản 1.1.

| Model | EN | VI có dấu | VI không dấu |
|---|---|---|---|
| decider-2b-text | ✓B 0.58 → ✓B 0.58 → ✓B 0.58 | ✓B 0.51 → ✗A 0.42 → ✓B 0.63 | ✓B 0.58 → ✓B 0.55 → ✓B 0.51 |
| decider-2b-keyed | ✓B 0.57 → ✓B 0.57 → ✓B 0.57 | ✗A 0.45 → ✗A 0.36 → ✓B 0.51 | ✓B 0.57 → ✓B 0.57 → ✓B 0.51 |
| decider-ft-ep2-text | ✓B 0.61 → ✓B 0.61 → ✓B 0.61 | ✗A 0.42 → ✗A 0.26 → ✓B 0.53 | ✓B 0.72 → ✓B 0.63 → ✗A 0.45 |
| decider-ft-ep2-keyed | ✓B 0.68 → ✓B 0.68 → ✓B 0.68 | ✗A 0.37 → ✗A 0.18 → ✗A 0.28 | ✓B 0.55 → ✓B 0.58 → ✗A 0.50 |
| Qwen3-1.7B-fullvocab-s1 | ✗A 0.47 → ✗A 0.47 → ✗A 0.47 | ✗A 0.08 → ✗A 0.05 → ✗A 0.12 | ✓B 0.53 → ✓B 0.92 → ✓B 0.92 |
| Qwen3-1.7B | ✗A 0.00 → ✗A 0.00 → ✗A 0.00 | ✗A 0.07 → ✗A 0.01 → ✓B 0.78 | ✗A 0.01 → ✗A 0.10 → ✓B 0.82 |
| strip-s1 | ✗A 0.15 → ✗A 0.15 → ✗A 0.15 | ✗A 0.06 → ✗A 0.03 → ✗A 0.11 | ✓B 0.73 → ✓B 0.76 → ✓B 0.85 |

| Biến thể | 1.1 | thử 1.2 | người dùng |
|---|---|---|---|
| EN | 4/7 | 4/7 | 4/7 |
| VI có dấu | 1/7 | 0/7 | 4/7 |
| VI không dấu | 6/7 | 6/7 | 5/7 |
