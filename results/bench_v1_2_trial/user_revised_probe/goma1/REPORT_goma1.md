# Goma#1: chấm thử bản dịch của người dùng (DEV, một câu; đáp án D)

Ô = 1.1 → bản thử 1.2 → bản người dùng. Mỗi mốc: ✓/✗, chữ cái dự đoán, P(đáp án đúng). EN giữ nguyên bản 1.1.

| Model | EN | VI có dấu | VI không dấu |
|---|---|---|---|
| decider-2b-text | ✓D 0.99 → ✓D 0.99 → ✓D 0.99 | ✓D 0.49 → ✗A 0.19 → ✓D 0.95 | ✗A 0.15 → ✗B 0.11 → ✓D 0.38 |
| decider-2b-keyed | ✓D 0.98 → ✓D 0.98 → ✓D 0.98 | ✗A 0.32 → ✗A 0.14 → ✓D 0.94 | ✗A 0.13 → ✗B 0.11 → ✗B 0.24 |
| decider-ft-ep2-text | ✓D 1.00 → ✓D 1.00 → ✓D 1.00 | ✗A 0.14 → ✗A 0.03 → ✓D 0.98 | ✗A 0.04 → ✗A 0.01 → ✗A 0.23 |
| decider-ft-ep2-keyed | ✓D 1.00 → ✓D 1.00 → ✓D 1.00 | ✗A 0.08 → ✗A 0.02 → ✓D 0.98 | ✗A 0.03 → ✗A 0.01 → ✗A 0.10 |
| Qwen3-1.7B-fullvocab-s1 | ✓D 0.98 → ✓D 0.98 → ✓D 0.98 | ✗A 0.03 → ✗A 0.03 → ✓D 0.95 | ✗A 0.20 → ✗A 0.17 → ✓D 0.78 |
| Qwen3-1.7B | ✓D 1.00 → ✓D 1.00 → ✓D 1.00 | ✗A 0.00 → ✗A 0.00 → ✗A 0.18 | ✗A 0.00 → ✗A 0.00 → ✓D 0.68 |
| strip-s1 | ✓D 0.98 → ✓D 0.98 → ✓D 0.98 | ✗A 0.14 → ✗A 0.10 → ✓D 0.95 | ✗A 0.10 → ✗A 0.12 → ✓D 0.67 |

| Biến thể | 1.1 | thử 1.2 | người dùng |
|---|---|---|---|
| EN | 7/7 | 7/7 | 7/7 |
| VI có dấu | 1/7 | 0/7 | 6/7 |
| VI không dấu | 0/7 | 0/7 | 4/7 |
