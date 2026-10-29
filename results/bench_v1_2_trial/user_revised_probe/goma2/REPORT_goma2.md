# Goma#2: chấm thử bản dịch của người dùng (DEV, một câu; đáp án A)

Ô = 1.1 → bản thử 1.2 → bản người dùng. Mỗi mốc: ✓/✗, chữ cái dự đoán, P(đáp án đúng). EN giữ nguyên bản 1.1.

| Model | EN | VI có dấu | VI không dấu |
|---|---|---|---|
| decider-2b-text | ✓A 0.92 → ✓A 0.92 → ✓A 0.92 | ✗C 0.30 → ✗C 0.18 → ✓A 0.87 | ✓A 0.51 → ✓A 0.53 → ✓A 0.45 |
| decider-2b-keyed | ✓A 0.92 → ✓A 0.92 → ✓A 0.92 | ✗C 0.19 → ✗C 0.13 → ✓A 0.92 | ✓A 0.42 → ✓A 0.41 → ✓A 0.43 |
| decider-ft-ep2-text | ✓A 0.99 → ✓A 0.99 → ✓A 0.99 | ✗C 0.03 → ✗C 0.04 → ✓A 0.98 | ✗D 0.11 → ✗C 0.25 → ✓A 0.48 |
| decider-ft-ep2-keyed | ✓A 0.99 → ✓A 0.99 → ✓A 0.99 | ✗C 0.03 → ✗C 0.03 → ✓A 0.99 | ✗D 0.09 → ✗C 0.24 → ✓A 0.65 |
| Qwen3-1.7B-fullvocab-s1 | ✗C 0.18 → ✗C 0.18 → ✗C 0.18 | ✓A 0.53 → ✗C 0.39 → ✓A 0.98 | ✓A 0.74 → ✓A 0.84 → ✓A 0.97 |
| Qwen3-1.7B | ✓A 0.97 → ✓A 0.97 → ✓A 0.97 | ✓A 0.99 → ✓A 1.00 → ✓A 0.99 | ✓A 0.96 → ✓A 0.96 → ✓A 0.99 |
| strip-s1 | ✓A 0.96 → ✓A 0.96 → ✓A 0.96 | ✓A 0.66 → ✓A 0.57 → ✓A 0.98 | ✓A 0.91 → ✓A 0.91 → ✓A 0.94 |

| Biến thể | 1.1 | thử 1.2 | người dùng |
|---|---|---|---|
| EN | 6/7 | 6/7 | 6/7 |
| VI có dấu | 3/7 | 2/7 | 7/7 |
| VI không dấu | 5/7 | 5/7 | 7/7 |
