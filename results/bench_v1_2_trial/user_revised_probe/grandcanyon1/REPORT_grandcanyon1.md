# Grand_Canyon#1: chấm thử bản dịch của người dùng (DEV, một câu; đáp án A)

Ô = 1.1 → bản thử 1.2 → bản người dùng. Mỗi mốc: ✓/✗, chữ cái dự đoán, P(đáp án đúng). EN giữ nguyên bản 1.1.

| Model | EN | VI có dấu | VI không dấu |
|---|---|---|---|
| decider-2b-text | ✗B 0.27 → ✗B 0.27 → ✗B 0.27 | ✗B 0.24 → ✗B 0.23 → ✓A 0.47 | ✗B 0.06 → ✗B 0.07 → ✓A 0.34 |
| decider-2b-keyed | ✗B 0.23 → ✗B 0.23 → ✗B 0.23 | ✗B 0.22 → ✗B 0.24 → ✓A 0.42 | ✗B 0.07 → ✗B 0.08 → ✓A 0.36 |
| decider-ft-ep2-text | ✗B 0.23 → ✗B 0.23 → ✗B 0.23 | ✓A 0.39 → ✗B 0.22 → ✓A 0.77 | ✗B 0.10 → ✗B 0.11 → ✗D 0.38 |
| decider-ft-ep2-keyed | ✗B 0.19 → ✗B 0.19 → ✗B 0.19 | ✓A 0.42 → ✗B 0.26 → ✓A 0.69 | ✗B 0.12 → ✗B 0.12 → ✗D 0.39 |
| Qwen3-1.7B-fullvocab-s1 | ✓A 0.62 → ✓A 0.62 → ✓A 0.62 | ✗B 0.20 → ✗B 0.20 → ✓A 0.40 | ✗B 0.10 → ✗B 0.10 → ✗B 0.20 |
| Qwen3-1.7B | ✓A 1.00 → ✓A 1.00 → ✓A 1.00 | ✗D 0.43 → ✗D 0.41 → ✓A 0.94 | ✓A 0.65 → ✗D 0.36 → ✗D 0.01 |
| strip-s1 | ✓A 0.47 → ✓A 0.47 → ✓A 0.47 | ✗B 0.13 → ✗C 0.08 → ✗B 0.23 | ✗B 0.15 → ✗B 0.14 → ✓A 0.36 |

| Biến thể | 1.1 | thử 1.2 | người dùng |
|---|---|---|---|
| EN | 3/7 | 3/7 | 3/7 |
| VI có dấu | 2/7 | 0/7 | 6/7 |
| VI không dấu | 1/7 | 0/7 | 3/7 |
