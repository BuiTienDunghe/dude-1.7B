# xcopa:81: chấm thử bản dịch của người dùng (DEV, một câu; đáp án B)

Ô = 1.1 → bản thử 1.2 → bản người dùng. Mỗi mốc: ✓/✗, chữ cái dự đoán, P(đáp án đúng). EN giữ nguyên bản 1.1.

| Model | EN | VI có dấu | VI không dấu |
|---|---|---|---|
| decider-2b-text | ✓B 0.84 → ✓B 0.84 → ✓B 0.84 | ✗A 0.38 → ✗A 0.42 → ✗A 0.36 | ✗A 0.45 → ✗A 0.43 → ✗A 0.43 |
| decider-2b-keyed | ✓B 0.85 → ✓B 0.85 → ✓B 0.85 | ✗A 0.37 → ✗A 0.37 → ✗A 0.32 | ✗A 0.43 → ✗A 0.43 → ✗A 0.43 |
| decider-ft-ep2-text | ✓B 0.98 → ✓B 0.98 → ✓B 0.98 | ✗A 0.32 → ✗A 0.28 → ✗A 0.23 | ✗A 0.47 → ✗A 0.45 → ✗A 0.47 |
| decider-ft-ep2-keyed | ✓B 0.98 → ✓B 0.98 → ✓B 0.98 | ✗A 0.26 → ✗A 0.14 → ✗A 0.14 | ✗A 0.45 → ✗A 0.42 → ✗A 0.45 |
| Qwen3-1.7B-fullvocab-s1 | ✗A 0.02 → ✗A 0.02 → ✗A 0.02 | ✗A 0.24 → ✗A 0.03 → ✗A 0.05 | ✗A 0.12 → ✗A 0.38 → ✗A 0.41 |
| Qwen3-1.7B | ✗A 0.00 → ✗A 0.00 → ✗A 0.00 | ✓B 0.94 → ✗A 0.44 → ✓B 0.56 | ✗A 0.05 → ✗A 0.02 → ✗A 0.11 |
| strip-s1 | ✗A 0.02 → ✗A 0.02 → ✗A 0.02 | ✗A 0.10 → ✗A 0.03 → ✗A 0.05 | ✗A 0.44 → ✗A 0.32 → ✗A 0.38 |

| Biến thể | 1.1 | thử 1.2 | người dùng |
|---|---|---|---|
| EN | 4/7 | 4/7 | 4/7 |
| VI có dấu | 1/7 | 0/7 | 1/7 |
| VI không dấu | 0/7 | 0/7 | 0/7 |
