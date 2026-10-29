---
license: apache-2.0
language:
- vi
base_model: convaiinnovations/laya
library_name: transformers
pipeline_tag: text-classification
tags:
- laya
- system-one
- calibrated-decisions
- rlcd
- classification
- routing
- scoring
- moderation
- vietnamese
---

# laya-vi

**A Vietnamese fine-tune of [Laya](https://huggingface.co/convaiinnovations/laya).** Give it a **state** (an email,
report, chat log or document) and **typed questions**, and it returns typed answers with calibrated probabilities
in a single forward pass. Like Laya, it never generates text, so there is nothing to parse and nothing to
hallucinate.

`laya-vi` starts from the `laya-multilingual` checkpoint and is trained on Vietnamese business decisions. On
Vietnamese decisions from the domains it was trained on, accuracy rises from **0.45 to 0.89**. On domains held
out from training it also improves, and its probabilities are much better calibrated than the base checkpoint's.

## Installation

```bash
pip install laya
```

`laya-vi` uses the standard Laya checkpoint format, so the Laya runtime loads it with no extra code.

## Quickstart

```python
import laya

agent = laya.load("vanty120/laya-vi")

state = ("Chào shop, tôi đã chuyển khoản 2 lần cho đơn hàng #4521 tháng 3. "
         "Nhờ shop hoàn lại khoản bị trừ trùng trong hôm nay, nếu không tôi sẽ huỷ tài khoản.")
questions = {
    "bo_phan": {"type": "choice", "instructions": "Bộ phận nào nên xử lý yêu cầu này?",
                "criteria": {"thanh_toan": "hoá đơn, thanh toán, hoàn tiền",
                             "ky_thuat": "lỗi, sự cố, hệ thống",
                             "khac": "các vấn đề khác"}},
    "muc_do_khan": {"type": "score", "instructions": "Yêu cầu này khẩn cấp đến mức nào?",
                    "criteria": ["không gấp", "sớm", "rất gấp"]},
    "nguy_co_roi_bo": {"type": "noul", "instructions": "Khách hàng có đe doạ huỷ hoặc rời bỏ dịch vụ không?"},
}

result = agent.predict(state, questions)
print(result["answers"]["bo_phan"]["choice"])        # thanh_toan   (p = 0.96)
print(result["answers"]["muc_do_khan"]["score"])     # 1.71 on a 0–2 scale ("rất gấp": p = 0.75)
print(result["answers"]["nguy_co_roi_bo"]["noul"])   # 0.94  (probability the answer is yes)
```

The three question types:

| Type | Answer |
|---|---|
| `noul` | probability that the answer is yes |
| `choice` | one of the listed options (3–10), with a probability for each |
| `score` | expected position on an ordinal scale (3–5 levels), with a probability for each level |

It also works with `laya-serve` and the other Laya integrations. See the
[Laya documentation](https://nandhakishorm.github.io/laya/) for details.

## Results

| Test set | n | Accuracy (laya-multilingual → laya-vi) | ECE (laya-multilingual → laya-vi) |
|---|---|---|---|
| Trained domains, unseen cases | 13,130 | 0.452 → **0.890** | 0.338 → **0.081** |
| Held-out domains (zero-shot) | 6,220 | 0.498 → **0.596** | 0.290 → **0.091** |
| MASSIVE vi-VN (human labels) | 1,000 | 0.520 → **0.571** | 0.218 → **0.134** |

`n` counts (document, question) pairs. `laya-multilingual` is the original Laya checkpoint, evaluated zero-shot.
ECE is expected calibration error, where lower means the probabilities better match observed accuracy.
Brier score, KL, score MAE and other metrics are in [`eval_results.json`](eval_results.json).

Median latency is about 24 ms per request on a single GPU.

## Training data

The training data is Vietnamese and fully synthetic. No real customer or user data was used. It covers everyday
decisions in Vietnamese business settings:

- customer-service emails
- invoice reconciliation
- power-grid incident reports
- occupational safety
- tender and bidding documents
- recruitment
- product reviews
- comment moderation
- project team group chats
- subcontracting agreements

The documents use Vietnamese names, companies, currency and workplace conventions. About a quarter of them also
appear without diacritics (*không dấu*), as Vietnamese is often typed that way. Two of the domains are held out
of training and used only to measure zero-shot performance.

Documents are generated from known underlying facts. Each case is labeled by an LLM teacher three times, and the
gold label is the average of the three answer distributions. Automatic checks remove cases where the label
contradicts the facts the document was built from, near-duplicates and malformed text.

## Training procedure

- 1 epoch of soft cross-entropy against the teacher distributions, then 3 epochs of RLCD (Laya's
  reinforcement-learning objective based on proper scoring rules). English replay makes up 15% of training, to
  limit forgetting.
- The best epoch is chosen on a separate calibration split. Temperatures are then fitted per question type on
  that split.
- About 1.8 hours on a single NVIDIA GB10. The full configuration is in [`train_log.json`](train_log.json).

## Limitations

- Gains are largest on the domains in the training data. On new domains, expect results closer to the held-out
  row above, and fine-tune on your own decisions where accuracy matters.
- Labels come from an LLM teacher, so the model can inherit that teacher's mistakes and biases.
- It is not a substitute for human judgement in safety-critical decisions.

## License and acknowledgements

Apache 2.0, the same license as the base model. `laya-vi` is built on
[Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations
([GitHub](https://github.com/NandhaKishorM/laya)). Please credit Laya when you use this model.
