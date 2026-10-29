---
license: apache-2.0
library_name: quyet
base_model: jhu-clsp/mmBERT-small
base_model_relation: finetune
tags:
- quyet
- decision-model
- calibrated
- classification
---

# Quyet-1.0-Tiny

Quyet-1.0-Tiny is a **decision model**: given a state (any text, JSON or conversation) and one or more typed questions, it picks one option per question and returns calibrated probabilities. Question types: `choice` (pick one label), `score` (an ordered scale) and `noul` (true / false). It is part of the Quyet 1.0 family (Large, Medium, Small, Small-EN, Tiny), released by Chinh Nguyen under Apache-2.0.

| | |
|---|---|
| Base model | [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) |
| Architecture | mmBERT-small (first 16 layers) + wide option-query head, distilled from Quyet-1.0-Medium |
| Parameters | 183M |
| Languages | English; also tuned for Vietnamese. Other languages work, with lower accuracy. |
| Input | 8,192 tokens |
| Weights | 732 MB (F32) |
| License | Apache-2.0 (see LICENSE and NOTICE) |
| Homepage | [quyet.ai](https://quyet.ai) |

## How to use

**Live demo:** try Quyet-1.0-Large in your browser at [quyet.ai](https://quyet.ai).

```bash
pip install quyet
```

```python
import quyet

m = quyet.load("chinhnc/Quyet-1.0-Tiny")          # pip install quyet; downloads from Hugging Face
r = m.predict(
    {"message": "Please close my card, I lost it yesterday."},
    {"intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
     "urgent": {"type": "noul", "instructions": "The request is urgent."},
     "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}},
)
print(r["answers"])   # {"intent": {"choice": ..., "confidence": ..., "probabilities": {...}}, "urgent": {"noul": P(true)}, ...}
```

Runs on CPU or any GPU. It reads the question, every option and the state in one pass (8,192 tokens) and scores each option; probabilities are temperature-calibrated per question type and option count.

Answers follow the TypeSafe `/v1/systemone` shape: `choice` (with `probabilities`), `score` (expected level, `probabilities`, `legend`) and `noul` (P(true)). At most 10 options per question. Only the state is ever truncated: conversation lists keep their most recent turns, other states keep their beginning.

## Credits

- mmBERT-small by the mmBERT authors, JHU CLSP (MIT); first 16 of its 22 layers.
- Distilled from Quyet-1.0-Medium, which is fine-tuned from Qwen3.5-4B by Alibaba Cloud (Apache-2.0): https://huggingface.co/Qwen/Qwen3.5-4B.
- The option rendering follows Laya by Convai Innovations (Apache-2.0); no Laya weights are used.

## Citation

```bibtex
@misc{quyet2026,
  title  = {Quyet 1.0: calibrated decision models},
  author = {Chinh Nguyen},
  year   = {2026},
  url    = {https://huggingface.co/chinhnc/Quyet-1.0-Tiny}
}
```

## License

Apache-2.0. Keep the NOTICE file (it starts with "Quyet by Chinh Nguyen") when you redistribute this model or anything derived from it. Questions and issues: email@quyet.ai.
