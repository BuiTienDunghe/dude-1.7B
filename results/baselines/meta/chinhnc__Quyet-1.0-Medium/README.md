---
license: apache-2.0
library_name: quyet
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
tags:
- quyet
- decision-model
- calibrated
- classification
---

# Quyet-1.0-Medium

Quyet-1.0-Medium is a **decision model**: given a state (any text, JSON or conversation) and one or more typed questions, it picks one option per question and returns calibrated probabilities. Question types: `choice` (pick one label), `score` (an ordered scale) and `noul` (true / false). It is part of the Quyet 1.0 family (Large, Medium, Small, Small-EN, Tiny), released by Chinh Nguyen under Apache-2.0.

| | |
|---|---|
| Base model | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Architecture | Qwen3.5-4B with a merged LoRA fine-tune (rank 16), letter-readout decision prompt |
| Parameters | 4.66B (4.21B text) |
| Languages | English; also tuned for Vietnamese. Other languages work, with lower accuracy. |
| Input | state up to 6,000 tokens inside an 8,000-token prompt |
| Weights | 9.3 GB (bf16) |
| License | Apache-2.0 (see LICENSE and NOTICE) |
| Homepage | [quyet.ai](https://quyet.ai) |

## How to use

**Live demo:** try Quyet-1.0-Large in your browser at [quyet.ai](https://quyet.ai).

```bash
pip install quyet
```

```python
import quyet

m = quyet.load("chinhnc/Quyet-1.0-Medium")          # pip install quyet; downloads from Hugging Face
r = m.predict(
    {"message": "Please close my card, I lost it yesterday."},
    {"intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
     "urgent": {"type": "noul", "instructions": "The request is urgent."},
     "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}},
)
print(r["answers"])   # {"intent": {"choice": ..., "confidence": ..., "probabilities": {...}}, "urgent": {"noul": P(true)}, ...}
```

Runs in bf16 on a 16 GB GPU. The model answers by reading the next-token probabilities of the option letters (A, B, ...) after a fixed prompt; the `quyet` package builds that prompt and applies the calibrated temperatures. It also loads with transformers (and serves with vLLM) as a standard Qwen3.5-4B-architecture checkpoint, but the decision prompt and temperatures live in the package and in `quyet_config.json`. This model uses prompt version 1, the prompt it was trained with.

Answers follow the TypeSafe `/v1/systemone` shape: `choice` (with `probabilities`), `score` (expected level, `probabilities`, `legend`) and `noul` (P(true)). At most 10 options per question. Only the state is ever truncated: conversation lists keep their most recent turns, other states keep their beginning.

## Credits

- Qwen3.5 by Alibaba Cloud (Apache-2.0).

## Citation

```bibtex
@misc{quyet2026,
  title  = {Quyet 1.0: calibrated decision models},
  author = {Chinh Nguyen},
  year   = {2026},
  url    = {https://huggingface.co/chinhnc/Quyet-1.0-Medium}
}
```

## License

Apache-2.0. Keep the NOTICE file (it starts with "Quyet by Chinh Nguyen") when you redistribute this model or anything derived from it. Questions and issues: email@quyet.ai.
