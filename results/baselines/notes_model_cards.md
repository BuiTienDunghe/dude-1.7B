# Model-card notes: Mapika/decider-2b and Qwen/Qwen3.5-2B

Scope (per coordinator message): only `Mapika__decider-2b` and `Qwen__Qwen3.5-2B`. Quyet, Laya and Jeff were dropped and are not covered here.

Sources: local files only, under `C:\Users\dungbt06\Fine-tune\results\baselines\meta\<org>__<name>\` plus `baselines\pins_models.json`. Nothing was run, installed or fetched. (One stray no-op `python3 -` call with empty input was issued by mistake; it executed nothing and wrote nothing.) "không nêu" = not stated in the local files. Items marked "(derived from code)" come from the bundled `decider/*.py` files in the model repo, not from the README prose. Line numbers refer to the local file.

Pinned revisions (pins_models.json):

| model | sha | license | base_model | datasets | safetensors_params | pipeline_tag |
|---|---|---|---|---|---|---|
| Mapika/decider-2b | ff99089b6810691bd79134ec6febb5d9afa6e619 | apache-2.0 | Qwen/Qwen3.5-2B-Base | null | 1,881,825,088 | text-classification |
| Qwen/Qwen3.5-2B | 15852e8c16360a2fea060d615a32b45270f8a8fc | apache-2.0 | [Qwen/Qwen3.5-2B-Base] | null | 2,274,069,824 | image-text-to-text |

---

# 1. Mapika/decider-2b (v11)

## 1.1 How to load and call it

- Library: the bundled `decider/` package in the repo (README L162 comment: "decider/ is included in this repo"; L24: "`decider/` in this repository is the inference subset of that package"). The README also refers to the PyPI-style package version "decider-ai 1.4.0" (needed for `temperature_by_type`; "decider-ai 1.3.0 and earlier ignore the map and serve every answer at 1.145"). An exact `pip install ...` command is không nêu. The plain-transformers route is documented as an alternative (below).
- `trust_remote_code`: not mentioned anywhere and not needed. `decider/model.py` calls `AutoTokenizer.from_pretrained(name)` and `AutoModelForCausalLM.from_pretrained(name, dtype=dtype)` with no `trust_remote_code` (derived from code). `config.json`: `"architectures": ["Qwen3_5ForCausalLM"]`, `"model_type": "qwen3_5_text"`, `"transformers_version": "5.17.0"`, `"dtype": "bfloat16"`, `max_position_embeddings` 262144.
- Requirements (README L201-202): "Requirements: `torch`, `transformers>=5`, and `flash-linear-attention` (Triton kernels for the Qwen3.5 linear-attention layers; the model runs without it but several times slower). Python 3.11 or newer lets those kernels use `torch.compile`."
- The package picks device automatically (cuda, else mps, else cpu); on CUDA it uses shape-bucketed CUDA graphs unless `use_graphs=False` (derived from code, `decider/infer.py`).

README usage example 1 (verbatim, L161-168):

```python
from decider.infer import Decider          # decider/ is included in this repo
d = Decider("Mapika/decider-2b")
d.decide("My card was charged twice for the same purchase.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
# [{'choice': 'billing', 'confidence': 0.93, 'probs': {...}}, {'choice': 'yes', 'confidence': 0.70, 'probs': {...}}]   (v11, decider-ai 1.4.0, eager)
```

README usage example 2, Jev/TypeSafe-shaped request (verbatim, L176-186):

```python
d.system_one({"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
              "refund_policy": "Duplicate charges are eligible for a refund."},
             {"department": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                          "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
              "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
              "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
# {"model": "decider-2b-v11", "answers": {"department": {"type": "choice", "choice": "billing", "confidence": ..., "certainty": ..., "probabilities": {...}},
#  "refund_requested": {"type": "noul", "noul": ...}, "frustration": {"type": "score", "score": ..., "legend": {...}, ...}}, "usage": {...}}
```

README usage example 3, plain `transformers` with no helper package (verbatim, L204-217; `REPO` is not defined in the README):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(REPO); m = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).cuda().eval()
prompt = ("Context:\nMy card was charged twice for the same purchase.\n\n"
          "Question: Which department should handle this?\nOptions:\n(A) billing\n(B) technical support\n(C) sales\nAnswer: (")
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = m(**ids).logits[0, -1]
letters = [tok.encode(L, add_special_tokens=False)[0] for L in "ABC"]
probs = torch.softmax(logits[letters].float() / 1.164, -1)     # -> P(billing), P(technical support), P(sales); 1.164 is the stored choice temperature
```

Other README statements about calling it:
- "`decide_batch` scores many states, each with many questions, in one call. `abstain_below=t` returns `None` for decisions with confidence under `t`."
- "For several questions in one pass, append further `Question k: ... Answer k: (` blocks and read the logits at each `(` position (see `decider/prompt.py`)."
- Older revisions: `Decider(snapshot_download("Mapika/decider-2b", revision="v10"))` (tags `v10`, `v8`). Verbatim (L135-139):

```python
from huggingface_hub import snapshot_download
from decider.infer import Decider
d = Decider(snapshot_download("Mapika/decider-2b", revision="v10"))
```

- HTTP server: `POST /v1/systemone` (TypeSafe format) and `POST /decide`; "For the HTTP server, set `DECIDER_MODEL` to the same downloaded folder." (`decider/serve.py`).
- Adapter hints (derived from code, `decider/infer.py`):
  - `decide()` returns, per question, `dict(choice=..., confidence=<top prob>, probs={option text: p}, probs_list=[...in option order])`. The `probs` dict is keyed by option text, so duplicate option strings would collapse; `probs_list` keeps order.
  - `decide()` raises an AssertionError if a question does not have 2..255 options.
  - In `system_one`, `confidence` is TypeSafe's `(n·x_p_max − 1)/(n − 1)` and the raw top probability is `x_p_max` (README L237-244; "Before decider-ai 1.3.0, `confidence` in `system_one` and `POST /v1/systemone` answers was the probability that is now `x_p_max`. The plain form (`decide`, `POST /decide`) still reports the top probability as `confidence`.").
  - `noul` answer = probability of "yes" (README L236 "**noul**: probability of "yes""); options are `["no", "yes"]`, probability read from index 1.
  - `decider_config.json` has `"neutralize_none": false`. The code default (key missing) is `True`, which rewrites options like "none of the above"/"none"/"n/a" to "not listed here" and maps them back. For v11 the rewrite is off.

## 1.2 Input format

Two entry points:

A. `Decider.decide(context: str, questions: list[{"question": str, "options": list[str]}])`. Options are plain strings only (no separate label/description fields); to give a description, put it in the option string yourself. 2 to 255 options per question (README L170-172: "A question can have 2 to 255 options (more than 10 options use one label token per option, see `decider/prompt.py`)"). All questions of one request are packed into one row, one forward pass.

B. `Decider.system_one(state, questions)` (Jev/TypeSafe shape; `decider/systemone.py` docstring):
- `state`: str | dict | list; dict/list are serialised with `json.dumps(..., ensure_ascii=False)` after writing `_index` into arrays of length >= 8 (derived from code). Questions may name a part of the state by path (e.g. `ticket.messages[0].text`).
- `questions`: `{id: {"type": "choice", "instructions": ..., "criteria": {name: description | {...} | [...] | None}}` (up to 255 options); `{"type": "score", "instructions": ..., "criteria": [level 0 description, level 1 description, ...]}` (2..10 levels); `{"type": "noul", "instructions": ... (optional), "criteria": {"true": ..., "false": ...} (optional)}`. "a noul question needs instructions or at least one true/false description." "ids are never shown to the model. `instructions` and every description may be a string or any JSON value."
- So a choice option can carry a label plus a description: it is rendered as `"{name}: {description}"`, or just `name` when the description is None or "" (derived from code). The answer is reported under `name`. Score levels are rendered `"{i}: {description}"`. noul options are `"no"`/`"yes"`, with `": {description}"` appended if criteria give true/false text. If a noul question has no instructions the question text is the fixed sentence `Which answer fits the context?` (`NOUL_WITHOUT_INSTRUCTIONS`).
- Score with `isolated_levels: true` (set in decider_config.json; README: "Each Score level is likewise judged in its own row, without its number or its neighbours, and the per-level fits are normalised (`"isolated": false` restores listwise scoring)"). Template of each level row (verbatim from `decider/systemone.py`): `ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"` with options `["no", "yes"]`; a leading "N:" number is stripped from the level text. The score is the expected level over normalised P(yes) of the rows; answer also has `level_fit` and `fit_mass`.
- `independent=True` (default): every question is scored in its own row, "so an answer does not depend on which other questions are asked" (`independent=False` packs them into one row, "about half the latency for short states").

Prompt template. No system prompt, no chat template: `decider_config.json` has `"layout": "plain"` (the chat layout exists in the code but is not used by v11). The plain state-first layout built by `decider/prompt.py::build` (verbatim pieces):
- context: `"Context:\n" + example.context`
- per question k (suffix " k" only when the row has more than one question): `head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"`, options `f"\n({LETTERS[j]}) {q.options[oi]}"` for `LETTERS = "ABCDEFGHIJ"`, and `tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("`.
- Example for one question: `Context:\n<state>\n\nQuestion: <q>\nOptions:\n(A) <opt1>\n(B) <opt2>\nAnswer: (` (same as the README plain-transformers example).
- More than 10 options ("wide" rendering): each option is `"\n(" + <label token> + ") text"`, label tokens being A..Z then two-letter upper-case strings that are single tokens (AA, AB, ...), up to 255 labels.
- Option order is kept as given at inference (`_NoShuffle`); during training the order was shuffled.
- The package tokenizes the context and each question block separately and concatenates the token ids (derived from code); the README plain snippet tokenizes the whole string at once, so token boundaries may differ slightly. Use the package for exact reproduction.
- `decider_config.json` also has `"schema_first": false` and `"schema_first_trained": false`; `Decider.schema()` (questions-first layout) "still runs; its accuracy was not measured on v11".

How probabilities are read (README L222-232): "The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read from one pass." The slot is the last token of `Answer k: (`. Code: `logits = F.linear(hidden_at_slot, lm_head.weight[letters])`, masked to `-inf` beyond the number of options, divided by T, softmax. Each slot sees the context and the questions up to its own, but no earlier answers.

Input is English only (see 1.5).

## 1.3 Max input length, truncation, number of options

- Two different caps in the code:
  - `Decider.decide()` / `decide_batch()`: `max_ctx_tokens=1536` by default (`decider/infer.py`; HTTP `/decide` has `DECIDE_MAX_CTX_TOKENS = 1536  # /decide context cap, unchanged from 1.0.x` in `decider/serve.py`). Truncation is silent: `tok.encode("Context:\n" + example.context)[:max_ctx_tokens]` keeps the beginning, drops the end of the context (the `"Context:\n"` tokens count toward the cap). Question and option text is appended after the truncated context and is never truncated. Raising it is possible via `decide(..., max_ctx_tokens=N)` (passes through `**kw`).
  - `Decider.system_one()`: `max_state_tokens=32768` by default (also `max_fwd_tokens=65536`); `decider_config.json`: `"max_state_tokens": 32768`. Also silent head truncation (same slicing). README (L188): "The state may be a string, object or array (up to 32k tokens with the questions)." The server rejects (HTTP 413) a request when a row exceeds `DECIDER_MAX_ROW_TOKENS` (default `DECIDER_MAX_STATE_TOKENS + 4096`), rows exceed `DECIDER_MAX_ROWS` (1024), or the request exceeds `DECIDER_MAX_REQUEST_TOKENS` (1048576).
  - No error-instead-of-truncate mode found (no `strict` flag) in `infer.py`.
  - CUDA graph buckets go up to 2048 tokens; longer inputs run eagerly (derived from code, `engine.py`). Model `max_position_embeddings` = 262144 (config.json).
- Number of options: `decider_config.json` `"max_options": 255`; 2..255 for choice, 2..10 for score levels, 2 for noul. Accuracy cost with many options (README Limitations): "Full label sets cost accuracy against 10 sampled options: CLINC 151-way 0.88 against 0.98; DBpedia level 2 with 70 labels is the least calibrated case (ECE 0.14)." Training sub-sampled label sets to at most 10 options per example ("Large label sets were sub-sampled to at most 10 options per training example (gold always kept, order shuffled)"); evaluation protocol also "Large label sets sub-sampled to 10 options".

## 1.4 Calibration / temperature shipped

`decider_config.json` (verbatim values): `"temperature": 1.145`, `"temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124}`, `"requires": "decider-ai>=1.4.0 for temperature_by_type; older versions serve every answer at temperature"`. Probabilities are `softmax(letter_logits / T)`.
- Plain `decide()` questions are type "choice" (T = 1.164 with decider-ai >= 1.4.0; 1.145 with older versions). README transformers snippet: "1.164 is the stored choice temperature". In `system_one`, noul rows use 1.624, and every row of an isolated Score question uses the "score" T 1.124.
- Override: `Decider(path, temperature=T)` or env `DECIDER_TEMPERATURE=T` replaces `temperature` and switches the by-type map off (all answers use T).
- How fitted (README L296-299): "`temperature` 1.145, fitted by NLL on the in-task half of the public regression set without Banking77, CLINC-OOS, MMLU, ARC, Winogrande and HellaSwag (61 tasks, 102,804 rows). `temperature_by_type` fitted by NLL per answer type with `decider.calibrate.fit_by_type` on a pool of those regression rows and our own validation rows: 108,910 Choice answers, 1,677 yes/no answers and 535 Score answers." (eval_results.json `temperature_fit` repeats this.) Parent v10 used one T = 1.30 (v10 stays under Hub tag `v10`).
- No schema-first temperature in the config (`temperature_schema_first` absent).
- Remaining miscalibration (README): "Still overconfident on hard items: calibration error 0.156 on the held-out generated families, where our release limit is 0.08."

## 1.5 Languages (as claimed)

README front matter: `language: [en]`; Limitations (L474): "English only. Calibration is measured on public datasets and teacher-labelled probes, not on your traffic. Check it on your own labels before using confidence for routing." The card nevertheless reports a German MASSIVE number (`massive-de-DE` 0.783, not marked trained, in Bespoke's suite) but makes no claim of non-English support. Tag in pins: `en`.

## 1.6 Training data

Base: Qwen/Qwen3.5-2B-Base (pretraining data: không nêu). Quotes from the README:

- Intro (L16-23): "The supervised stages (v1 to v8) fine-tune it with cross-entropy, a proper scoring rule, on a mixture of about 95 public decision datasets, agent trajectories, web element choice, game states and teacher-written custom questions, in two prompt layouts and with isolated Score levels. v10 continued the v8 weights for 384 steps of calibration-aware reinforcement learning whose only rewards are outcomes (live browser task checkers and the exact probability laws of games) ... **This repository holds v11**: v10 plus a LoRA of rank 64, trained for 2 epochs on 42,749 rows of harder decisions and replay ..."
- Supervised stages (L249-257): "One epoch on a mixture of public decision datasets (intent detection, ticket routing, topic classification, sentiment, emotion, moderation, NLI, paraphrase, fact verification, passage relevance, reading comprehension, multiple-choice QA, ordinal rating scales, pairwise response preference, tool selection), then continuation epochs that added next-action choice from agent trajectories (AgentGym), web element choice (Mind2Web), teacher-written situations and game states, the input shapes of the Jev API ... The full list of components with sizes is in `decider/data/mixture.py` of the GitHub repository" (that file is not in the local folder).
- RL stage (v8 to v10): live MiniWoB++ tasks, minesweeper, game boards; "No gold labels were used."
- LoRA stage (v10 to v11), 42,749 rows: 8,000 generated decision families; 11,356 document questions written by Qwen3.6-27B; "human-labelled public sets (training halves)" 3,293 rows: "MMLU, ARC, CommonsenseQA, BoolQ, MNLI, SNLI, Banking77, RACE, OpenBookQA, LogiQA 2, MedQA, Winogrande"; 20,100 replay rows from the public mixture (trained toward v10's own distribution). "No JevBench item and no Decision Index item was used for training ..."
- Held-out definition (L303-307, verbatim): "'In-task' means the test splits of the training datasets; 'held-out' means datasets never seen in training (TREC, BBC news, PAWS, SciQ, Social IQa, StrategyQA, PubMedQA, TruthfulQA, tweet irony, financial sentiment, ADE, MASSIVE scenario, student question categories, Dolly categories, CR reviews, Financial PhraseBank, CommitmentBank, QuALITY, XStoryCloze, RewardBench, Arena preferences, Hermes tool selection, and an abstention probe)."

Status of the dev-set sources in the docs:

| source | status in decider-2b docs |
|---|---|
| MASSIVE (Amazon) | MIXED / internally inconsistent. (a) "held-out ... MASSIVE scenario" (L304-307, quoted above) and per-task row `massive_scenario | 0.766 / 0.040 | 0.756 / 0.041` (v8 / v10) in the held-out table. (b) Bespoke-suite table (L372-373): `massive-en-US (choice; trained) | 0.823 | 0.814` and `massive-de-DE (choice) | 0.797 | 0.783`. `eval_results.json` `bespoke`: `massive-en-US` `"trained": true`, `massive-de-DE` `"trained": false`. The card does not say which MASSIVE label field (intent or scenario) `massive-en-US` uses, nor which splits/locales were trained. So: en-US MASSIVE is flagged as trained in one table while "MASSIVE scenario" is listed as never seen in another; the card does not reconcile them. The word "intent detection" appears generically in the training list. Treat MASSIVE (at least en-US) as possibly used in training. |
| Belebele | not mentioned |
| SIB-200 | not mentioned |
| FLORES-200 | not mentioned |
| COPA / XCOPA | not mentioned. (XStoryCloze is listed as held-out; Winogrande is in the LoRA training halves; neither is COPA.) |
| SuperGLUE | the suite is not named. Component tasks that are named: BoolQ is in the training list ("human-labelled public sets (training halves)" L278, and Bespoke `boolq (noul; trained)`); CommitmentBank (`cb`) is listed as held-out (L306; per-task `cb | 0.911 / 0.090 | 0.857 / 0.093`). |

Same pattern as MASSIVE for PAWS: listed as held-out in L305 and per-task table, but Bespoke table says `paws (noul; trained)` (L376, eval_results `"trained": true`). The card does not reconcile this either; the "held-out" list is stated for the "94 public tasks, original protocol".

## 1.7 Benchmark numbers and examples with stated output

Protocol (README L303-): "Large label sets sub-sampled to 10 options; one temperature fitted on in-task data and stored in `decider_config.json`." ECE with 15 bins.

| model | in-task acc / NLL / ECE | held-out acc / NLL / ECE |
|---|---|---|
| Qwen3.5-2B-Base, zero-shot (69 / 24 tasks) | 0.620 / 0.908 / 0.121 | 0.642 / 0.853 / 0.105 |
| decider-2b v8, T=1.30 | 0.811 / 0.460 / 0.037 | 0.741 / 0.655 / 0.088 |
| decider-2b v10, rebuilt set (67 / 28 tasks), T=1.30 | 0.805 / 0.474 / 0.037 | 0.755 / 0.622 / 0.084 |
| **decider-2b v11 (this repo), rebuilt set, per-type map** | 0.802 / 0.481 / 0.038 | 0.752 / 0.626 / 0.083 |
| v8, questions-first layout (schema cache), T=1.18 | 0.790 / 0.500 / 0.038 | 0.707 / 0.757 / 0.104 |

("rebuilt set" numbers are comparable to each other, not to the rows above them; README L320-322.)

Other v11 numbers (README "Changes from v10" table): held-out generated families (5,000 rows) acc 0.429, ECE 0.156; held-out document questions (449 rows) 0.753 / ECE 0.075; human-labelled public sets (1,595 rows) 0.781; knowledge guard (MMLU, ARC and others; 2,994 rows) 0.715; OpenJev (5,252 rows) 64.6% / NLL 0.860; Mind2Web (1,770 rows) 83.9%; TypeSafe workflow decisions (102 rows) 75.5%; JevBench public easy / standard / hard 1.000 / 0.889 / 0.577 (hard-tier ECE 0.175); Bespoke public suite (13 subsets, 3,880 records) macro / micro 0.706 / 0.711 (macro over the subsets not trained on 0.645); live MiniWoB++ sampled 90.3%; MNLI validation_matched (9,815 items) 0.864. Bespoke subsets include `massive-en-US` 0.814 (trained flag true), `massive-de-DE` 0.783, `boolq` 0.850 (trained), `paws` 0.760 (trained).

Per-task held-out accuracy / ECE shown only for v8 and v10 ("not measured per task for v11"), relevant lines (v10): `massive_scenario 0.756 / 0.041`, `xstory_cloze 0.962 / 0.017`, `cb 0.857 / 0.093`, `trec 0.784 / 0.066`, `sciq 0.982 / 0.024`, `social_iqa 0.708 / 0.077`.

README examples with a stated expected output:
1. Usage example 1 (input in 1.1). Stated output (verbatim comment): `# [{'choice': 'billing', 'confidence': 0.93, 'probs': {...}}, {'choice': 'yes', 'confidence': 0.70, 'probs': {...}}]   (v11, decider-ai 1.4.0, eager)`. Exact `probs` values: không nêu.
2. `system_one` example: stated output is only the schema (`"model": "decider-2b-v11"`, `answers` with `choice: "billing"`, `confidence: ...`), numbers elided: không nêu.
3. Plain-transformers snippet: comment "-> P(billing), P(technical support), P(sales)"; numeric values không nêu.
Also eval_results.json holds per-set metrics, fixtures, games, browser, Bespoke and JevBench results, issue-#9 cases (no raw input/output text for the README examples).

## 1.8 License

Apache-2.0 (README front matter `license: apache-2.0`; pins `apache-2.0`). Base model Qwen/Qwen3.5-2B-Base.

---

# 2. Qwen/Qwen3.5-2B

## 2.0 What kind of model it is

- Post-trained (chat/instruct-style, hybrid thinking/non-thinking), not the base. README L17: "This repository contains model weights and configuration files for the post-trained model in the Hugging Face Transformers format." "Training Stage: Pre-training & Post-training". Front matter: `base_model: - Qwen/Qwen3.5-2B-Base` (the pretrained base is a separate repo; decider-2b is fine-tuned from that Base, not from this model). pins tags: `conversational`, `image-text-to-text`, pipeline_tag `image-text-to-text`.
- It is a multimodal causal LM with vision encoder: "Type: Causal Language Model with Vision Encoder"; `config.json` `"architectures": ["Qwen3_5ForConditionalGeneration"]`, `"model_type": "qwen3_5"` with `text_config.model_type` `qwen3_5_text` (24 layers, hidden 2048, vocab 248320, tied embeddings). decider-2b ships the text-only `Qwen3_5ForCausalLM` class instead. `chat_template.jinja` of decider-2b is byte-identical to this repo's (diff check), but decider v11 uses `"layout": "plain"`, so it does not use the template.
- Intended use (README L21): "In light of its parameter scale, the intended use cases are prototyping, task-specific fine-tuning, and other research or development purposes."
- It is not a decision model: no state/question/options interface, no probability output, no calibration.

## 2.1 How to load and call it

- Library: `library_name: transformers`. The README has NO `AutoModel...from_pretrained` snippet. The Quickstart is about serving (SGLang, vLLM, KTransformers, `transformers serve`) and calling an OpenAI-compatible chat-completions API. It says: "Hugging Face Transformers contains a _lightweight_ server ... The latest `transformers` is required for Qwen3.5:" followed by:

```shell
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
```

  "Please also make sure torchvision and pillow are installed." Then:

```shell
transformers serve --force-model Qwen/Qwen3.5-2B --port 8000 --continuous-batching
```

  vLLM text-only (skips the vision encoder): `vllm serve Qwen/Qwen3.5-2B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --language-model-only`.
- `trust_remote_code`: not mentioned and not shown in any example. `config.json` has `"transformers_version": "4.57.0.dev0"`; README says the latest/main transformers is required.
- Text-only call example (verbatim):

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()
messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=messages,
    max_tokens=32768,
    temperature=1.0,
    top_p=1.0,
    presence_penalty=2.0,
    extra_body={
        "top_k": 20,
    }, 
)
print("Chat response:", chat_response)
```

  with env: `pip install -U openai`, `export OPENAI_BASE_URL="http://localhost:8000/v1"`, `export OPENAI_API_KEY="EMPTY"`.
- Thinking-mode example (verbatim; input `Type "I love Qwen3.5" backwards`; stated output: không nêu):

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()
messages = [
    {"role": "user", "content": "Type \"I love Qwen3.5\" backwards"},
]
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=messages,
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "enable_thinking": True,
    }, 
)
print("Chat response:", chat_response)
```

## 2.2 Input format and how an answer would be read

- Input: chat messages `[{"role": "user", "content": "..."}]`; images/video optional (content list with `image_url`/`video_url`). No fields for state, question, options, or criteria. No prescribed system prompt. No prompt template for classification.
- The only guidance about options is Best Practices item 3 (verbatim): "**Standardize Output Format**: We recommend using prompts to standardize model outputs when benchmarking. - **Math Problems**: Include "Please reason step by step, and put your final answer within \boxed{}." in the prompt. - **Multiple-Choice Questions**: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.""
- Probabilities: not provided by the model card. How to read option probabilities (letter tokens, a head, etc.) is không nêu; any logit-based letter readout would be the evaluator's own adapter design.
- Chat template (`chat_template.jinja`, final lines, verbatim):

```
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

## 2.3 Thinking mode (enable_thinking)

- Yes, the chat template supports `enable_thinking`. If `enable_thinking` is defined and true, the generation prompt ends with `<|im_start|>assistant\n<think>\n` (the model then writes its reasoning). In every other case (variable undefined, or false) the generation prompt ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` (an empty think block, i.e. non-thinking). So the default is non-thinking, matching the README: "Qwen3.5 models support both non-thinking and thinking mode. **Qwen3.5-2B operates in non-thinking mode by default**. To enable thinking, refer to the examples [here](#thinking-mode)."
- README: "Qwen3.5 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`." The switch is an API/template parameter: `extra_body={"top_k": 20, "enable_thinking": True}` in the OpenAI example; the Qwen-Agent comment shows `'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}`. A direct `tokenizer.apply_chat_template(..., enable_thinking=...)` call is not shown in the README (the template reads that variable).
- History: "**No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2."
- Caveat for this size (verbatim): "In thinking mode, we have observed that when using the recommended sampling parameters, Qwen3.5-2B is more prone to entering thinking loops compared to other Qwen3.5 models, which may prevent it from terminating generation properly. We recommend further tuning the sampling parameters specific to your use case and utilizing the API's streaming generation mode (if supported) to enable timely detection and interruption of such anomalous generation behaviors."

## 2.4 Max length, truncation, option count

- "Context Length: 262,144 natively" (README Model Overview); `config.json` `max_position_embeddings` 262144; README "The model has a default context length of 262,144 tokens. If you encounter out-of-memory (OOM) errors, consider reducing the context window." Truncation behaviour: không nêu (depends on the serving framework; no truncation option in the model card). Option-count limit: not applicable / không nêu.
- Output length recommendation: 32,768 tokens for most queries, 81,920 for complex math/programming; vision preprocessor settings are separate and irrelevant for text.

## 2.5 Calibration / temperature

- No calibration temperature shipped, no confidence output. (`generation_config.json` is not listed among the repo files in pins_models.json, so no shipped generation defaults in the local metadata; `config.json` has none.)
- Recommended sampling parameters (README "Tip" box in the chat-completions section and "Best Practices" item 1, verbatim):
  - Non-thinking mode for text tasks: `temperature=1.0, top_p=1.00, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0`
  - Non-thinking mode for VL tasks: `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
  - Thinking mode for text tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
  - Thinking mode for VL or precise coding (e.g. WebDev) tasks: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`
  - "For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance." Also "Please note that the support for sampling parameters varies according to inference frameworks."
  - Benchmark settings stated for the language table: "top_p=0.95, top_k=20, presence_penalty=1.5, and temperature=1.0 were used." (VL table: thinking temperature 0.6, non-thinking 0.7 with top_p=0.8.)
- For logit/letter-readout scoring (no sampling), none of these apply; the card says nothing about greedy decoding or logit scaling.

## 2.6 Languages

README highlight: "**Global Linguistic Coverage**: Expanded support to 201 languages and dialects, enabling inclusive, worldwide deployment with nuanced cultural and regional understanding." The list of the 201 languages is không nêu. Multilingual benchmarks reported (below). No `language:` tags in front matter or pins.

## 2.7 Training data

The README gives no dataset names and no data description. Only generic highlights: "Unified Vision-Language Foundation: Early fusion training on multimodal tokens ...", "Scalable RL Generalization: Reinforcement learning scaled across million-agent environments with progressively complex task distributions", "Next-Generation Training Infrastructure: Near-100% multimodal training efficiency compared to text-only training and asynchronous RL frameworks". Training stage: "Pre-training & Post-training". Pins `datasets: null`.

Dev-set sources: MASSIVE: not mentioned (the only text hit for "massive" is "massive-scale agent scaffolds", unrelated). Belebele: not mentioned. SIB-200: not mentioned. FLORES-200: not mentioned. COPA / XCOPA / SuperGLUE: not mentioned. (The multilingual evaluation sets named, MMMLU, MMLU-ProX, NOVA-63, INCLUDE, Global PIQA, PolyMATH, WMT24++, MAXIFE, are evaluations only; pretraining data composition is not described, so the card neither includes nor excludes these sources.)

## 2.8 Benchmarks reported (Qwen3.5-2B column; README tables, columns are Qwen3-4B-2507 / Qwen3-1.7B / Qwen3.5-2B / Qwen3.5-0.8B)

- Instruct (Non-Thinking): MMLU-Pro 55.3, MMLU-Redux 69.2, C-Eval 65.2, SuperGPQA 30.4, IFEval 61.2, MMMLU 56.9.
- Knowledge & STEM (Thinking): MMLU-Pro 66.5, MMLU-Redux 79.6, C-Eval 73.2, SuperGPQA 37.5, GPQA 51.6.
- Instruction following (Thinking): IFEval 78.6, IFBench 41.3, MultiChallenge 33.7.
- Long context (Thinking): AA-LCR 25.6, LongBench v2 38.7.
- Reasoning (Thinking): HMMT Feb 25 22.9, HMMT Nov 25 19.6.
- General agent (Thinking): BFCL-V4 43.6, TAU2-Bench 48.8.
- Multilingualism (Thinking): MMMLU 63.1, MMLU-ProX 52.3 (29 languages), NOVA-63 46.4, INCLUDE 55.4, Global PIQA 69.3, PolyMATH 26.1, WMT24++ 45.8 (55 languages, XCOMET-XXL), MAXIFE 60.6 (23 settings).
- Vision-language tables also exist (MMMU 64.2/64.2 thinking/non-thinking, etc.); not relevant to text decisions.
- No decision/classification-with-probabilities benchmark, no README example with a stated expected output (the thinking example `Type "I love Qwen3.5" backwards` has no stated output: không nêu).

## 2.9 License

Apache-2.0 (front matter `license: apache-2.0`, `license_link: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE`; pins `apache-2.0`).

---

# 3. Summary table

| model | params (pins safetensors / README) | library | trust_remote_code | max length | languages (claimed) | MASSIVE / Belebele / SIB / FLORES / COPA mentioned in training data |
|---|---|---|---|---|---|---|
| Mapika/decider-2b v11 | 1,881,825,088 / "1.9B" | bundled `decider/` package (`decider-ai >= 1.4.0` for the per-type temperature map); also plain `transformers>=5` letter readout (README snippet) | not mentioned, not used in code | 1536 context tokens in `decide()`; 32768 state tokens in `system_one()`; silent head truncation; no strict mode; 2..255 options (score 2..10 levels) | English only (`language: [en]`, "English only.") | MASSIVE: inconsistent (en-US flagged "trained" in Bespoke table / eval_results.json; "MASSIVE scenario" listed as held-out); Belebele: no; SIB-200: no; FLORES-200: no; COPA/XCOPA: no (SuperGLUE suite not named; BoolQ trained, CommitmentBank held-out; XStoryCloze held-out) |
| Qwen/Qwen3.5-2B (post-trained) | 2,274,069,824 / "2B" | `transformers` (latest/main) or vLLM/SGLang/KTransformers serving; no from_pretrained snippet; no decision interface | not mentioned | 262,144 natively; truncation behaviour không nêu; option limit n/a | "201 languages and dialects" (list không nêu) | none mentioned (training data not described at all) |

Open items for the adapter author:
- decider-2b: choose between `decide()` (1536-token context cap, `confidence` = top prob) and `system_one()` (32768-token cap, TypeSafe `confidence` plus `x_p_max`); choice temperature is 1.164 only with decider-ai >= 1.4.0, else 1.145; the README plain-transformers snippet reproduces the single-question `decide()` prompt but tokenizes it in one piece.
- Qwen3.5-2B: the card gives no decision protocol; the only option-related prompt guidance is the `"answer": "C"` JSON instruction, thinking is off by default via the template, and the template reads `enable_thinking`.
- Quyet, Laya and Jeff were read before the scope change but are intentionally omitted here.
