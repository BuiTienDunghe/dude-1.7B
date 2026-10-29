# Code review: bundled `decider` package of Mapika/decider-2b (v11)

Scope (after coordinator scope change): only the `decider/` inference subset shipped inside the HF repo Mapika/decider-2b.
`quyet` and `laya` were NOT reviewed for this document (see "Not covered" at the end).

Reviewed (read in full or grepped; nothing was run, imported, installed, or fetched):
`C:\Users\dungbt06\Fine-tune\results\baselines\meta\Mapika__decider-2b\decider__*.py` (flat names, `__` = `/`; so `decider__infer.py` = `decider/infer.py`),
plus `config.json`, `decider_config.json`, `generation_config.json`, `README.md`, head of `eval_results.json`.
Line numbers below are line numbers of those local files. Local copies contain only the `.py` files and small JSON/MD files, no weights, so weight-file format was NOT checked.

Files in the package: `__init__` (empty), `batching`, `calibrate`, `engine`, `engine_v2`, `fp8`, `infer`, `metrics`, `model`, `mps_moe`, `mps_ops`, `prompt`, `prompt_fast`, `schema_engine`, `serve`, `shared_prefix`, `systemone`, `temperature`.

---------------------------------------------------------------------------------------------------

## Task A: security review

### Verdict
**Nothing unusual found on the scoring path.** No telemetry, no outbound network other than Hugging Face Hub downloads (via `huggingface_hub` / `transformers`), no file writes by the package itself, no exec/eval/pickle/torch.load/subprocess/trust_remote_code, no obfuscation. The package does monkey-patch `transformers` and torch global settings in-process and (on the default CUDA path) triggers `torch.compile`; these are listed below as behaviours to be aware of, not as malicious findings.

Methods used: full read of `infer.py`, `prompt.py`, `systemone.py`, `temperature.py`, `model.py`, `engine.py`, `calibrate.py`, `fp8.py`, `metrics.py`, `mps_moe.py`, `mps_ops.py`, `prompt_fast.py`, head of `engine_v2.py`, `schema_engine.py`, `batching.py`, head+tail of `serve.py`; regex sweeps over all 19 files for network / file / exec / env / obfuscation patterns. Sweeps: no `://` anywhere in the code; no line longer than 400 chars; no 80+ char base64-like run; no `base64`, `marshal`, `zlib`, `gzip`, `lzma`, `bz2`, `fromhex`, `decompress`, `codecs`; no `import requests|urllib|socket|http|httpx|aiohttp|subprocess|pickle|ctypes`; no `torch.load`, `trust_remote`, `os.system`, `subprocess`, `exec(`, Python `eval(`.
Not independently verified: I did not read the middle of `serve.py` (lines ~130-340, request-prep/batcher code), `engine_v2.py` lines 80-213, `schema_engine.py` lines 60-150, `shared_prefix.py`, `batching.py` lines 30-95 line-by-line; those were covered by the regex sweeps only (and they are not on the normal scoring path except `shared_prefix.score_shared`, which the sweep showed only does cache manipulation + one env read).

### A1. Network access

| Where | Code | Reached on "load from HF Hub + score one question"? |
|---|---|---|
| `infer.py:92-93` | `from huggingface_hub import hf_hub_download` / `cfg_path = os.path.join(path, "decider_config.json") if os.path.isdir(path) else hf_hub_download(path, "decider_config.json")` | YES if `path` is a repo id (not a local folder). Hub only. |
| `model.py:12-13` | `self.tok = AutoTokenizer.from_pretrained(name)` / `self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)` | YES. Hub via transformers (or local folder). No `trust_remote_code`. |
| `prompt.py:249-250` | `from huggingface_hub import hf_hub_download` / `return json.load(open(hf_hub_download(path, "decider_config.json")))` (`load_decider_config`) | NO. Only used by `chat_for_model`, which `engine.py`'s `__main__` calls (`engine.py:186`). |
| `serve.py:106-107` | same `hf_hub_download` pattern in `load_config` | NO (HTTP server only). |
| `serve.py:53-54, 444, 479-558` | `from fastapi import FastAPI, HTTPException`, `app = FastAPI(title="decider", lifespan=lifespan)`, routes `POST /decide`, `POST /v1/systemone`, `GET /v1/models`, `/health`, `/stats`; docstring line 10: `uvicorn decider.serve:app --host 0.0.0.0 --port 8000` | NO. Only if someone launches the server. Listens (inbound) on whatever host the operator passes; no outbound calls; the code has no authentication and no CORS config (grep for cors/auth/api_key/middleware: no hits). |

No `requests`, `urllib`, `http.client`, `socket`, `httpx`, `aiohttp`, `websockets`; no analytics/telemetry; no hard-coded URLs (the only URLs in the repo are in README.md/ docs strings, none in code).
`transformers`/`huggingface_hub` read `HF_TOKEN` / `HF_HOME` etc. implicitly and send the token only to huggingface.co; `decider` code itself never touches tokens (no `HF_TOKEN`, `use_auth_token`, `token=` in any file).

### A2. File access outside HF cache / user-given path

All file access is **read-only**:
- `infer.py:94` `cfg = json.load(open(cfg_path))` (YES, normal path; `decider_config.json` in the model folder or HF cache)
- `prompt.py:247` `return json.load(open(f)) if os.path.exists(f) else {}` (not on normal path)
- `serve.py:105`, `serve.py:411` `json.load(open(pre))` (env `DECIDER_SCHEMAS`; server only)
- `calibrate.py:216` `with open(a.records) as f:` (CLI `python -m decider.calibrate records.jsonl`; user-given path; not on scoring path)

There is no `write`, `write_text`, `mkdir`, `makedirs`, `tempfile`, `shutil`, `os.remove/rename/replace` anywhere. No home-dir, `/tmp`, cwd, `.env` access. Indirect writes by torch only: with `torch.compile` active (see A3) Inductor/Triton write their compile cache to torch's default temp dir (`%TEMP%\torchinductor_<user>` on Windows). That is torch behaviour, not decider code.

### A3. Dynamic code, pickle, remote code, subprocess

- `exec`/Python `eval`/`compile` on strings: none. (`mps_ops.py:84,92` `mx.eval(...)` is MLX array evaluation; MPS-only.) 
- `pickle` / `torch.load` / `marshal`: none. Weights are loaded only by `from_pretrained` (`model.py:13`).  Whether the repo ships safetensors only could not be checked locally (no file listing of weights); to my knowledge (not verified here) recent transformers prefers safetensors and restricts torch loading of `.bin` files, but check the repo file list before relying on that.
- `trust_remote_code`: **not used and not needed.** `config.json` has `"architectures": ["Qwen3_5ForCausalLM"]`, `"model_type": "qwen3_5_text"`, no `auto_map`. The model class is native transformers (needs a transformers version that has Qwen3.5).
- `importlib`: single use `mps_moe.py:89` `mod = importlib.import_module(name)` with `name` taken from a hard-coded dict (`mps_moe.py:83-84`: `"transformers.integrations.moe"`, `"transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"`). MPS-only (`patch_mps`), nothing downloaded.
- `subprocess` / `os.system` / shell: none.
- Native/JIT compile: `engine.py:100` `self._fwd_impl = torch.compile(self._fwd_eager, dynamic=False)`; also `engine_v2.py:68`, `schema_engine.py:95`, `fp8.py:50`. **On the default CUDA path `Decider` builds `Engine(path, device=device, dtype=dtype)` (`infer.py:103-104`) and `Engine.__init__` has `compile=True` as default (`engine.py:81-82`: `def __init__(self, path, device="cuda", dtype=torch.bfloat16, use_graphs=True, max_ctx_tokens=1536, compile=True, fp8=False, conv_patch=True)`)**, so `torch.compile` (Inductor/Triton, needs a C/C++ compiler and Triton) + CUDA graph capture happen on the default CUDA path. Not malicious, but it runs a compiler toolchain and may fail on Windows. Avoid with `Decider(path, use_graphs=False)` (eager path, `infer.py:105-109`).
- In-process monkey-patching (all local, nothing downloaded):
  - `engine.py:43-45` `def patch_conv(): from transformers.models.qwen3_5 import modeling_qwen3_5 as mq; mq.causal_conv1d_fn = fused_causal_conv1d_fn`. Called from `Engine.__init__` when `conv_patch=True` (default) on non-MPS (`engine.py:84-89`). Replaces transformers' conv kernel with a pure-PyTorch version (numerically ~equal, not bit-identical). NOT applied on the eager `use_graphs=False` path.
  - `model.py:10-11` and `engine.py:73-74` `torch.backends.cuda.enable_cudnn_sdp(False)`: global torch setting, turns off the cuDNN SDPA backend for the whole process (comment: wrong results on Blackwell with torch 2.14).
  - `engine.py:99`, `engine_v2.py:63-67` change `torch._dynamo.config` limits.
  - `mps_ops.py:264` (`mq.torch_chunk_gated_delta_rule, mq.causal_conv1d_fn = delta, conv`) and `mps_moe.py:95` (`mod.torch = _Proxy(torch, **over)`) patch transformers modules; MPS only (`infer.py:106-108`).
  - `fp8.py:39` `setattr(mod, cname, FP8Linear(child))`: only when `fp8=True` (off by default).
  - `shared_prefix.py` `setattr` on KV-cache objects: cache manipulation only.
- MPS-only import side effect: `mps_ops.py:26-28` `import mlx.core as mx` inside try/except at import time; module is imported only when device is `mps`. A Metal shader source string (`mps_ops.py:30-57`) is compiled by MLX there. Not on CUDA/CPU paths.

### A4. Environment variables / credentials

On the normal Hub-load + score path (`Decider` -> `Engine`/`DecisionModel` -> `build`/`system_one`), the package reads **no** environment variables, except:
- `shared_prefix.py:107` (in `fork_budget_bytes`) `gb = float(os.environ.get("DECIDER_SHARED_FORK_GB", DEFAULT_FORK_GB)) if gb is None else float(gb)`: reached via `Engine.score_shared` -> `shared_prefix.score_shared` (CUDA `Engine` path, `system_one` with more than one row); confirmed by grep: `shared_prefix.py:183` `m = chunk_rows(cache_row_bytes(cache), n, budget_bytes, dev)` -> `chunk_rows` (`:122`) -> `fork_budget_bytes` (`:102`). Harmless memory-chunk size; never sent anywhere.
Optional-only:
- `mps_ops.py:230` `DECIDER_MPS_MOE_PATCH` (MPS).
- `serve.py:63-97, 354, 358, 410` (`DECIDER_MODEL`, `DECIDER_DEVICE`, `DECIDER_COMPILE`, `DECIDER_FP8`, `DECIDER_TEMPERATURE`, `DECIDER_SCHEMA_CACHE`, `DECIDER_SCHEMAS`, batching limits...): HTTP server only.
No credentials read, nothing sent anywhere.
README/code mismatch worth knowing: README says `DECIDER_TEMPERATURE=T` or `Decider(path, temperature=T)` overrides the temperature; in the code only `serve.py:354` reads `DECIDER_TEMPERATURE`. `infer.Decider` does **not** read that env var (the `temperature=` argument works).

### A5. Obfuscation
None found (see sweeps above).

### A6. Other concerns (non-security but important for evaluation fidelity)
1. **Silent config fallback** `infer.py:90-96`:
   ```python
   cfg = {}
   try:                                          # model folder may carry decider_config.json (temperature, flags)
       from huggingface_hub import hf_hub_download
       cfg_path = os.path.join(path, "decider_config.json") if os.path.isdir(path) else hf_hub_download(path, "decider_config.json")
       cfg = json.load(open(cfg_path))
   except Exception:
       pass
   ```
   If the config cannot be read (offline, rate limit, bad path) the model runs with `cfg = {}`: temperature 1.0 (not 1.145/1.164/1.624/1.124), `neutralize_none` True (config says false), `isolated_levels` False (config says true), layout plain. No warning. Load from a local snapshot folder and assert `d.T == 1.145`, `d.T_by_type == {"choice": 1.164, "noul": 1.624, "score": 1.124}`, `d.isolated_levels is True`, `d.neutralize_none is False`.
2. The `decider_config.json` keys `max_options` and `max_state_tokens` are NOT read by `infer.py` (only `layout`, `chat_template`, `temperature*`, `neutralize_none`, `version`, `schema_first`, `isolated_levels`, `schema_first_trained` (serve only), `release_date` (serve only) are read). Limits are hard-coded in code (see B2).

---------------------------------------------------------------------------------------------------

## Task B: API summary for decider

### B1. Entry points

**Importing the bundled package**: `decider/` is a plain Python package (empty `__init__.py`) inside the HF repo. It is imported by putting the **snapshot folder** on `sys.path` (e.g. `sys.path.insert(0, snapshot_dir)` where `snapshot_dir = huggingface_hub.snapshot_download("Mapika/decider-2b")`), then `from decider.infer import Decider`. README says: `from decider.infer import Decider          # decider/ is included in this repo` and `d = Decider("Mapika/decider-2b")`. **No `trust_remote_code` is involved.** (README also mentions "decider-ai 1.4.0", a pip distribution of the same package; I did not look at it and do not know whether it is published/identical. The bundled `temperature.py` is the 1.4.0 version and supports `temperature_by_type`.)

**Constructor** (`infer.py:75`):
```python
def __init__(self, path, device=None, dtype=None, temperature=None, abstain_below=0.0, use_graphs=None, temperature_by_type=None):
```
`path`: local folder or HF repo id. `device=None` -> `"cuda"` if available, else `"mps"`, else `"cpu"` (`infer.py:84-85`). `dtype=None` -> `torch.float16` on MPS else `torch.bfloat16` (`infer.py:86-87`; CPU also gets bf16). `use_graphs=None` -> True on cuda (`infer.py:100-101`).

**Which entry point the README recommends**: README "Usage" shows `d.decide(context, [{"question":..., "options":[...]}, ...])` first, then `d.system_one(state, questions)` (TypeSafe/Jev request shape, "the same request shape as TypeSafe's Jev (`POST /v1/systemone`), in process or over HTTP"), `decide_batch` for many states, `d.schema(questions)` for fixed question sets (questions-first layout, "costs accuracy" and v11 is **not** trained for it: `schema_first_trained: false`), and a plain-transformers recipe. For the typed (choice/noul/score) semantics asked for in this evaluation, **`system_one` is the matching entry point**; `decide` handles untyped option lists.

**(1) `decide` / `decide_batch`** (`infer.py:132-157`)
```python
@torch.no_grad()
def decide_batch(self, requests, max_ctx_tokens=1536):
    """requests: list of (context:str, questions:list[dict(question, options)]). One forward pass for everything."""
def decide(self, context, questions, **kw):
    return self.decide_batch([(context, questions)], **kw)[0]
```
- request = `(context: str, [ {"question": str, "options": [str, ...]}, ... ])`; 2..255 options each (`infer.py:125` `assert 2 <= len(q["options"]) <= MAX_OPTIONS, f"2..{MAX_OPTIONS} options required"`). Options must be strings (`neutralize_options` calls `.strip()`, only if `neutralize_none`; `False` for this model).
- **All questions of one request are packed into ONE row** ("Question 1 ... Answer 1: (" ... "Question 2 ..."), so later questions see earlier question texts (README: reversing order changes up to 12% of answers). For independent scoring use `system_one` (default `independent=True`) or one question per request.
- Returns `list` (per request) of `list` (per question) of dicts (`infer.py:149-152`):
  ```python
  p = probs[k, :len(q["options"])].tolist(); k += 1
  j = max(range(len(p)), key=p.__getitem__); back = q.get("_back", {})
  names = [back.get(o, o) for o in q["options"]]
  res.append(dict(choice=names[j] if p[j] >= self.abstain_below else None, confidence=p[j],
                  probs={o: pi for o, pi in zip(names, p)}, probs_list=p))
  ```
  `probs_list` is **unrounded, in the order the options were given**, already temperature-scaled. `probs` is a dict keyed by option string (duplicate option strings would collapse; use `probs_list`). `confidence` = top probability. `choice` None only if `abstain_below` > top prob (default 0.0).
- Temperature type: a plain `decide` question is typed `"choice"` (`infer.py:129` `it["types"] = [q.get("_type", "choice") for q in qs]`), so yes/no questions through `decide` get the **choice** temperature 1.164, not the noul 1.624. Put `"_type": "noul"` (or `"score"`) in the question dict to get the other temperatures (that is what `decide_json` does, `infer.py:281`).

**(2) `system_one`** (`infer.py:180`)
```python
def system_one(self, state, questions, independent=True, max_state_tokens=32768, max_fwd_tokens=65536, layout=None, isolated=None):
```
- `state`: `str`, `dict` or `list`. Strings are used verbatim; others are `json.dumps(annotate_indices(state), ensure_ascii=False)` (`systemone.py:34-37`; arrays of >= 8 elements get `{"_index": i, ...}` written into them, `systemone.py:19-31`).
- `questions`: `{question_id: spec}` (ids are never shown to the model), spec parsed by `systemone.render_question` (`systemone.py:40-74`):
  - `{"type": "choice", "instructions": <str|any JSON>, "criteria": {label: description|JSON|None, ...} | [labels]}`: 2..255 options (`MAX_CHOICE = 255`); option text shown = `label` if description is None/"" else `f"{label}: {description_as_text}"`; answer keys = labels in given order.
  - `{"type": "score", "instructions": ..., "criteria": [level0_desc, level1_desc, ...]}` (or legend dict `{"0": ..., "1": ...}` sorted by `float(key)`); 2..10 levels (`MAX_LEVELS = 10`); answer keys `"0".."K-1"`.
  - `{"type": "noul", "instructions": ... (optional), "criteria": {"true": ..., "false": ...} (optional)}`; options `["no" | "no: <false desc>", "yes" | "yes: <true desc>"]`; `"bool"` is accepted as alias. A noul without `instructions` needs at least one true/false description (then question text is `NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"`).
  - Also accepts `"options"` in place of `"criteria"` and `"question"` in place of `"instructions"` (`systemone.py:42-43`).
- Returns (`infer.py:192-193`) `{"model": "decider-2b-v11", "answers": {qid: answer}, "usage": {"input_tokens": ..., "output_tokens": 0}}`. Answer shapes (`systemone.py:176-189`, `nd=4` so **rounded to 4 decimals**):
  ```python
  if rq["type"] == "noul":
      return {"type": "noul", "noul": round(p[1], nd)}                       # p[1] = P("yes")/P(true)
  if rq["type"] == "choice":
      return {"type": "choice", "choice": rq["names"][j], "confidence": round(choice_confidence(p), nd), "x_p_max": round(p[j], nd),
              "certainty": round(certainty(p), nd), "probabilities": {n: round(x, nd) for n, x in zip(rq["names"], p)}}
  return {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 2), "confidence": round(score_confidence(p), nd), "x_p_max": round(p[j], nd),
          "certainty": round(certainty(p), nd),
          "legend": {str(i): d for i, d in enumerate(rq["legend"])}, "probabilities": {str(i): round(x, nd) for i, x in enumerate(p)}}
  ```
  Note `confidence` here is TypeSafe's `(n*p_max-1)/(n-1)`, NOT the top probability; the calibrated top probability is `x_p_max`. Score answers (isolated levels) additionally carry `level_fit` and `fit_mass` (`systemone.py:132-133`). `probabilities` keys: choice = the criteria keys in the order given (`names = list(crit)`; if `criteria` was a list, the keys are `str(c)`, `systemone.py:55-58`), score = `"0".."K-1"`, noul has no `probabilities` field (only `noul` = P(yes)). Option order never changes (no shuffling).
- For **unrounded** probabilities use the lower-level pair used by `calibrate.collect` (`calibrate.py:189-190`): `rqs, index, items = d._system_one_items(state, questions, independent, layout="state_first")` then `rows = d._system_one_probs(items, "state_first", temperature=1.0)` (this reads at T=1; pass the per-item temperature from `TT.for_items(d.T, d.T_by_type, items)` to get the calibrated values), or use `decide`/`decide_batch` (`probs_list`). These are private methods; verify the signature if the package is updated.
- Isolated Score levels (config `isolated_levels: true`, so default for this model): each Score level becomes its own yes/no row (`systemone.py:84-96, 106-115`):
  ```python
  ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"
  def strip_level_number(text): ... re.sub(r"^\s*-?\d+\s*:\s*", "", text)
  def isolated_rows(question, levels):
      return [(ISOLATED.format(q=question, level=strip_level_number(l)), ["no", "yes"]) for l in levels]
  def combine_isolated(p_yes):
      tot = sum(p_yes) or 1e-9
      return [x / tot for x in p_yes], tot
  ```
  The score distribution = P(yes) of each level row normalised to sum 1. `isolated=False` restores listwise scoring (all levels as options `"0: desc"`, `"1: desc"` ...), but the model was trained with isolated levels.

**(3) `decide_json` / `decide_json_batch`** (`infer.py:293-318`): schema form `{question_text: {"type": "bool"} | {"type": "choice", "options": [...]} | {"type": "scale", "legend": [...] or {num: desc}}}` -> routes through `decide_batch` with `_type` set; all questions packed in one row. Not needed for adapters.

**(4) `schema()`** (`infer.py:161`): compiled fixed question set, questions-first layout; not trained for v11 (README: "`Decider.schema()` still runs; its accuracy was not measured on v11"). Do not use.

### B2. Truncation

- **Only the context/state is ever truncated, silently, keeping the head**; there is no option to raise instead of truncating, no warning, no flag in the output. Question text and option text are never truncated (and not counted against the cap).
- `decide_batch(max_ctx_tokens=1536)` default (`infer.py:133`): `build(..., max_ctx_tokens=max_ctx_tokens)` -> `prompt.py:128` `ctx_ids = tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]` (so the cap also counts the few tokens of the `Context:` prefix).
- `system_one(max_state_tokens=32768)` default (`infer.py:180`), passed as `max_ctx_tokens=max_state_tokens` to `build` (`infer.py:224-225`): same slice, 32,768 tokens. (`decider_config.json` has `"max_state_tokens": 32768` but the library does not read it. The README says "up to 32k tokens with the questions", but the cap applies to the state tokens only.)
- Raise the cap by passing a larger `max_ctx_tokens=` / `max_state_tokens=`. The model's `max_position_embeddings` is 262144 (`config.json`).
- For an error instead of silent truncation, check in the adapter: `len(tok.encode("Context:\n" + state_text, add_special_tokens=False)) <= cap` (state_text must be exactly what `render_state` produces for dict/list states), and raise yourself.
- Server-only (not library): `serve.py:168` `check_size` returns HTTP 413 when rows / per-row tokens / request tokens exceed `DECIDER_MAX_ROWS=1024`, `DECIDER_MAX_ROW_TOKENS=MAX_STATE_TOKENS+4096`, `DECIDER_MAX_REQUEST_TOKENS=1<<20`; states are still cut at `DECIDER_MAX_STATE_TOKENS` (default 32768) before that check.
- **Max number of options**: 255 for choice and `decide` (`prompt.py:12` `MAX_OPTIONS = 255`; `systemone.py:12` `MAX_CHOICE, MAX_LEVELS = 255, 10`); minimum 2. Score: 2..10 levels. noul: fixed 2. More than 10 options switches to the "wide" rendering (one label token per option).  README: large label sets cost accuracy (training sub-sampled to <= 10 options).  `_select`/sub-sampling branch in `build` is unreachable from `Decider` (`max_options=MAX_OPTIONS`, and `_NoShuffle.sample` returns `xs[:k]`).

### B3. Calibration / temperature

- Source: `decider_config.json` of the model folder/repo: `"temperature": 1.145`, `"temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124}` (also in README and `eval_results.json`). Loaded at `infer.py:91-98`: 
  `(temperature, self.T_by_type), (T_schema, self.T_schema_by_type) = TT.from_config(cfg, temperature, temperature_by_type)`.
- Application: logits at the answer slot restricted to the option-letter tokens are divided by T then softmaxed: `temperature.py:136-144` `scaled_softmax(lg, temperature)`: `torch.softmax(lg / temperature, -1)` (per-row temperature list when a by-type map exists; `for_items`, `temperature.py:104-109`: `[[m.get(t, T) for t in item_types(it)] for it in items]`). A missing type falls back to `"temperature"` (1.145).
- Which type each row gets: `decide` -> "choice" unless `_type` given; `system_one` -> the question's type; isolated Score level rows (yes/no rows) -> "score" (1.124) (`systemone.py:118-124` `row_types`).
- **To keep the shipped temperatures (no re-fit)**: construct with `Decider(path)` and do NOT pass `temperature=` or `temperature_by_type=` (passing `temperature=` switches the by-type map off, `temperature.py:71-73`: an explicit temperature without an explicit map gives `m = {}`). Then check `d.T == 1.145` and `d.T_by_type == {"choice": 1.164, "noul": 1.624, "score": 1.124}` after construction (guard against the silent config fallback in A6.1). Nothing in `infer.py`/`systemone.py` fits temperatures; `decider.calibrate` (`fit_by_type`, `main`) is a separate offline tool and is never called by `Decider`.
- For raw (T=1) probabilities: `Decider(path, temperature=1.0)`.
- Packages older than the 1.4.0 `temperature.py` (not this bundle) ignore the by-type map and use 1.145 everywhere (README).

### B4. Device / dtype / batching / randomness

- Device: `Decider(path, device="cuda")` (or `"cuda:1"`, `"cpu"`, `"mps"`); default auto (`infer.py:84-85`).
- dtype: default `torch.bfloat16` (also on CPU), `torch.float16` on MPS (`infer.py:86-87`). Override with `dtype=torch.float16/float32`. The model weights are loaded in that dtype (`model.py:13`); readout logits are cast to float32 (`model.py:24` `.float()`, `engine.py:109` `.float()`).
- Default CUDA path = `Engine` with shape-bucketed CUDA graphs + `torch.compile` (`engine.py:81-82,100`), `conv_patch=True` (`patch_conv`). `Decider(path, use_graphs=False)` gives the plain eager path: `self.m = DecisionModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()` (`infer.py:109`), no compile, no conv patch (the stock transformers kernels / `flash-linear-attention` are used). README: `flash-linear-attention` (Triton) strongly recommended, model "runs without it but several times slower"; on Windows Triton/torch.compile availability is uncertain, so for this machine use `device="cuda", use_graphs=False` first.
- Batch handling:
  - `decide_batch(requests)`: everything in ONE forward pass, no internal chunking. Eager path: one `collate` of all rows (`infer.py:140-143`), right-padded to a multiple of 64 (`model.py:36`), so memory scales with `len(requests)`. Engine path: padded to bucket `B_BUCKETS = [1, 2, 4, 8, 16, 32, 64]` and `T_BUCKETS` up to 2048 (`engine.py:12-14`, `engine.py:148-150`): `B = (_bucket(len(items), B_BUCKETS) or len(items)) if T <= GRAPH_MAX_T else len(items)`, i.e. more than 64 rows captures a fresh CUDA graph for that exact B; rows longer than 2048 tokens run eager. Feed it chunks (e.g. <= 32 requests).
  - `system_one`: `independent=True` -> one row per question (per Score level when isolated); on the Engine path with >1 row and state-first layout it uses `score_shared` (state run once, cache forked per question; `DECIDER_SHARED_FORK_GB` chunking, `infer.py:199-200`); otherwise rows are chunked by `per = max(1, max_fwd_tokens // max(len(it["ids"]) for it in items))` (`infer.py:202`; `max_fwd_tokens=65536`).
- Randomness: none. `.eval()`, `@torch.no_grad()`, `attention_dropout: 0.0` (`config.json`), no sampling; option order is preserved (`_NoShuffle`, `infer.py:47-50`: `def shuffle(self, x): pass`). Residual run-to-run/batch-to-batch differences only from kernel round-off (bf16, padding bucket, graph vs eager, `score_shared` vs full forward; README: "Same answers as score_items up to kernel round-off").

### B5. Prompt construction and which logits are read (verbatim)

**Default model config** (`decider_config.json`): `"layout": "plain"` (so the chat template / `chat_template.jinja` is NOT used), `"schema_first": false`, `"isolated_levels": true`, `"neutralize_none": false`, `"max_options": 255`, `"max_state_tokens": 32768`. Layout resolution: `prompt.py:175-188` `resolve_layout`.

Constants (`prompt.py:10-14`):
```python
LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)      # <= NARROW options: the original "(A) .. (J)" rendering, tokenized as a string (unchanged since v1)
MAX_OPTIONS = 255          # width of the label head. > NARROW options: "wide" rendering, one label token per option:
                           # A..Z then the first 229 two-letter upper-case strings that are single tokens (AA, AB, ...)
ABSTAIN_PREFIXES = ("none of the above", "none of these", "not listed", "no suitable", "does not apply", "cannot tell")
ABSTAIN_EXACT = ("other", "unsure", "something else", "neither of these", "other / not covered")
```

`build` for the plain, state-first layout (`prompt.py:119-157`), used by both `decide_batch` and `system_one`:
```python
def build(example, tok, rng=None, max_options=NARROW, max_ctx_tokens=1536, layout="state_first", chat=None):
    if layout == "schema_first":
        return build_schema_first(example, tok, rng, max_options, max_ctx_tokens, chat=chat)
    if chat is not None:
        return build_chat(example, tok, chat, rng, max_options, max_ctx_tokens)
    rng = rng or random
    ctx_ids = tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    ids = list(ctx_ids)
    slots, golds, nopts, perms = [], [], [], []
    multi = len(example.qs) > 1
    for k, q in enumerate(example.qs):
        opts = list(range(len(q.options)))
        if len(opts) > max_options:
            ... (sub-sampling; unreachable from Decider)
        rng.shuffle(opts)
        head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"
        tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("
        if len(opts) <= NARROW:
            lines = [head] + [f"\n({LETTERS[j]}) {q.options[oi]}" for j, oi in enumerate(opts)] + [tail]
            piece = tok.encode("".join(lines), add_special_tokens=False)
        else:                                   # wide: "\n(" + <label token> + ") text", built from ids so every label is one token
            _, lab_ids, open_ids = label_table(tok)
            piece = tok.encode(head, add_special_tokens=False)
            for j, oi in enumerate(opts):
                piece += open_ids + [lab_ids[j]] + _enc_opt(tok, q.options[oi])
            piece += tok.encode(tail, add_special_tokens=False)
        ids.extend(piece)
        slots.append(len(ids) - 1)          # position of " (" token
        golds.append(opts.index(q.gold) if q.gold in opts else -1)
        nopts.append(len(opts))
        perms.append(opts)
    return dict(ids=ids, slots=slots, golds=golds, nopts=nopts, perms=perms)
```
Important for exact replication: the prompt is **token-id concatenation of separately encoded pieces** (context piece, then one piece per question block), not one joint tokenization of the whole string; no BOS/special tokens are added (`add_special_tokens=False`).

Wide rendering helpers (`prompt.py:27-35, 38-55, 69-75`):
```python
def _enc_opt(tok, text):
    """Token ids of ") <option text>" (cached: ...)."""
    ...
        v = tok.encode(f") {text}", add_special_tokens=False)
def label_table(tok):
    """(label strings, label token ids), MAX_OPTIONS entries; the first NARROW are A..J so narrow questions are unchanged."""
    ...
        names = list(U) + [a + b for a in U for b in U]
        out = []
        for n in names:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                out.append((n, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        assert len(out) == MAX_OPTIONS and len({i for _, i in out}) == MAX_OPTIONS
        _LABELS[key] = ([n for n, _ in out], [i for _, i in out], tok.encode("\n(", add_special_tokens=False))
def _options_ids(tok, q, opts):
    if len(opts) <= NARROW:
        return tok.encode("".join(f"\n({LETTERS[j]}) {q.options[oi]}" for j, oi in enumerate(opts)), add_special_tokens=False)
    _, lab_ids, open_ids = label_table(tok); out = []
    for j, oi in enumerate(opts):
        out += open_ids + [lab_ids[j]] + _enc_opt(tok, q.options[oi])
    return out
```

Rendered example (single question; this is exactly the README's plain-transformers prompt):
```
Context:
My card was charged twice for the same purchase.

Question: Which department should handle this?
Options:
(A) billing
(B) technical support
(C) sales
Answer: (
```
Two questions in one row (`decide` with 2 questions): `Context:\n...` then `\n\nQuestion 1: ...\nOptions:\n(A) ...\n(B) ...\nAnswer 1: (` then `\n\nQuestion 2: ...\nOptions:\n(A) ...\nAnswer 2: (`; no answer letters are inserted, each `(` is a slot.

`system_one` rows (`infer.py:215-229`): `ctx = render_state(state)`; each question becomes its own `Example(ctx, [Q(r["question"], opts(r), 0)])` -> single-question prompt as above with `Question: <instructions>`; options = `label` or `label: description` (choice), `no`/`no: ...`,`yes`/`yes: ...` (noul), and for Score the isolated yes/no rows (`Question: <instructions>\nProposed answer: <level text without leading "N:">\nDoes the proposed answer fit?` with options `(A) no`, `(B) yes`). `build(..., layout=layout)` with layout `"state_first"` (since `schema_first` is False for v11; `infer.py:181-182`).

**Which logits are read** (`model.py:18-27`):
```python
def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
    h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    hs = h[slot_batch, slot_idx]                                   # [N,H]
    W = self.lm.lm_head.weight[self.letters]                       # [K,H]
    logits = F.linear(hs, W).float()                               # [N,K]
    ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
    logits = logits.masked_fill(ar >= nopts[:, None], float("-inf"))
    return logits
```
i.e. at the position of the final `(` token of each `Answer k: (` slot, the (final-normed) hidden state is projected onto the **LM-head rows of the option-letter tokens only** (`letters` = ids of `A..J` for <= 10 options; `A..Z, AA, AB, ...` single-token strings for up to 255), the first `nopts` letters are kept (others `-inf`), divided by the temperature and softmaxed. Letter token ids come from `letter_ids(tok)` (`prompt.py:278-282`): `tok.encode(L, add_special_tokens=False) == [ids[j]]` asserted for `A..J` (no leading space: the letter follows `(`). Engine path computes the same thing via `F.linear(h, self.W)` with `self.W = self.m.lm.lm_head.weight[self.m.letters].detach().clone()` (`engine.py:93,109`) and `read_slots` (`engine.py:48-56`). README's plain-transformers equivalent: `logits[0, -1]` restricted to `[tok.encode(L, add_special_tokens=False)[0] for L in "ABC"]`, then `softmax(logits[letters].float() / 1.164)`.

### B6. Dependencies that could conflict

- **transformers >= 5** (README: "`torch`, `transformers>=5`, and `flash-linear-attention`"; `config.json` `"transformers_version": "5.17.0"`, `model_type: qwen3_5_text`; `mps_ops.py:236-243` explicitly requires >= 5.17 for the MPS patch). Code specifics: `model.py:13` uses the `dtype=` keyword of `from_pretrained` (new name for `torch_dtype`); `engine.py:44` imports `transformers.models.qwen3_5.modeling_qwen3_5` for `patch_conv` (breaks on a transformers without Qwen3.5 support; not used if `use_graphs=False`). A pin of transformers 4.x elsewhere in the same environment would break this model.
- **torch**: no pin in the bundle; README "Python 3.11 or newer lets those kernels use torch.compile". Needs a torch recent enough for the transformers version (I recall transformers 5.x needing torch >= 2.4, from a README I read outside this bundle; unverified here, check the installed transformers' metadata). `torch.backends.cuda.enable_cudnn_sdp` is guarded by `hasattr`. `torch._scaled_mm` only for FP8 (off).
- **flash-linear-attention** (Triton kernels for Qwen3.5 linear-attention layers): optional but strongly recommended, otherwise much slower; Triton on Windows is not an official torch dependency (uncertain on this machine). Optional: `causal-conv1d` (picked up by transformers if present), `mlx` (MPS only).
- `huggingface_hub` (for `hf_hub_download` when `path` is a repo id), `numpy` (calibrate/metrics/engine helpers).
- `fastapi`, `pydantic`, `uvicorn` only for `decider.serve`.
- Python >= 3.10 required by modern transformers/torch; code uses `str | None` in `serve.py` (3.10+).

---------------------------------------------------------------------------------------------------

## Recommended safe usage for evaluation adapters (derived from the above)

1. `snapshot_download("Mapika/decider-2b", revision=<pinned sha>)` -> local folder; `sys.path.insert(0, folder)`; `from decider.infer import Decider`.
2. `d = Decider(folder, device="cuda", dtype=torch.bfloat16, use_graphs=False)` (eager; avoids `torch.compile`/Triton/CUDA-graph and the `patch_conv` transformers monkey-patch; note the global `torch.backends.cuda.enable_cudnn_sdp(False)` switch in `DecisionModel.__init__`, `model.py:10-11`, still happens on every path). Do not pass `temperature`/`temperature_by_type`. Assert `d.T == 1.145`, `d.T_by_type`, `d.layout == "plain"`, `d.isolated_levels`, `not d.neutralize_none`.
3. Typed questions: `d.system_one(state, {"q": {"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}}, max_state_tokens=<explicit>)`; read `answers[qid]["probabilities"]` (choice/score) or `["noul"]`; use `x_p_max` for the calibrated top probability. For unrounded values use the private `_system_one_items` / `_system_one_probs` pair.
4. Do your own pre-check of state length (silent head truncation at `max_state_tokens` / `max_ctx_tokens`), and feed `decide_batch` in chunks.

## Not covered
`quyet` 1.0.2 and `laya` 0.4.6 were not reviewed for this document (coordinator scope change). Before the change, a partial look at their wheels found only `.py` files + dist-info (no `.pth`, native libraries, or setup hooks) and RECORD hashes matching the files on disk, but that is NOT a code review or clearance.
