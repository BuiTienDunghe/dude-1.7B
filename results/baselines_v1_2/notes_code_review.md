# Code review (bảo mật / chạy offline): gói pip `quyet` 1.0.2 và `laya` 0.4.6, và model Jeff-Qwen3.5-2B

Mục tiêu: xác nhận các gói chạy hoàn toàn cục bộ sau khi file model đã nằm trong HF cache (benchmark có dữ liệu chỉ được dùng nội bộ).
Chế độ: CHỈ ĐỌC. Không cài, không import, không chạy, không tải, không dùng mạng. Không in dữ liệu benchmark.

Nguồn đã đọc (đường dẫn gốc `C:\Users\dungbt06\Fine-tune\results\baselines\`):
- `pkgsrc\quyet-1.0.2\quyet\` (16 file .py, 783 dòng) + `quyet-1.0.2.dist-info\` (METADATA, RECORD, entry_points, NOTICE).
- `pkgsrc\laya-0.4.6\laya\` (44 file .py, 23.060 dòng) + `laya-0.4.6.dist-info\` (METADATA = README PyPI, RECORD).
- `meta\<org>__<model>\` (README.md, config.json, quyet_config.json, rl_agent_config.json, decision_config.json...), `pins_models.json` (sha + danh sách file HF).
- Số dòng `file:dòng` dưới đây là số dòng trong file local (đã kiểm bằng `cat -n`).
- Mẫu định dạng: `notes_code_review.md` (decider) ở thư mục `baselines\`.

## 0. Kết luận ngắn

| Gói | Có gửi dữ liệu ra ngoài máy (ngoài tải model từ HF Hub)? | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` đủ? | Cờ/knob bắt buộc để giữ offline |
|---|---|---|---|
| `quyet` 1.0.2 | KHÔNG. Chỉ có đúng 1 chỗ chạm mạng: `huggingface_hub.snapshot_download` (`hub.py:15-16`), và chỉ khi tham số không phải thư mục có sẵn. | ĐỦ. | Không có. Khuyến nghị thêm: truyền đường dẫn snapshot local thay vì repo id. |
| `laya` 0.4.6 (đường `laya.load(...)` + `agent.predict(...)`, mặc định) | KHÔNG. Chỉ `snapshot_download` (`agent.py:581-595`) + 2 nhánh fallback tải repo phụ `jhu-clsp/mmBERT-base` (`agent.py:236-237`, `common.py:706`) mà 2 repo đang đánh giá KHÔNG chạm tới (đủ thư mục `tokenizer/` và `encoder/`). | ĐỦ, với điều kiện snapshot đầy đủ và KHÔNG dùng `Router`, `serve`, `mcp`, `integrations`, `fast=True`, `compile=True`, `backend="auto"`, không đặt hook. | Xem mục 2.6 (danh sách việc phải làm / phải tránh). |
| Jeff-Qwen3.5-2B | KHÔNG KẾT LUẬN ĐƯỢC. Phần chạy model (prompt, `readout.safetensors`, server `jeff-serve`) nằm ngoài HF repo, trong repo GitHub `firelex/jeff` mà không có local. | Không đánh giá được. | Xem mục 3. |

Cờ đỏ bảo mật: KHÔNG có cờ đỏ trong `quyet` và `laya` ở đường chạy mặc định (không telemetry, không exec/eval chuỗi, không pickle/`torch.load`, không subprocess, không `trust_remote_code`, không obfuscation). Có 3 hành vi cần biết (không độc hại): (1) `laya` ghi đè file tokenizer_config.json trong snapshot cache khi gặp điều kiện nhất định (mục 2.3); (2) `laya` có `PINNED_REVISIONS` khác sha team đang pin (mục 2.7); (3) Jeff không review được (mục 3).
Ngoài phạm vi (không phải lỗi gói): các gói có tính năng gửi dữ liệu tới URL do người dùng nhập (`laya` integrations/MCP remote) và server `laya-serve` nghe `0.0.0.0` mặc định không khóa. Chỉ xảy ra khi người dùng chủ động gọi; mục 2.2.

Định dạng weight: theo `pins_models.json`, mọi repo (Quyet Tiny/Small/Medium, laya-multilingual, laya-vi, Jeff) chỉ có `*.safetensors` + JSON/tokenizer; không có `.bin`/`.pt`/`.pkl`/`.py` trong repo HF (Jeff có thêm `readout.safetensors`). Không file nào trong meta có `auto_map` hay `trust_remote_code` (đã `grep` trên toàn bộ `meta\`, trừ README).

---------------------------------------------------------------------------------------------------

## 1. quyet 1.0.2 (đọc TOÀN BỘ 16 file)

File: `__init__`, `__main__`, `errors`, `base`, `hub`, `calibration`, `answers`, `cli`, `questions`, `encoder/{__init__,inputs,modeling,runtime}`, `llm/{__init__,prompt,runtime}`. Tổng 783 dòng, tất cả đã đọc từng dòng. RECORD liệt kê đúng 16 .py + dist-info; wheel `py3-none-any`, `Root-Is-Purelib: true`; không `.pth`, không thư viện native, entry point duy nhất `quyet = quyet.cli:main`.

Sweep regex toàn gói (`environ|getenv|requests|urllib|http|socket|subprocess|os.system|pickle|torch.load|marshal|base64|zlib|trust_remote|exec(|eval(|compile(|open(|write|mkdir|tempfile|shutil|importlib|__import__|atexit|threading|setattr|torch.backends|torch._dynamo`): không có match nào ngoài các dòng liệt kê dưới đây.

| # | Hạng mục | Vị trí và đoạn code | Có chạy trên đường chấm điểm local? |
|---|---|---|---|
| Q1 | Mạng | `hub.py:15-16` `from huggingface_hub import snapshot_download` / `return Path(snapshot_download(str(name_or_path), revision=revision))` | CÓ nếu `name_or_path` không phải thư mục tồn tại (`hub.py:12-14` `p.is_dir()` thì trả luôn `p`). Chỉ HF Hub; không `allow_patterns` nên tải cả repo (README Medium: 9,3 GB). |
| Q2 | Mạng khác | Không có `requests/urllib/http/socket`. Không URL cứng trong code (URL chỉ có trong METADATA/README). | - |
| Q3 | Tải model qua transformers | `encoder/runtime.py:29` `AutoTokenizer.from_pretrained(str(self.path))`; `llm/runtime.py:21,25,27` `AutoTokenizer/AutoModelForCausalLM/AutoModelForImageTextToText.from_pretrained(str(self.path), **kw)`; `encoder/modeling.py:122` `AutoModel.from_config(AutoConfig.for_model(**config["encoder_config"]), ...)` | CÓ. Luôn truyền ĐƯỜNG DẪN LOCAL (đã resolve ở `hub.resolve`) nên transformers không gọi Hub. `from_config` dựng kiến trúc từ JSON, không tải gì, không remote code. |
| Q4 | Nạp weight | `encoder/runtime.py:19,27` `safetensors.torch.load_file(...)`, `load_state_dict(..., strict=True)` | CÓ (encoder). safetensors, không pickle. LLM nạp qua `from_pretrained` (safetensors theo `pins_models.json`). |
| Q5 | Biến môi trường | Không có (`grep environ|getenv` = 0 match). | - |
| Q6 | Ghi file | Không có. Duy nhất `cli.py:23` `open(a.input, encoding="utf8")` (đọc input của CLI, không chạy khi gọi `quyet.load`). | - |
| Q7 | subprocess / exec / eval / compile | Không có. Không `torch.compile`, không CUDA graph, không patch kernel. | - |
| Q8 | pickle/marshal/torch.load/base64/zlib | Không có. | - |
| Q9 | trust_remote_code | Không có. `quyet_config.json` không có `auto_map` (Tiny/Small/Medium). | - |
| Q10 | Import-time side effect | `__init__.py:1-6` chỉ import `errors` và `hub` (hub chỉ import json/pathlib). Không import torch/transformers lúc `import quyet`; import trễ trong hàm. `__main__.py:5` `sys.exit(main())` chỉ khi chạy `python -m quyet`. | - |
| Q11 | Monkey-patch | KHÔNG có patch torch/transformers toàn cục. Chỉ dùng `attn_implementation="sdpa"` (`encoder/modeling.py:122`). | - |
| Q12 | Chat template | `llm/prompt.py:71` `tok.apply_chat_template(..., enable_thinking=False)` thực thi `chat_template.jinja` của repo trong Jinja sandbox của transformers. Đã diff `chinhnc__Quyet-1.0-Medium\chat_template.jinja` với bản của `Qwen/Qwen3.5-2B`: chỉ khác khối `{%- if add_generation_prompt %}` (đảo mặc định `enable_thinking`). Không có code lạ. | CÓ (chỉ Medium). |
| Q13 | Chuỗi đặc biệt trong text | `encoder/inputs.py:37,44-47` `Tokens.encode` dùng `split_special_tokens=True` (docstring: chuỗi special-token viết literal trong text chỉ là text) rồi ném `QuestionError` nếu `input_ids` vẫn chứa id nằm trong `tok.all_special_ids`. Không phải vấn đề bảo mật; là lưới an toàn có thể ném lỗi cho cả lời gọi nếu tokenizer sinh `<unk>`/special cho ký tự lạ. Adapter nên bắt `QuestionError` và ghi log theo từng item, không bỏ qua im lặng. | CÓ (encoder). |

Kết luận quyet: sạch. Offline: `HF_HUB_OFFLINE=1` là đủ; không cần knob nào. Nếu truyền thư mục snapshot thì không có lời gọi Hub nào.
Lưu ý (không phải bảo mật): `hub.py:12` coi `name_or_path` là thư mục nếu tồn tại thư mục tương đối trùng chuỗi repo id trong CWD (ví dụ `chinhnc/Quyet-1.0-Small`). Luôn truyền đường dẫn tuyệt đối tới snapshot để tránh nhầm.

---------------------------------------------------------------------------------------------------

## 2. laya 0.4.6

### 2.1. Phạm vi đọc

Điểm vào mà README dùng: `import laya; agent = laya.load("<repo_id>"); agent.predict(state, questions)` (README `laya-multilingual`, README `laya-vi`). `laya.load` = `agent.py:2088-2155`, `Agent.predict = system_one` (`agent.py:1981`) -> `predict_batch` (`agent.py:1409`).

Module ĐỌC TỪNG DÒNG (đường import + gọi mặc định):
- `__init__.py` (167 dòng), `agent.py` (2155; đọc toàn bộ), `common.py` (956), `calibrate.py` (727), `revisions.py` (145), `_compile.py` (107), `hooks.py` (547), `backends/__init__.py` (114), `backends/eager.py` (28), `backends/compile.py` (phần 30-138), `backends/tilelang.py` (45).
Module ĐỌC MỨC MODULE-LEVEL + QUÉT REGEX (đều là import-time của `import laya` nhưng không có hàm nào được gọi khi `load`+`predict` mặc định): `router.py` (đọc docstring, hằng số, `canonical_name`, `normalise_name`, `resolve_model_spec`, các dòng có `os.`/`open`), `structured.py`, `lang.py`, `email.py`, `presets.py`, `confidence.py` (đọc đầu file + `apply_confidence_gate`; các hàm gate `flag_low_confidence`, `check_min_confidence` chỉ quét), `backends/base.py`.
Chỉ QUÉT REGEX (không nằm trên đường mặc định, không đọc từng dòng): `serve.py`, `cli.py`, `mcp/*`, `integrations/*`, `train.py`, `train_cli.py`, `evals*.py`, `_agreement.py`, `_eval_policy.py`, `evidence.py`, `shortlist.py`, `onnx_agent.py`, `fast.py`, `tl_kernels.py`. Các module này chỉ được import khi người dùng chủ động (`_LAZY_ATTRS` ở `__init__.py:43-81`, entry point CLI, hoặc `backend=`/`fast=`). `apply_confidence_gate` thoát ngay khi `min_confidence is None` (`confidence.py:282`), `Agent.__init__` không cài backend nào nếu không truyền `backend=`/`fast=` (`agent.py:811-814`).

Sweep toàn 44 file (kết quả đã đối chiếu thủ công, bỏ match nhiễu như `empty`, `re.compile`, `.eval()`):
- mạng: `urllib` chỉ trong `integrations/{crewai,langchain,llamaindex}.py` và `mcp/remote.py`; `socket` chỉ `serve.py:1483-1495`. Không `requests/httpx/aiohttp/websocket`. Không `telemetry|analytics|posthog|sentry|mixpanel|wandb|segment` (chỉ có chữ "telemetry" trong comment `agent.py:367`, `hooks.py:523`).
- `"://"` trong code: chỉ chuỗi cảnh báo `agent.py:824-825` (link pytorch.org trong thông báo), docstring `mcp/remote.py:65`, `mcp/server.py:135,141-142` (chuẩn hóa `LAYA_BASE_URL`), comment `shortlist.py:16`.
- subprocess / os.system / ctypes: không có. exec / eval chuỗi / `__import__` động: không có, ngoại trừ `mcp/tools.py:910,947` `__import__("laya")` / `__import__(pkg)` với pkg cố định `("laya","transformers")` để đọc `__version__` (không nằm trên đường chạy).
- pickle / marshal / `torch.load` / `np.load` / joblib: không có. Weight chỉ qua `safetensors.torch.load_file` (`agent.py:571,681`, `train.py:575`).
- base64 / zlib / gzip / hex / obfuscation: không có.
- `trust_remote_code` / `auto_map` / `remote_code`: không có.

### 2.2. Mạng (bảng)

| # | Vị trí / đoạn code | Có chạm trên đường `load`+`predict` mặc định? |
|---|---|---|
| L1 | `agent.py:581,595` `from huggingface_hub import snapshot_download` / `model_dir = snapshot_download(model_id_or_path, **kw)`; `kw` (587-592) gồm `token` = `token or os.environ.get("HF_TOKEN")`, `allow_patterns` = `rl_agent_config.json, model.safetensors, tokenizer/*, encoder/*`, và `revision` nếu có. | CÓ nếu `model_id_or_path` không phải đường dẫn tồn tại (`agent.py:575`). Chỉ HF Hub, chỉ 4 nhóm file trên. Token HF chỉ đi tới HF. |
| L2 | `agent.py:236-237` `if not os.path.isdir(tok_dir): return AutoTokenizer.from_pretrained(cfg.get("encoder"))` (`encoder` = `jhu-clsp/mmBERT-base` trong `rl_agent_config.json`) | KHÔNG với 2 repo này: `pins_models.json` cho thấy cả laya-multilingual và laya-vi có `tokenizer/tokenizer.json` + `tokenizer/tokenizer_config.json`. NHƯNG nếu snapshot thiếu `tokenizer/` thì code tải THÊM repo `jhu-clsp/mmBERT-base` (repo ngoài model). Adapter phải assert thư mục tồn tại. |
| L3 | `common.py:705-706` `if not pretrained or (encoder_dir and os.path.exists(encoder_dir)): ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])` | `Agent` gọi `build_model(..., encoder_dir=enc_dir if os.path.exists(enc_dir) else None, pretrained=False)` (`agent.py:677-678`). Cả 2 repo có `encoder/config.json`, nên dùng local. Nếu thiếu `encoder/` thì tải config từ `jhu-clsp/mmBERT-base` (chỉ config, không weight). Cũng cần assert. |
| L4 | `common.py:715` `enc = AutoModel.from_pretrained(cfg["encoder"], **kw)` (tải weight encoder gốc từ Hub) | KHÔNG: chỉ khi `pretrained=True` (huấn luyện); `Agent` luôn truyền `pretrained=False`. |
| L5 | `train.py:556-564`, `onnx_agent.py:113-125` `snapshot_download` | KHÔNG (train / ONNX). |
| L6 | `integrations/crewai.py:186`, `langchain.py:197`, `llamaindex.py:182` `opener.open(req, ...)`; `mcp/remote.py:96` `urllib.request.urlopen(req, ...)` | KHÔNG. POST `state` + `questions` tới `base_url` do người dùng truyền (laya-serve ở xa). Nếu có ai truyền `base_url` thì dữ liệu RỜI MÁY. Không dùng các lớp này. |
| L7 | `serve.py:1499-1505` `uvicorn.run(create_app(), host=os.environ.get("LAYA_HOST", "0.0.0.0"), ...)`; không có API key mặc định (`serve.py:1070` `LAYA_API_KEY` tùy chọn) | KHÔNG (chỉ khi chạy `laya-serve`). Nghe cổng vào trên mọi interface; không gọi ra ngoài. |

Router (README khuyến nghị `Router()`): `router.py:51-63` `DEFAULT_MODELS` / `STANDALONE_MODELS` gồm 3 checkpoint; `Router()` mặc định giữ `english` + `multilingual` (README laya-multilingual: "The default `Router()` keeps both `english` and this checkpoint resident"), tức là tự tải THÊM repo `convaiinnovations/laya` (hoặc subfolder của nó) ngoài repo model đang đánh giá. Adapter KHÔNG được dùng `Router`; chỉ dùng `laya.load(<snapshot_dir>)` -> `Agent`.
Cẩn thận alias: `laya.load("laya-multilingual")` (không có dấu `/`) được resolve thành repo bundle `convaiinnovations/laya` + `subfolder="multilingual"` (`router.py:80-84`, `agent.py:2131-2136`) = repo KHÁC. Với repo id đầy đủ (`convaiinnovations/laya-multilingual`) thì `_is_local_checkpoint_arg` trả True vì có `/` (`agent.py:2075-2078`), nên không qua alias. Dùng repo id đầy đủ hoặc đường dẫn local.

### 2.3. Ghi file

| # | Vị trí | Mô tả | Chạm trên đường mặc định? |
|---|---|---|---|
| L8 | `agent.py:61-113` `_fix_tokenizer_config(path)`, gọi ở `agent.py:609`; ghi ở 89-100 (`mkstemp(dir=cfg_dir, ...)`, `json.dump`, `os.chmod`, `os.replace(tmp_file, cfg_file)`) | Mở `<model_dir>/tokenizer/tokenizer_config.json`; NẾU `tokenizer_class` là `None`/`"TokenizersBackend"` hoặc `extra_special_tokens` là list thì sửa và GHI ĐÈ file bằng file thường (thay symlink trong `snapshots/<sha>/`). Lỗi ghi chỉ thành `RuntimeWarning` (107-113). Có điều kiện: không xem được nội dung `tokenizer_config.json` (502 và 624 byte) nên không biết có kích hoạt hay không. | CÓ ĐIỀU KIỆN. Không phải mạng, lành tính, nhưng nó SỬA snapshot đã pin (blob gốc giữ nguyên, entry trong `snapshots/<sha>/` đổi thành file thường). Ảnh hưởng mọi bước kiểm sha sau đó nếu hash theo đường dẫn snapshot. Cách tránh: copy snapshot sang thư mục tạm có quyền ghi (dereference symlink) rồi load từ đó, hoặc kiểm tra điều kiện trước, hoặc `chmod` read-only để nó chỉ cảnh báo. |
| L9 | `_compile.py:53` `os.makedirs(directory, exist_ok=True)` (`~/.cache/laya/torchinductor`), `backends/compile.py:47-57` đặt `TORCHINDUCTOR_CACHE_DIR` | Chỉ khi `compile=True` + `compile_cache=True` hoặc `backend="compile"/"auto"` (trên CUDA). | KHÔNG mặc định. |
| L10 | `agent.py:2008-2039` `save_calibration`, `train.py:592-602,1235-1264`, `evals_cli.py:463-538` | Chỉ khi gọi tường minh. | KHÔNG. |

### 2.4. Biến môi trường được đọc

Trên đường mặc định (khi `Agent.__init__` / `predict` chạy). Gating: `LAYA_REVISION` (`revisions.py:75`, qua `agent.py:585`) và `HF_TOKEN` (`agent.py:588`) CHỈ được đọc trong nhánh `if not os.path.exists(model_dir)` (`agent.py:575-597`), tức khi đi tới Hub; nếu truyền thư mục snapshot local thì cả hai KHÔNG được đọc. `LAYA_SHA256_DIGESTS` (`verify_digests`, `agent.py:607`, `revisions.py:114`) được đọc ở CẢ hai đường. Danh sách: `LAYA_REVISION` (dùng khi KHÔNG truyền `revision=`; ghi đè ngầm sha khi đi qua Hub), `LAYA_SHA256_DIGESTS` (kiểm hash nếu đặt), `HF_TOKEN`, `LAYA_CUDA_AMP` (`agent.py:304`, đổi bf16 <-> fp16 trên CUDA), `LAYA_CPU_AMP` (`agent.py:317`), `LAYA_MPS_AMP_MIN_ROWS` (`agent.py:291,760`). Import-time: `__init__.py:22` `_os.environ.setdefault("USE_TF", "0")` (ghi env của process; lành tính).
Chỉ khi dùng compile: `LAYA_INDUCTOR_CACHE_DIR`, `LAYA_COMPILE_WARMUP`, `XDG_CACHE_HOME`, `TORCHINDUCTOR_*` (`backends/compile.py:48-61`, `_compile.py:49-52`).
Chỉ server / mcp / router: `LAYA_MODELS`, `LAYA_HOST`, `LAYA_PORT`, `LAYA_API_KEY`, `LAYA_BASE_URL`, `LAYA_REMOTE_TIMEOUT`, ... (`serve.py`, `mcp/*`, `router.py:439,729`).
Hành động: đảm bảo `LAYA_REVISION` và `LAYA_SHA256_DIGESTS` KHÔNG được đặt ngoài ý muốn trong môi trường chạy; luôn truyền `revision=` rõ ràng hoặc dùng thư mục snapshot.

### 2.5. Monkey-patch toàn cục torch/transformers

Đường mặc định: KHÔNG có patch toàn cục. Các thay đổi nhỏ:
- `agent.py:690` `self.model.encoder.config.reference_compile = compile` (False mặc định): đổi thuộc tính trên config của instance encoder để ModernBERT không tự `torch.compile`. Tốt cho eager.
- `common.py:661-681` `_apply_rope_config` sửa thuộc tính trên đối tượng `ecfg` vừa tạo (không toàn cục).
- `common.py:684-690, 708` dùng `transformers.initialization.no_init_weights()` (context manager của transformers; tạm thời bỏ qua init weight lúc dựng model, tự khôi phục).
- `os.environ.setdefault("USE_TF","0")` (`__init__.py:22`).
Chỉ khi bật tính năng: `agent.py:900` `self.model.forward = self._fast.forward` (fast=True/TileLang); `backends/*` thay `model.forward`; `_compile.py:82-107` đặt tạm `torch.fx.experimental._config.use_duck_shape=False` khi chạy forward đã compile; `backends/compile.py:48-57` đặt `TORCHINDUCTOR_*` và `torch._inductor.config.fx_graph_cache=True`; `tl_kernels.py:270` `tilelang.compile(..., target="c")` (biên dịch kernel C bằng toolchain cục bộ) và `fast.py`/`backends/tilelang.py` (CUDA). `backend="auto"` trên CUDA tự chọn tilelang nếu cài, ngược lại `compile` (`backends/__init__.py:50-59`): TRÁNH.

### 2.6. Offline: `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` có đủ không?

Có, nhưng chỉ khi:
1. Snapshot đã đủ file: `rl_agent_config.json`, `model.safetensors`, `tokenizer/*`, `encoder/*` (laya-multilingual: 1720e3e3..., laya-vi: 860b2b5f...). Assert tồn tại `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `encoder/config.json` trước khi load (chặn nhánh tải repo phụ L2/L3).
2. Tốt nhất truyền ĐƯỜNG DẪN snapshot local vào `laya.load(path)`: khi đó `agent.py:575` thấy thư mục tồn tại và bỏ qua toàn bộ `snapshot_download`; `revision` lúc đó bị bỏ qua. (Với repo id + `HF_HUB_OFFLINE=1`, `snapshot_download` phải tự tìm snapshot trong cache; hành vi này thuộc huggingface_hub, chưa kiểm chứng ở đây.)
3. Mặc định `backend=None`, `fast=False`, `compile=False` (không truyền `backend="auto"`). Không dùng `Router`, `laya.serve`, `laya.mcp`, `LayaRouter/LayaGuardrail/...` (`integrations`), không đặt `hooks` (hooks mặc định rỗng: `hooks.py:200`).
4. Không đặt `LAYA_REVISION`, `LAYA_SHA256_DIGESTS` (hoặc đặt đúng ý), không truyền `calibration=` trừ khi chủ ý.
5. Xử lý L8 (ghi đè tokenizer_config) theo mục 2.3.
Không gói nào ở đường này gửi dữ liệu ra ngoài ngoài HF Hub.

### 2.7. Điểm khác cần biết cho adapter (phát hiện trong khi review)

- `revisions.py:27-31` `PINNED_REVISIONS["convaiinnovations/laya-multilingual"] = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"` KHÁC sha team pin (`1720e3e3357cfe1e281542e223f8273b0890ca34`, `pins_models.json`). Không dùng `LAYA_REVISION=reviewed`. `laya-vi` không có trong bảng này. Luôn truyền sha của team qua `revision=` hoặc dùng snapshot dir.
- `agent.py:761-776`: trên CUDA `amp_enabled=True` và forward chạy bf16 autocast (compute capability >= 8; fp16 nếu thấp hơn); không có tham số constructor để tắt. Muốn fp32 eager thuần: sau khi load, đặt `agent.amp_enabled = False` (thuộc tính công khai, đọc ở `agent.py:1191`). Trên CPU mặc định fp32.
- Chuỗi `[MASK]` trong state/câu hỏi bị thay bằng dấu cách (`agent.py:1151`, `common.py:237,246`) nên không gây lỗi như Quyet encoder.

---------------------------------------------------------------------------------------------------

## 3. mstrasser/Jeff-Qwen3.5-2B (sha 568a137a3e912016db67b13d80dcddb43c491932)

Nguồn: `README.md`, `config.json`, `decision_config.json`, `processor_config.json`, `chat_template.jinja` (local), danh sách file HF trong `pins_models.json`. KHÔNG có code Python nào của Jeff trong HF repo hay local.

Có cần `trust_remote_code` hoặc code ngoài transformers không?
- `trust_remote_code`: KHÔNG cho phần backbone. `config.json` có `model_type: "qwen3_5"`, `architectures: ["Qwen3_5Model"]`, không có `auto_map`; `transformers_version: 5.17.0`. Backbone nạp được bằng class gốc của transformers (cần phiên bản có Qwen3.5). Chat template giống hệt `Qwen/Qwen3.5-2B` (sha256 trùng `273d8e0e...`).
- Code ngoài transformers: CÓ. `architectures` là `Qwen3_5Model` (không có LM head), HF tag `feature-extraction`. Repo có thêm `readout.safetensors` (1.044.568 byte; suy luận từ kích thước khoảng 254 hoặc 255 hàng x 2048 x 2 byte, chưa xác minh) và `decision_config.json` (`max_options: 254`, `codes` A..Z, AA.. (255 mã, trong khi `max_options` là 254; lệch 1 chưa rõ lý do), `token_ids` (255 id token tương ứng, không trùng), `temperature: 1.1382043387839655`, `prompt_layout: "state-first"`). README không nhắc `readout.safetensors` và không có dòng Python nào dùng transformers. Prompt (template của state-first layout) và cách áp dụng readout KHÔNG có trong HF repo; các hash trong `decision_config.json["provenance"]` (`model.py`, `encoder.py`, `decoder.py`, `train.py`...) trỏ tới file nguồn của repo GitHub, không có local. Tôi không suy đoán cách readout được áp dụng.
- README chỉ đưa lệnh shell (README dòng 182-187), tóm tắt tác dụng:
  - `git clone https://github.com/firelex/jeff && cd jeff`: tải code chạy model từ GitHub (chưa review, không có local).
  - `uv sync --no-default-groups` (thêm `--extra cuda` hoặc `--extra mac`): cài phụ thuộc từ mạng theo `uv.lock`.
  - `uv run --no-default-groups hf download mstrasser/Jeff-Qwen3.5-2B --local-dir Jeff-Qwen3.5-2B`: tải weight HF vào thư mục local (README dòng 94 cũng nói `--revision v1.0` cho bản cũ).
  - `JEFF_CHECKPOINT=Jeff-Qwen3.5-2B PORT=8765 uv run --no-default-groups jeff-serve`: chạy HTTP server (địa chỉ bind không rõ vì không có code).
  - `curl -s localhost:8765/v1/systemone -H 'content-type: application/json' -d @request.json`: gửi request tới server.
- Kết luận: không thể xác nhận Jeff chạy hoàn toàn local và an toàn bằng tài liệu local. Để dùng: (1) tải repo `firelex/jeff` ở commit `a095c98f3fd2782bba033b378f7375ecc33c35fa` (lấy từ `decision_config.json["provenance"]["git_commit"]`, suy ra từ file; chưa xác nhận đúng với bản serve) trên máy có mạng, review toàn bộ code, vendor vào dự án (không `git clone` + `uv sync` ở máy chấm); hoặc (2) tự cài đặt lại readout dựa trên `decision_config.json` + `readout.safetensors` (rủi ro sai prompt). Cả hai đều cần quyết định của người phụ trách; đây là điểm CHẶN adapter (mục "blockers" ở notes_model_api.md).

---------------------------------------------------------------------------------------------------

## 4. Giới hạn của review này

- Không xem được nội dung `tokenizer/tokenizer_config.json` của laya (502/624 byte) nên không biết `_fix_tokenizer_config` có ghi hay không; không xem được các file weight/tokenizer của bất kỳ model nào (chỉ có metadata).
- Hành vi của `huggingface_hub`/`transformers` (offline mode, `snapshot_download` offline với `allow_patterns`, Jinja sandbox) lấy theo hiểu biết, chưa chạy kiểm chứng.
- Các module laya liệt kê ở 2.1 là "chỉ quét regex" chưa đọc từng dòng; bằng chứng chúng không nằm trên đường mặc định là cấu trúc import (`_LAZY_ATTRS`, import trong hàm) và các điểm thoát sớm đã nêu.
- Jeff: không có code, nên mọi nhận định về runtime của Jeff là "không rõ".
