# Ghi chú API để viết adapter: Quyet Tiny/Small/Medium, laya-multilingual, laya-vi, Jeff-Qwen3.5-2B

Nguồn: code local `pkgsrc\quyet-1.0.2` và `pkgsrc\laya-0.4.6`, model card/config trong `meta\<org>__<model>\`, `pins_models.json`. Chỉ đọc; không chạy gì. Số dòng là số dòng file local.
Trích README trong ghi chú này đều <= 15 từ. Mọi chỗ không có bằng chứng ghi "không rõ".
Đường dẫn gốc: `C:\Users\dungbt06\Fine-tune\results\baselines\`.
Review bảo mật nằm ở `notes_code_review.md` (cùng thư mục này).

## 0. Bảng tổng quan

| Model | Pin sha (pins_models.json) | Loại / base | Tham số | Entry point |
|---|---|---|---|---|
| chinhnc/Quyet-1.0-Tiny | fba8ca9292a705e21b14a4c5150e79a4ddcc8db3 | encoder, `mmBERT-small` (16/22 layer đầu) + head `option_query_wide`, distil từ Quyet-1.0-Medium | 183M (182.924.673) | `quyet.load(...).predict(state, questions)` |
| chinhnc/Quyet-1.0-Small | f007dfeb4874f0f3b3a3cc0a63f4d4192164c1c2 | encoder, `aisingapore/SEA-LION-ModernBERT-300M` + head `refine_pool_v1` (2 layer) | 328M (327.609.601) | như trên |
| chinhnc/Quyet-1.0-Medium | bd509d64070e8ff39328e0b883ac86b9313d15f0 | LLM, `Qwen/Qwen3.5-4B` + LoRA rank 16 đã merge, đọc logit chữ cái A..J | 4,66B (4.659.865.088; 4,21B text) | như trên |
| convaiinnovations/laya-multilingual | 1720e3e3357cfe1e281542e223f8273b0890ca34 | encoder `jhu-clsp/mmBERT-base` (22 layer, hidden 768) + head 2 layer + scorer + act head; "trained from scratch with RLCD" | 322M (321.908.998) | `laya.load(...).predict(state, questions)` |
| vanty120/laya-vi | 860b2b5f145a1013161dcedd93fda7c263a3c9f0 | fine-tune của laya-multilingual (xem 2.9: front-matter ghi `base_model: convaiinnovations/laya`) | 322M (321.908.998) | như trên |
| mstrasser/Jeff-Qwen3.5-2B | 568a137a3e912016db67b13d80dcddb43c491932 | full fine-tune `Qwen/Qwen3.5-2B` (rev 15852e8c..., theo `decision_config.json`), head `readout.safetensors` | khoảng 2,21B (2.213.241.664; README ghi "2B") | chỉ qua repo GitHub `firelex/jeff` (không có local) |

Phụ thuộc: quyet cần `torch>=2.4`, `transformers>=5.17`, `safetensors>=0.4`, `huggingface-hub>=0.34`, `numpy>=1.26` (METADATA). laya cần `torch>=2.0`, `transformers>=4.48`, `safetensors`, `huggingface_hub`, `numpy` (checkpoint lưu bằng transformers 5.x; `common._apply_rope_config` xử lý rope, `common.py:661-681`). Một venv với transformers >= 5.17 phục vụ được cả quyet, laya và decider.

---------------------------------------------------------------------------------------------------

## 1. Quyet 1.0 (Tiny, Small, Medium), gói `quyet` 1.0.2

Kiểu model đọc từ `quyet_config.json["kind"]`: Tiny/Small = `encoder`, Medium = `llm` (`hub.py:27-28,37,48`).

### 1a. Nạp model ở revision đã pin

```python
quyet.load(name_or_path, *, device=None, revision=None, dtype=None, device_map=None)   # hub.py:32
```
- `revision=` CÓ: `hub.py:16` `Path(snapshot_download(str(name_or_path), revision=revision))`. Nếu `name_or_path` là THƯ MỤC tồn tại (`hub.py:12-14`) thì dùng thẳng và `revision` bị bỏ qua. Không dùng `allow_patterns` nên tải toàn repo.
- Cách pin: `quyet.load(r"<snapshot_dir>")` với snapshot đã tải đúng sha, hoặc `quyet.load("chinhnc/Quyet-1.0-Small", revision="f007dfeb4874f0f3b3a3cc0a63f4d4192164c1c2")`. Truyền đường dẫn tuyệt đối; chuỗi repo id trùng thư mục tương đối trong CWD sẽ bị coi là thư mục.
- `device=None`: cuda nếu có, ngược lại cpu (`hub.py:40-43,45-47`). `dtype` và `device_map` chỉ áp dụng cho LLM (Medium); `dtype` mặc định `torch.bfloat16` (`llm/runtime.py:22`). `device_map` với encoder -> ValueError (`hub.py:38-39`).
- Thiếu `quyet_config.json` -> `ValueError` (`hub.py:20-22`); `quyet_format != 1` hoặc `kind` lạ -> `ValueError` (`hub.py:24-28`).

### 1b. Định dạng input (một câu hỏi chọn đáp án)

```python
m.predict(state, questions, *, strict=False)        # base.py:36
m.predict_many(requests, *, strict=False)           # base.py:40; requests = [{"state":..., "questions":...}, ...]
m.raw_probabilities(requests, *, strict=False)      # base.py:45
```
- `state`: `str | dict | list` (`base.py:19-21`, ngoài ra `QuestionError`). Chuỗi dùng nguyên văn; dict/list bị serialize: encoder `json.dumps(state, ensure_ascii=False)` (`encoder/inputs.py:20-21`), LLM `json.dumps(..., ensure_ascii=False, indent=1)` (`llm/prompt.py:42-47`). Danh sách (list) được coi là hội thoại: cắt giữ phần đuôi. Với benchmark nên truyền `state` là `str` để khống chế đúng văn bản.
- `questions`: dict KHÔNG rỗng `{question_id: question}` (`questions.py:71-80`); mỗi `question` là dict (`questions.py:38-68`):
  - `type`: một trong `"choice" | "score" | "noul"` (`questions.py:16`).
  - `instructions`: `str` không rỗng (`questions.py:44-45`); dict/list/số -> `QuestionError`.
  - `criteria`:
    - `choice`: dict `{label: description}` hoặc list nhãn; nhãn được `str()` và phải duy nhất (`questions.py:46-54`). Khóa trả về = nhãn, theo đúng thứ tự đưa vào.
    - `score`: list mô tả mức, mức thấp trước; khóa trả về `"0".."K-1"` (`questions.py:55-58`).
    - `noul`: dict với khóa tùy chọn `"true"`/`"false"` (hoặc bỏ `criteria`); khóa khác bị bỏ qua im lặng (`questions.py:59-63`). Khóa trả về `("true","false")`.
  - Số option: 2..10; ngoài khoảng -> `QuestionError` (`questions.py:17,64-67`). README: "At most 10 options per question".
  - Mô tả "rỗng" (`None`, `""`, `0`, `False`, `{}`, `[]`) coi như không có mô tả; giá trị không phải str -> JSON gọn (`questions.py:28-35`).
- Nhiều câu hỏi trong một request được chấm ĐỘC LẬP (mỗi câu một sequence/row: `encoder/runtime.py:52-53`, `llm/runtime.py:47-48`), không nhồi chung một prompt.

### 1c. Option dạng nhãn trần vs khóa trung tính + mô tả

Encoder (Tiny, Small): sequence = `CLS "<type> question: <instructions>" SEP {MASK " <label>[:]" [" <text>"]}* SEP state SEP` (docstring `encoder/inputs.py:1-9`; code `inputs.py:50-73`). Câu hỏi và option đứng TRƯỚC state.
- (1) Nhãn trần là toàn bộ nội dung option: `"criteria": ["nội dung A", "nội dung B", ...]` (hoặc dict giá trị `None`) -> render `" nội dung A"` (không dấu hai chấm, không span mô tả) (`inputs.py:59-62`).
- (2) Khóa trung tính + mô tả: `"criteria": {"A": "nội dung A", "B": "nội dung B"}` -> `" A: nội dung A"`: khóa VÀ mô tả cùng được model đọc.
- Score: nhãn mô hình thấy là `level i` + mô tả: `("level %d" % i, text or "level %d" % i)` (`inputs.py:28-30`). Noul: thứ tự nội bộ `[false, true]`, mô tả mặc định "no, the statement does not hold" / "yes, the statement holds" (`inputs.py:31-33`), kết quả được ánh xạ lại (`to_label=[1,0]`).

LLM (Medium): prompt (`llm/prompt.py:62-73`): system + user = `State:\n...\n\nQuestion: <instructions>\n<KIND>\n\nOptions:\n<A. ...>` + `Answer with one letter.`; chữ cái A..J theo VỊ TRÍ (`prompt.py:14,64`).
- Văn bản option = mô tả nếu có, nếu không là nhãn (`prompt.py:23-29`: `t if t is not None else k`). Nghĩa là KHÓA BỊ ẨN khi có mô tả. (1) `["nội dung A", ...]` -> `A. nội dung A`; (2) `{"X": "nội dung A"}` -> `A. nội dung A` (khóa `X` không vào prompt). Hai cách cho cùng prompt khi nội dung giống nhau.
- Score -> `A. <mô tả hoặc "0">`; noul -> `A. <desc true hoặc "true">`, `B. <desc false hoặc "false">`, kèm câu "Choose A if the statement is true for this state, B if it is not." (`prompt.py:18-29`).

### 1d. Giá trị trả về, làm tròn, nhiệt độ

`predict` trả `{"model", "answers", "usage", "warnings"}` (`answers.py:26-31`). Trong `answers[qid]`:
- choice: `{"type","choice": nhãn argmax, "confidence": top-prob, "probabilities": {nhãn: p}}`; score: `score` (kỳ vọng mức), `confidence` = max(p), `legend`, `probabilities` khóa `"0".."K-1"`; noul: `noul` = P(true), `confidence` = max(p, 1-p) (`answers.py:9-23`). Ở Quyet `confidence` chính là xác suất top-1 (khác Laya/decider).
- Làm tròn: `_r(x) = round(float(x), 4)` (`answers.py:5-6`) cho mọi xác suất. Thứ tự = thứ tự nhãn đưa vào. Tổng = 1 trước khi làm tròn (`softmax_t` chuẩn hóa: `calibration.py:16-19`); sau làm tròn có thể lệch tối đa ~k*5e-5.
- Xác suất CHƯA làm tròn: `m.raw_probabilities([ {"state":..., "questions":...} ], strict=True)` -> list theo request, mỗi phần tử là `{qid: [float,...]}` căn theo thứ tự nhãn (`base.py:45-47`; noul `[P(true), P(false)]`). CẢNH BÁO: các số này ĐÃ chia nhiệt độ (calibrated), chỉ không làm tròn: `_run` gọi `softmax_t(lg, temperature(...))` (`encoder/runtime.py:69`, `llm/runtime.py:106`) rồi `raw_probabilities` trả `r.probs`. README (METADATA) nói hàm trả "the uncalibrated distributions", mâu thuẫn với code; tin code. Muốn T=1: sau khi load, đặt `m.temps = {}` (`calibration.py:13` rơi về 1.0).
- Nhiệt độ: đọc từ `quyet_config.json["temperatures"]` lúc khởi tạo (`encoder/runtime.py:24`, `llm/runtime.py:33`: `dict(config.get("temperatures") or {})`). Tra bảng (`calibration.py:12-13`): `temps.get(f"{qtype}:{bucket}", temps.get(qtype, 1.0))`, bucket = `"2"` (k<=2), `"3-5"`, `"6-10"` (`calibration.py:8-9`).

| Model | Giá trị phát hành (`quyet_config.json`) |
|---|---|
| Tiny | choice:2 = 2.048; choice:3-5 = 1.3947; choice:6-10 = 1.1443; noul:2 = 1.3469; score:3-5 = 1.1713 |
| Small | choice:2 = 3.4582; choice:3-5 = 2.0963; choice:6-10 = 1.5309; noul:2 = 1.4276; score:3-5 = 1.9322 |
| Medium (theo loại, mọi k) | choice = 1.3313; score = 1.6039; noul = 1.4276 |

- Khe hở: Tiny/Small chỉ có `score:3-5`; score 2 mức hoặc 6-10 mức không có khóa nào (không có khóa `score` trần) -> T = 1.0 IM LẶNG. choice/noul được phủ hết (choice tối đa 10 option; noul luôn 2).
- Đảm bảo dùng đúng giá trị phát hành: load từ thư mục snapshot, rồi assert `m.temps == json.load(open(snapshot/"quyet_config.json"))["temperatures"]`; không có đường nào khác thay đổi `temps`. Thiếu file config -> lỗi (không fallback); chỉ thiếu khóa `temperatures` hoặc thiếu bucket mới rơi T=1 mà không cảnh báo.

### 1e. Độ dài tối đa / cắt

- Encoder: `max_len = quyet_config.json["max_len"]` = 8192 (`encoder/runtime.py:23`). Chỉ cắt state (`inputs.py:65-69`: `room = max_len - len(head) - 1`); list giữ đuôi (`keep_end`), còn lại giữ đầu. Câu hỏi + option vượt cửa sổ -> `QuestionError` bất kể `strict` (`inputs.py:66-67`).
- LLM Medium: `limits` = `max_state_tokens 6000`, `max_prompt_tokens 8000`, `min_state_tokens 256` (`quyet_config.json`; `llm/runtime.py:35-38`). `_prompt_ids` (`llm/runtime.py:42-63`) cắt state về `cap`, co `cap` đến khi prompt dài nhất <= 8000, và nếu `cap < 256` ném `QuestionError` (54-55). Khi cắt, state được decode lại kèm dấu " …" (hoặc "… " đầu với list) (`llm/prompt.py:59`): dấu này nằm trong prompt.
- Cắt KHÔNG im lặng ở `predict`: response có `warnings: [{"code":"state_truncated","state_tokens":N,"kept_tokens":K,"kept":"head"|"tail"}]` (`base.py:24-26`) và `answers[qid]["truncated"] = True` (`answers.py:21-22`). Chế độ lỗi thay vì cắt: `strict=True` (`encoder/runtime.py:57-59`, `llm/runtime.py:60-61`), tức `predict(..., strict=True)`; `raw_probabilities(..., strict=...)` mặc định `False` và KHÔNG trả `warnings` nên phải truyền `strict=True` hoặc tự kiểm độ dài. Card: "Only the state is ever truncated".
- Nâng giới hạn: `m.max_len` (encoder, `runtime.py:23`) hoặc `m.max_state_tokens` / `m.max_prompt_tokens` (LLM, `llm/runtime.py:36-37`) là thuộc tính thường có thể ghi đè sau khi load; nhưng encoder chỉ có `max_position_embeddings 8192` (Small dùng `position_embedding_type: absolute`, không vượt được 8192) và Medium chỉ huấn luyện với prompt tối đa 8000 token (README: "state up to 6,000 tokens inside an 8,000-token prompt"). Không khuyến nghị.

### 1f. Batch-1, device, dtype, kernel

- `m.predict(state, questions)` là batch-1; mỗi câu hỏi một row. Encoder gom theo độ dài: `max_batch_tokens=32768`, `max_batch_rows=256` (`encoder/runtime.py:25,36`); LLM `max_batch_tokens=8000`, padding trái, `logits_to_keep=1` (`llm/runtime.py:39,87-88`).
- Encoder: weight F32; CUDA chạy `torch.autocast("cuda", dtype=torch.bfloat16)`, CPU fp32 (`encoder/runtime.py:35-39`); attention `sdpa` (`encoder/modeling.py:122`). LLM: `dtype` mặc định bf16, `device_map` tùy chọn.
- Không có `torch.compile`, CUDA graph, patch kernel, patch toàn cục nào trong quyet. Bản "eager" là mặc định duy nhất.
- Medium: `config.json` ghi `architectures: ["Qwen3_5ForConditionalGeneration"]`; code thử `AutoModelForCausalLM` rồi nếu `ValueError/KeyError` thì `AutoModelForImageTextToText` (`llm/runtime.py:24-27`); lớp thực dùng phụ thuộc phiên bản transformers (nếu ImageText thì nạp cả vision tower, tổng 4,66B so với 4,21B text). `chat_template.jinja` của Medium mặc định BẬT thinking (`<think>\n`); quyet truyền `enable_thinking=False` (`llm/prompt.py:71`) để ra `<think>\n\n</think>\n\n`. Nhánh `except TypeError` (`prompt.py:72-73`) bỏ tham số này và sẽ sinh prompt khác trên transformers cũ. Dùng transformers >= 5.17, đừng tự dựng prompt bằng `apply_chat_template` mặc định.

### 1g. Ví dụ trong README

Ba README (Tiny, Small, Medium) và METADATA dùng chung một ví dụ, KHÔNG có output số: chỉ có chú thích hình dạng kết quả.
```python
m.predict({"message": "Please close my card, I lost it yesterday."},
  {"intent": {"type": "choice", "instructions": "What does the customer want?",
              "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
   "urgent": {"type": "noul", "instructions": "The request is urgent."},
   "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}})
```
Chú thích gốc (rút gọn): `{"intent": {"choice": ..., "confidence": ..., "probabilities": {...}}, "urgent": {"noul": P(true)}, ...}`. Không có giá trị kỳ vọng để so sánh; chỉ kiểm hình dạng (3 khóa, tổng xác suất ~1, thứ tự nhãn `cancel, limit, other`, score có `legend`). Lưu ý `"other": None` -> nhãn trần (xem 1c).

### 1h. Dữ liệu huấn luyện (theo model card / config)

- README ba model KHÔNG nêu dữ liệu huấn luyện; HF metadata `datasets: null`. Chỉ có: "English; also tuned for Vietnamese" (không nói dữ liệu tiếng Việt từ đâu); Tiny "distilled from Quyet-1.0-Medium"; Medium "fine-tuned from Qwen3.5-4B" (LoRA r16, `source.step` = 120000); số bước: Tiny 50000, Small 44000.
- MASSIVE: không rõ. Belebele: không rõ. SIB-200/FLORES: không rõ. COPA/XCOPA: không rõ. ViFactCheck, ViWikiFC và dữ liệu fact-checking/NLI tiếng Việt khác: không rõ.
- Base có tiền huấn luyện riêng (mmBERT-small, SEA-LION-ModernBERT-300M, Qwen3.5-4B); nội dung tiền huấn luyện không được card nêu; không suy đoán.

### 1i. Tham số và base

Xem bảng mục 0. Medium: "Architecture: Qwen3.5-4B with a merged LoRA fine-tune (rank 16)", prompt version 1 ("This model uses prompt version 1").

---------------------------------------------------------------------------------------------------

## 2. Laya (laya-multilingual, laya-vi), gói `laya` 0.4.6

Hai model dùng cùng code, cùng kiến trúc, chỉ khác `rl_agent_config.json` và weight.

### 2a. Nạp model ở revision đã pin

```python
laya.load(model_id_or_path, device=None, token=None, subfolder=None, fast=False, compile=False, revision=None,
          expected_sha256=None, lang_temperatures=None, hooks=None, ..., calibration=None, backend=None, ...)   # agent.py:2088
```
- `revision=` CÓ: `agent.py:585,593-595` `snapshot_download(model_id, revision=..., allow_patterns=[rl_agent_config.json, model.safetensors, tokenizer/*, encoder/*])`. Hoặc truyền THƯ MỤC snapshot (khi đó `agent.py:575` bỏ qua Hub, `revision` không dùng).
- Dùng repo id ĐẦY ĐỦ có dấu `/` (`convaiinnovations/laya-multilingual`, `vanty120/laya-vi`). Tên ngắn `laya-multilingual` bị resolve thành repo bundle `convaiinnovations/laya` + subfolder `multilingual` (`router.py:80-84`, `agent.py:2131-2136`): khác repo.
- `revision` mặc định lấy từ biến môi trường `LAYA_REVISION` nếu không truyền (`revisions.py:75`). Bảng `PINNED_REVISIONS["convaiinnovations/laya-multilingual"]` (= `e4e9ddf2...`, `revisions.py:29`) KHÁC sha team pin (`1720e3e3...`): không dùng `LAYA_REVISION=reviewed`. Có thể kiểm toàn vẹn qua `expected_sha256={"model.safetensors": "<hex>", ...}` (`agent.py:607`, `revisions.py:102-145`).
- Lưu ý: `agent.py:609 _fix_tokenizer_config` có thể ghi đè `tokenizer/tokenizer_config.json` trong snapshot (xem `notes_code_review.md` mục 2.3). Nên load từ bản copy có quyền ghi hoặc chấp nhận.

### 2b. Định dạng input

```python
agent.predict(state, questions, lang=None, hooks=None, ..., max_len=None, head_max_len=None, min_confidence=None)   # agent.py:1882, predict = system_one (1981)
agent.predict_batch(states, questions, batch_size=None, ...)                                                         # agent.py:1409
```
- `state`: `str | dict | list` (`serialize_state`, `common.py:94-97`: str nguyên văn, còn lại `json.dumps(ensure_ascii=False)`); `None` -> TypeError. Chuỗi `[MASK]` trong state/câu hỏi bị thay bằng dấu cách (`agent.py:1151`, `common.py:237,246`).
- `questions`: `{qid: {...}}`. Trường (validate ở `agent.py:968-1106`, chuẩn hóa ở `agent.py:1109-1130`):
  - `type`: `"choice" | "score" | "noul"` (`common.py:18`).
  - `instructions`: chuỗi không rỗng; dict/list/số được `json.dumps(ensure_ascii=False)` (`agent.py:1117-1124`). Bắt buộc.
  - `criteria`: choice = dict `{nhãn: mô tả|None}` hoặc list nhãn; nhãn phải là vô hướng (str/int/float/bool), không null, không trùng; score = list mô tả (mức 0 trước, không có None); noul = tùy chọn, dict chỉ có khóa `"true"`/`"false"` (khóa khác -> `ValueError`, `agent.py:1070-1087`).
  - Tùy chọn: `option_order` (hoán vị chỉ số), `labels` (chỉ noul: `{"false":..,"true":..}` đổi chữ hiển thị).
- Sequence: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]` (`common.py:169`). Option đứng trước state, state ở cuối.
- Số option: không có giới hạn cứng trong `Agent`; thực tế bị chặn bởi ngân sách head (xem 2e). README laya-vi: choice 3-10 option, score 3-5 mức. README laya-multilingual: "Keep `choice` questions under ~20 options".
- Các câu hỏi trong một lời gọi chia chung state nhưng là các row ĐỘC LẬP trong cùng một forward.

### 2c. Nhãn trần vs khóa trung tính + mô tả

`render_options` (`common.py:127-153`):
- (1) Nhãn trần là toàn bộ nội dung: `"criteria": ["nội dung A", "nội dung B"]` (list) hoặc dict giá trị `None`/`""` -> option là `str(nhãn)` (`common.py:141`). Nhãn đồng thời là KHÓA trả về trong `probabilities`.
- (2) Khóa trung tính + mô tả: `"criteria": {"A": "nội dung A", "B": "nội dung B"}` -> option là `"A: nội dung A"` (`"%s: %s" % (k, render_criterion(v))`): khóa VÀ mô tả đều vào token. Khác với Quyet-Medium (ẩn khóa).
- Mỗi option bị cắt tối đa 48 token (`common.py:244-250`, `truncation=True, max_length=48`).
- score: option `"level %d: %s"` (`common.py:144`), tức chỉ số hiện trong text. noul: `"false: <desc|no, the statement does not hold>"`, `"true: <desc|yes, the statement holds>"` (`common.py:148-153`).
- Tài liệu cảnh báo (METADATA, "Honest limits"): tránh nhãn dạng từ boolean (`true/false`, `yes/no`) vì model có thể bám vào nhãn; khuyên dùng nhãn mờ như `A`/`B`; và nhãn ngữ nghĩa không làm phủ định an toàn hơn.

### 2d. Giá trị trả về, làm tròn, nhiệt độ

`predict` trả `{"model": "laya-rl-agent", "answers": {...}, "usage": {...}}` (`agent.py:1567-1571`). Trong `answers[qid]` (`agent.py:1367-1404`):
- choice: `choice` (argmax), `probabilities` `{nhãn: round(p,4)}` theo thứ tự `criteria`, `confidence` = 1 - entropy chuẩn hóa (KHÔNG phải top-prob), `answer_confidence` = max(p) (làm tròn 4; qua `binning_map` nếu có), `action.act_probability`.
- score: `score` = kỳ vọng mức (round 4), `legend`, `probabilities` khóa `"0".."K-1"`, `confidence`, `answer_confidence`.
- noul: `noul` = `round(p[1], 4)` = P(true), do thứ tự option là `[false, true]`; `confidence = max(p, 1-p)`.
- Tổng = 1 trước khi làm tròn (`p = exp(z - z.max()); p / p.sum()`, `agent.py:1345-1347`); làm tròn 4 chữ số cho mọi xác suất. Thứ tự giữ nguyên (`unpermute_probs` chỉ đổi khi có `option_order`).
- XÁC SUẤT CHƯA LÀM TRÒN: không có API công khai trả về. Dùng đường nội bộ giống `calibrate.records_from_labeled` (`calibrate.py:505-522`; hàm riêng tư, kiểm lại chữ ký khi nâng cấp gói):
```python
import numpy as np, torch
from laya.common import collate_items, temp_bucket
qdef = {...}                                              # một câu hỏi như 2b
agent._check_question("q", qdef); internal = {"q": agent._to_internal(qdef)}
with torch.no_grad():
    items = agent._encode_state(state, ["q"], internal)    # agent.py:1132; items[0]["state_stats"]["truncated"]
    logits, _ = agent._forward(collate_items([items], agent.tok.pad_token_id))   # agent.py:1326, numpy float32
k = len(items[0]["markers"]); qt = items[0]["qtype"]
T = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])  # agent.py:1341
z = logits[0, :k] / T; p = np.exp(z - z.max()); p /= p.sum()                      # thứ tự = thứ tự criteria; noul: [false, true]
```
- Nhiệt độ: `rl_agent_config.json` -> `temperature` (list 3 số theo loại `[choice, score, noul]`, `common.py:18`) và `temperature_by_options` (khóa bucket `"choice:2"|"choice:3-5"|"choice:6-10"|"choice:11+"`, ... ; `common.py:826-828`), đọc ở `agent.py:707-718`, KẸP về [0.5, 5.0] (`common.py:835-854`, có `RuntimeWarning` nếu bị đổi, `agent.py:739-744`). Áp dụng: bucket nếu có, không thì số theo loại (`agent.py:1341`).

| Model | `temperature` | `temperature_by_options` |
|---|---|---|
| laya-multilingual | [1.0, 1.0, 1.0] | {} (card: "Ships uncalibrated.") |
| laya-vi | [1.0953248739242554, 1.1482385396957397, 1.1208693981170654] | choice:3-5 = 1.093132495880127; choice:6-10 = 1.1110963821411133; score:3-5 = 1.1482385396957397; noul:2 = 1.1208693981170654 |

  Với laya-vi: choice 2 option hoặc >= 11 option (không có bucket) dùng 1.0953; score 2 mức hoặc >= 6 mức dùng 1.1482; noul luôn 1.1209.
- Có rơi im lặng về T=1 không? File thiếu -> `FileNotFoundError` (`agent.py:612-617`), không im lặng. Khóa `temperature` thiếu trong file -> `cfg.get("temperature", [1.0, 1.0, 1.0])` (`agent.py:707`) im lặng. Các đường ghi đè: tham số `calibration=`, `lang_temperatures=` + `lang=` ở `predict`, và gán `agent.temperature...`. Cách chắc chắn: load từ thư mục snapshot, KHÔNG truyền `calibration`/`lang_temperatures`/`lang`, rồi assert `agent.temperature_raw == cfg["temperature"]` và `agent.temperature_by_options_raw == cfg["temperature_by_options"]`, và không có `RuntimeWarning` về nhiệt độ bị kẹp.

### 2e. Độ dài tối đa / cắt

- Giới hạn lấy từ `rl_agent_config.json`: `max_len = 1024`, `head_max_len = 256` (cả hai model; mặc định trong code 512/192 chỉ dùng khi thiếu khóa: `agent.py:1139-1140`). README laya-multilingual: "1024 tokens per question, of which 256 go to the question and its options".
- Phần còn lại cho state: `room = max_len - len(head) - 1` (`common.py:199`). Chỉ state bị cắt, giữ ĐẦU (list giữ ĐUÔI: `agent.py:1144`, `common.py:204`).
- Cắt KHÔNG ném lỗi và không có chế độ strict. Phát hiện qua `result["usage"]`: `truncated` (bool), `state_tokens`, `state_tokens_dropped`, `truncated_questions` (`agent.py:1548-1558`); adapter tự `raise`/ghi log khi `usage["truncated"]` là True (hoặc đọc `items[0]["state_stats"]["truncated"]` ở công thức xác suất chưa làm tròn).
- Head (câu hỏi + option) cũng có thể bị cắt IM LẶNG: mỗi option tối đa 48 token; khi tổng option > `head_max_len - 16`, mỗi option chỉ còn `max(4, (head_max_len-16)//k)` token (`common.py:252-257`) và `instructions` bị cắt về `max(8, opt_budget)` token (`common.py:259`; câu hỏi dài sẽ bị cắt khi option dài). Chỉ báo hiệu qua `usage["options"]` khi các option trở nên giống hệt nhau (`agent.py:1564-1566`).
- Nâng giới hạn: theo từng lời gọi `agent.predict(state, questions, max_len=8192, head_max_len=512)` (`agent.py:1886-1887`) hoặc đổi `agent.cfg["max_len"]`/`["head_max_len"]`. README laya-multilingual: model đọc tới 8192 token ("pass `max_len=8192`" cho tài liệu dài). laya-vi được fine-tune ở `max_len 1024`, `head_max_len 256` (`train_log.json`), nên > 1024 ngoài phân phối huấn luyện. Còn `predict_long(state, questions, window=, stride=)` (`agent.py:1600`) quét cửa sổ nhưng xác suất trả về không hiệu chuẩn cho cả tài liệu.

### 2f. Batch-1, device, dtype, kernel

- Batch-1: `agent.predict(state, questions)`.
- Device: `device=None` -> cuda, mps, xpu, cpu theo thứ tự (`agent.py:662-670`); `device="cuda"`/`"cpu"` được; yêu cầu cuda mà không có -> cảnh báo rồi CPU (`agent.py:634-637`). Lỗi OOM trên GPU -> tự thử lại trên CPU có cảnh báo (`agent.py:1255-1303`).
- Backend: mặc định `backend=None, fast=False, compile=False` = eager. TRÁNH `backend="auto"` (CUDA tự chọn tilelang nếu cài hoặc `torch.compile`, `backends/__init__.py:50-59`), `fast=True`, `compile=True`, `compile_mode`. `agent.py:690` đặt `encoder.config.reference_compile = False` để ModernBERT không tự compile.
- Độ chính xác: tham số mô hình fp32; CUDA tự bật autocast bf16 (compute capability >= 8; checkpoint `amp_dtype: bf16`; thấp hơn thì fp16) (`agent.py:761-776`); đổi bằng env `LAYA_CUDA_AMP`; tắt autocast để chạy fp32 thuần: `agent.amp_enabled = False` sau khi load (đọc ở `agent.py:1191`). CPU: fp32 (env `LAYA_CPU_AMP=bf16` mới bật bf16).

### 2g. Ví dụ trong README và output nêu rõ

laya-multilingual (README dòng 52-60, "Quickstart"): chỉ nêu nhãn, không nêu số.
```python
agent = laya.load("convaiinnovations/laya-multilingual")
result = agent.predict(
    {"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।"},
    {"department": {"type": "choice", "instructions": "Which team should handle `body`?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}},
     "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"}})
print(result["answers"]["department"]["choice"])      # billing
```
Output kỳ vọng nêu rõ: `billing`. `refund_requested` không có giá trị nêu.

laya-vi (README dòng 44-61):
```python
state = ("Chào shop, tôi đã chuyển khoản 2 lần cho đơn hàng #4521 tháng 3. "
         "Nhờ shop hoàn lại khoản bị trừ trùng trong hôm nay, nếu không tôi sẽ huỷ tài khoản.")
questions = {
  "bo_phan": {"type": "choice", "instructions": "Bộ phận nào nên xử lý yêu cầu này?",
              "criteria": {"thanh_toan": "hoá đơn, thanh toán, hoàn tiền", "ky_thuat": "lỗi, sự cố, hệ thống", "khac": "các vấn đề khác"}},
  "muc_do_khan": {"type": "score", "instructions": "Yêu cầu này khẩn cấp đến mức nào?", "criteria": ["không gấp", "sớm", "rất gấp"]},
  "nguy_co_roi_bo": {"type": "noul", "instructions": "Khách hàng có đe doạ huỷ hoặc rời bỏ dịch vụ không?"}}
agent = laya.load("vanty120/laya-vi"); result = agent.predict(state, questions)
```
Output kỳ vọng (README chú thích): `bo_phan` -> `thanh_toan` "(p = 0.96)"; `muc_do_khan` -> score `1.71` trên thang 0-2 với "rất gấp": p = 0.75; `nguy_co_roi_bo` -> noul `0.94`. Đây là số làm tròn của card, dùng làm sanity check với dung sai (vd. +-0,02), không phải số chính xác; chạy ở bản pin và GPU/CPU khác có thể lệch ở chữ số thập phân cuối.

### 2h. Dữ liệu huấn luyện

- laya-multilingual (card HF): không nêu bộ dữ liệu. Chỉ có "Trained from scratch with RLCD: 15,987 updates, 4 epochs, ~4.97 h" và các bộ đánh giá (MASSIVE 51 ngôn ngữ, XNLI 15 ngôn ngữ, typed-decisions, SST-5). README PyPI (METADATA, bảng "English tasks") ghi AG News và BoolQ "in training mix", còn DAIR Emotion, prompt-injections, SST-5 "held out". Vì vậy mix huấn luyện có dữ liệu công khai; MASSIVE và XNLI KHÔNG được ghi là thuộc mix hay held-out => không rõ.
- laya-vi: "The training data is Vietnamese and fully synthetic." 10 miền nghiệp vụ (email CSKH, đối soát hóa đơn, sự cố lưới điện, an toàn lao động, đấu thầu, tuyển dụng, đánh giá sản phẩm, kiểm duyệt bình luận, chat nhóm dự án, hợp đồng thầu phụ); khoảng 1/4 tài liệu không dấu; nhãn từ LLM teacher gán 3 lần rồi lấy trung bình; "English replay makes up 15% of training" (6000 mục; nguồn replay: không rõ); `train_rows` 13.727, `calib_items` 4.505. MASSIVE vi-VN chỉ xuất hiện là tập kiểm `test_public` (1000 mục, nhãn người) trong bảng kết quả, không được nêu là dữ liệu huấn luyện; nhưng mix huấn luyện của checkpoint gốc laya-multilingual không biết nên không loại trừ được.
- Belebele, SIB-200, FLORES, COPA, XCOPA, ViFactCheck, ViWikiFC (và NLI/fact-check tiếng Việt khác): không rõ cho cả hai model.

### 2i. Tham số và base

Cả hai 321.908.998 tham số (322M), file `model.safetensors` 643.835.514 byte, encoder `jhu-clsp/mmBERT-base` (`rl_agent_config.json["encoder"]`; 22 layer, hidden 768, vocab 256000), head 2 layer + scorer + act head. laya-vi: "starts from the `laya-multilingual` checkpoint" (README), `train_log.json` ghi `base: convaiinnovations/laya (multilingual)`, nhưng front-matter ghi `base_model: convaiinnovations/laya` (mâu thuẫn nhẹ giữa các nguồn; đáng tin là multilingual theo văn bản và log huấn luyện). `eval_results.json` của laya-vi có `gates_passed: false` (test_unseen không đạt cổng accuracy/brier).

---------------------------------------------------------------------------------------------------

## 3. mstrasser/Jeff-Qwen3.5-2B

Ghi chú gốc: code chạy model nằm ở `https://github.com/firelex/jeff` (README), KHÔNG có trong HF repo và không có local, nên nhiều mục dưới là "không rõ".

3a. Nạp ở revision đã pin: README không đưa API Python. Cách duy nhất được nêu: `hf download mstrasser/Jeff-Qwen3.5-2B --local-dir Jeff-Qwen3.5-2B` rồi `JEFF_CHECKPOINT=Jeff-Qwen3.5-2B ... jeff-serve` (README dòng 184-185); README dòng 94 nhắc `--revision v1.0` (cờ chuẩn của `hf download`). Vậy: tải snapshot ở sha `568a137a...` về một thư mục (HF `revision=`), rồi trỏ `JEFF_CHECKPOINT` vào thư mục. Backbone chỉ nạp bằng transformers được như `Qwen3_5Model`, nhưng head `readout.safetensors` + prompt không có tài liệu ngoài code của repo GitHub.

3b. Input: README dòng 99-111 cho request kiểu Jev (card: "Jeff uses the same request format as Jev"):
```json
{"model": "jeff-latest", "state": {"voice_transcript": "open the engagment leter", "current_screen": "Deal overview"},
 "questions": {"intent": {"type": "choice", "instructions": "Which of these does the user want?",
                           "criteria": {"1": "Engagement letter", "2": "Inbox", "3": "Deal settings"}}}}
```
Loại câu hỏi: `choice` (tối đa 254 option, `decision_config.json["max_options"] = 254`), `noul` ("yes/no, returned as a probability"), `score` ("a point on a scale you describe"). Tên trường chính xác của noul/score: README không nêu (không rõ; giả định giống Jev chỉ là giả định). Nhiều câu hỏi độc lập trong một request được trả lời "together" (README dòng 115).

3c. Nhãn trần vs khóa+mô tả: README chỉ dẫn dạng khóa ngắn + mô tả ("Use short option keys and descriptive text"): `{"1": "Engagement letter"}`. Dạng nhãn trần: không rõ.

3d. Trả về: README dòng 113 "a probability per option" kèm option được chọn và confidence, ví dụ `{"1": 0.94, "2": 0.03, "3": 0.03}`. Tổng bằng 1, thứ tự, làm tròn: không rõ (ví dụ có thể chỉ minh họa). Nhiệt độ: một scalar duy nhất `decision_config.json["temperature"] = 1.1382043387839655` ("one scalar temperature fitted for calibration"); gói code đọc từ đâu và có rơi về T=1 khi thiếu: không rõ. Xác suất chưa làm tròn: không rõ.

3e. Độ dài: `max_position_embeddings 262144` (`config.json`), README có đánh giá tài liệu dài (ContractNLI, CUAD, MAUD, ConditionalQA) nhưng KHÔNG nêu giới hạn token mặc định, cắt có im lặng hay không, hay cách nâng: không rõ. Giới hạn option: 254.

3f. Batch-1/thiết bị: README bảng "Speed and size": 24 ms/quyết định (RTX PRO 6000), CPU 32 luồng 708 ms, MLX trên Apple silicon (`JEFF_BACKEND=mlx`). `uv sync ... --extra cuda` bật "fast kernels" trên NVIDIA (muốn eager thuần thì không thêm extra này, nhưng không kiểm chứng được). dtype `bfloat16` trong `config.json`. Có compile/CUDA graph/patch kernel hay không: không rõ.

3g. Ví dụ README: input là JSON ở 3b; output nêu rõ: `{"1": 0.94, "2": 0.03, "3": 0.03}` (README dòng 113). Chú ý: đây là dạng minh họa của "The answer is a probability per option".

3h. Dữ liệu huấn luyện (README "Training" + "What changed"): v1.2 train 284.747 câu hỏi, 1 epoch, full-weight SFT. "Training data also adds the training splits of MASSIVE and CLINC150" (v1.1, v1.2 giữ "the kinds of sources are those of v1.1"), kèm WinoGrande (40k), RAGTruth (15k), ContractNLI, CUAD, ConditionalQA, 32k câu hỏi long-list (code + train split MASSIVE/CLINC150), ~50k câu tổng hợp bởi teacher cục bộ Qwen3.8-Flash-Next, "public datasets converted to decisions (entailment, QA, sentiment, safety, fact verification and more)". README nói "results on those two data sets are no longer zero-shot" => đánh giá trên MASSIVE KHÔNG còn zero-shot với Jeff. Locale MASSIVE nào (có vi-VN không): không rõ. Danh sách đầy đủ ở `docs/data-sources.md` trên GitHub (không có local). Card ghi "English and text only". Belebele, SIB-200/FLORES, COPA/XCOPA, ViFactCheck, ViWikiFC: không rõ (không được nêu; các nhóm "QA/entailment/fact verification" quá chung để kết luận).

3i. Tham số/base: `safetensors_params` 2.213.241.664 (README: "2B"; weight 16-bit 4,2 GB); base `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc` (`decision_config.json`, khớp pin của Qwen3.5-2B); full fine-tune, lr 1e-5 (README "Training").

---------------------------------------------------------------------------------------------------

## 4. Điểm có thể chặn hoặc làm lệch adapter

1. Số option: Quyet tối đa 10 (QuestionError, không cắt). Laya không chặn nhưng thực tế < ~20 option vì ngân sách head 256 token (mỗi option còn `(256-16)//k` token; có thể nâng `head_max_len` theo từng lời gọi). Jeff tối đa 254.
2. Jeff: prompt template và áp dụng `readout.safetensors` không có local; README yêu cầu `git clone` + `uv sync` (mạng, code chưa review). Không thể viết adapter offline đã review nếu không vendor repo `firelex/jeff` sau khi review, hoặc tự cài lại phần readout (rủi ro sai).
3. Quyet `raw_probabilities` đã hiệu chuẩn nhiệt độ dù README bảo "uncalibrated". Không có cách lấy T=1 ngoài `m.temps = {}`.
4. Hiệu chuẩn thiếu: Quyet Tiny/Small score 2 hoặc 6-10 mức -> T=1.0; laya-multilingual toàn bộ T=1.0 ("uncalibrated"); laya-vi bucket choice 2 option/>= 11 và score ngoài 3-5 dùng scalar theo loại.
5. Rendering khác nhau: Quyet encoder và Laya đưa KHÓA vào text ("A: ..."), Quyet-Medium ẩn khóa. Với khóa trung tính + mô tả chỉ Quyet-Medium cho cùng kết quả với nhãn trần; ở encoder/Laya hai cách cho input khác nhau.
6. Cắt: Quyet có `strict=True` và `warnings`; Laya chỉ có `usage["truncated"]`; head Laya có thể cắt đề bài/option im lặng. Adapter cần tự kiểm độ dài token và ghi lại.
7. Laya `_fix_tokenizer_config` có thể sửa file trong snapshot; Laya CUDA mặc định autocast bf16; muốn fp32 phải `agent.amp_enabled = False`.
8. Quyet encoder ném `QuestionError` cho cả lời gọi nếu text sinh token đặc biệt (`inputs.py:45-46`), và `QuestionError` nếu câu hỏi/option vượt 8192 token. Bắt lỗi theo từng item.
9. Medium cần transformers >= 5.17 và truyền `enable_thinking=False` (dùng `quyet.llm.prompt.build_prompt`, đừng tự dựng).
10. Không có số kỳ vọng cho Quyet/laya-multilingual (chỉ nhãn `billing`); laya-vi và Jeff có số làm tròn nên dùng làm sanity check có dung sai.

## 5. Công thức một dòng: xác suất chưa làm tròn cho câu hỏi k option (state + question)

- Quyet (Tiny/Small/Medium): `p = quyet.load(SNAP).raw_probabilities([{"state": s, "questions": {"q": {"type": "choice", "instructions": instr, "criteria": {"A": t1, "B": t2, ...}}}}], strict=True)[0]["q"]` -> list k số float, đúng thứ tự option, ĐÃ chia nhiệt độ phát hành (muốn T=1 gán `m.temps = {}` trước).
- laya-multilingual / laya-vi: `agent = laya.load(SNAP_DIR); items = agent._encode_state(s, ["q"], {"q": agent._to_internal(qdef)}); z, _ = agent._forward(collate_items([items], agent.tok.pad_token_id)); p = softmax(z[0,:k] / agent.temperature_by_options.get(temp_bucket(qt,k), agent.temperature[qt]))` (đầy đủ ở mục 2d; hàm riêng tư; bọc `torch.no_grad()`).
- Jeff-Qwen3.5-2B: không có công thức từ tài liệu local (cần code của repo `firelex/jeff`, chưa có).
