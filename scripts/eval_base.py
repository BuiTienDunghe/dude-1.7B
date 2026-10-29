"""Chấm model (gốc hoặc gốc + LoRA) trên dev_v0 bằng logit token kế tiếp (không generate).

Dùng: eval_base.py <model_id> <out_dir> [--adapter DIR] [--reverse] [--rows JSONL] [--name NAME]
  --adapter  thư mục LoRA (peft); được merge vào trọng số bf16 rồi chấm bằng đúng đường chấm của model gốc
  --reverse  đảo ngược thứ tự lựa chọn của mỗi câu (target = k-1-target) để đo thiên vị vị trí
  --rows     chấm tập khác dev_v0 (ví dụ val nội bộ, để fit temperature)
  --name     tên thư mục con trong out_dir (mặc định: tên model)
  --revision ghim revision trực tiếp (G0, 28/10; model chưa có trong config/models.json)
  --frame en|vi  dùng một khung prompt cho mọi dòng (G0c: khung en cho dòng VI; nội dung, câu hỏi, lựa chọn giữ nguyên)
  --chat_template_from repo@rev  mượn chat template khi tokenizer không có (bản Base), ghi vào timing.json
Ra: <out_dir>/<name>/preds.jsonl (xác suất từng lựa chọn mỗi câu) và timing.json.
Không truyền tuỳ chọn nào thì hành vi giống hệt bản dùng ở bước dev_v0.
Lượt chấm chính chạy batch 1 (không padding) để kết quả không phụ thuộc cách gom batch.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, DATA, LETTERS, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("model_id")
ap.add_argument("out_dir", type=Path)
ap.add_argument("--adapter", type=Path)
ap.add_argument("--reverse", action="store_true")
ap.add_argument("--rows", type=Path, default=DATA / "dev_v0.jsonl")
ap.add_argument("--name")
ap.add_argument("--revision")            # G0 (28/10): ghim revision khi model chưa có trong config/models.json (không sửa config)
ap.add_argument("--frame", choices=["vi", "en"])          # G0c: dùng một khung cho mọi dòng (ví dụ khung en cho dòng VI)
ap.add_argument("--chat_template_from")  # "repo@revision": mượn chat template khi tokenizer (bản Base) không có
args = ap.parse_args()
model_id = args.model_id
rev = args.revision or model_revisions()[model_id]
slug = args.name or model_id.split("/")[-1]
out = args.out_dir / slug
out.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

frames = load_json(CONFIG / "prompt_frames.json")
rows = read_jsonl(args.rows)
if args.reverse:
    rows = [{**r, "options": r["options"][::-1], "target": len(r["options"]) - 1 - r["target"]} for r in rows]
if args.frame:
    frames = {"vi": frames[args.frame], "en": frames[args.frame]}
tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
template_src = None
if args.chat_template_from and not tok.chat_template:
    src_id, src_rev = args.chat_template_from.split("@")
    tok.chat_template = AutoTokenizer.from_pretrained(src_id, revision=src_rev).chat_template
    template_src = args.chat_template_from
model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa").eval()
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload().eval()
    model = model.to(torch.bfloat16)
assert next(model.parameters()).dtype == torch.bfloat16
letter_ids = [tok.convert_tokens_to_ids(L) for L in LETTERS]
assert all(len(tok.encode(L, add_special_tokens=False)) == 1 for L in LETTERS)

prompts = [render_prompt(r, tok, frames) for r in rows]
enc = [tok.encode(p, add_special_tokens=False) for p in prompts]
for r, p, ids in zip(rows, prompts, enc):
    if not args.reverse and r.get("n_tokens") is not None:
        assert len(ids) == r["n_tokens"], (r["id"], len(ids), r["n_tokens"])
    for L in LETTERS[:len(r["options"])]:  # ranh giới token: thêm chữ cái không làm đổi token trước đó
        assert tok.encode(p + L, add_special_tokens=False)[:-1] == ids
assert max(len(e) for e in enc) < model.config.max_position_embeddings


@torch.inference_mode()
def last_logits(batch_ids, attention_mask=None):
    return model(input_ids=batch_ids, attention_mask=attention_mask, logits_to_keep=1).logits[:, -1, :].float()


def score(logits_row, k):
    full = torch.softmax(logits_row, -1)
    sel = logits_row[letter_ids[:k]]
    probs = torch.softmax(sel, -1)
    top_id = int(logits_row.argmax())
    return probs.tolist(), float(full[letter_ids[:k]].sum()), tok.convert_ids_to_tokens(top_id)


# ---- warmup
for ids in enc[:10]:
    last_logits(torch.tensor([ids], device="cuda"))
torch.cuda.synchronize()

# ---- lượt chấm chính: batch 1, đo độ trễ từng câu
preds, lat = [], []
for r, ids in zip(rows, enc):
    k = len(r["options"])
    x = torch.tensor([ids], device="cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lg = last_logits(x)[0]
    torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
    probs, mass, top = score(lg, k)
    pred = max(range(k), key=lambda i: probs[i])
    p = {"id": r["id"], "pair_id": r.get("pair_id"), "source": r["source"], "lang": r["lang"],
         "model": model_id, "revision": rev, "k": k, "target": r["target"],
         "probs": probs, "pred": pred, "correct": pred == r["target"],
         "letter_mass": mass, "top_token": top}
    if args.adapter or args.reverse or args.rows != DATA / "dev_v0.jsonl":
        # chỉ thêm trường khi dùng tuỳ chọn, để lượt mặc định giữ nguyên từng byte như bước dev_v0
        p.update({"adapter": str(args.adapter) if args.adapter else None, "reversed": args.reverse,
                  "logits": lg[letter_ids[:k]].tolist()})
    preds.append(p)
write_jsonl(out / "preds.jsonl", preds)

# ---- thông lượng khi gom batch (left padding, sắp theo độ dài); so argmax với lượt batch 1
tok.padding_side = "left"
BS = 16
order = sorted(range(len(rows)), key=lambda i: len(enc[i]))
batched_pred = {}
torch.cuda.synchronize(); t0 = time.perf_counter()
for s in range(0, len(order), BS):
    idx = order[s:s + BS]
    b = tok.pad({"input_ids": [enc[i] for i in idx]}, return_tensors="pt").to("cuda")
    lg = last_logits(b["input_ids"], b["attention_mask"])
    for j, i in enumerate(idx):
        k = len(rows[i]["options"])
        batched_pred[i] = int(lg[j, letter_ids[:k]].argmax())
torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
agree = sum(batched_pred[i] == preds[i]["pred"] for i in range(len(rows))) / len(rows)

timing = {
    "model": model_id, "revision": rev, "dtype": "bfloat16", "attn_implementation": model.config._attn_implementation,
    "adapter": str(args.adapter) if args.adapter else None, "reversed": args.reverse, "rows_file": str(args.rows),
    "rows_sha256": sha256_file(args.rows),
    "batch1_median_latency_ms": 1000 * statistics.median(lat),
    "batch1_p90_latency_ms": 1000 * sorted(lat)[int(0.9 * len(lat))],
    "batch1_rows": len(lat), "warmup_rows": 10,
    "batched_batch_size": BS, "batched_rows_per_s": len(rows) / elapsed,
    "batched_vs_batch1_argmax_agreement": agree,
    "median_prompt_tokens": statistics.median(len(e) for e in enc),
    "hardware": {"gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                 "driver": "591.86 (nvidia-smi)", "os": platform.platform(), "env": "WSL2 Ubuntu 26.04"},
    "software": {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                 "python": platform.python_version()},
    "dev_v0_sha256": sha256_file(DATA / "dev_v0.jsonl"),
    "prompt_frames_sha256": sha256_file(CONFIG / "prompt_frames.json"),
    **({"frame_override": args.frame} if args.frame else {}),
    **({"chat_template_from": template_src} if args.chat_template_from else {}),
}
(out / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
print(json.dumps(timing, indent=2))
print("accuracy (all rows):", sum(p["correct"] for p in preds) / len(preds))
