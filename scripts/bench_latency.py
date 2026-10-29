"""Đo độ trễ batch 1 của hai model đã fine-tune (seed 1, LoRA merge vào bf16, đúng đường chấm của eval_base.py)
theo độ dài prompt ~128, ~512, ~2.048 token.

- Prompt dựng bằng common.render_prompt từ văn bản thật trong dev_v0, cùng văn bản cho cả hai model:
  ~128: câu dev_v0 tiếng Việt có prompt gần 128 token nhất (dùng nguyên câu);
  ~512 và ~2.048: dev_v0 không có câu nào dài tới mức này, nên state là các đoạn Belebele tiếng Việt khác nhau
  của dev_v0 nối lại rồi cắt theo token cho sát mức đích; câu hỏi và lựa chọn lấy từ câu Belebele đầu tiên.
- Mỗi model được nạp riêng (để VRAM đỉnh chỉ tính một model), lặp hai vòng 0.6B → 1.7B → 0.6B → 1.7B để thấy độ trôi.
- Mỗi mức: 10 lượt warmup, 60 lượt đo; torch.cuda.synchronize quanh mỗi lượt.
Ra: results/ft/latency_ft.json
"""
import gc
import json
import statistics
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, DATA, RESULTS, load_json, model_revisions, read_jsonl, render_prompt

TARGETS = [128, 512, 2048]
WARMUP, REPS, ROUNDS = 10, 60, 2
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
ADAPTER = {m: RESULTS / "ft/train" / f"{m.split('/')[1]}-s1" / "adapter" for m in MODELS}

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
frames = load_json(CONFIG / "prompt_frames.json")
rows = read_jsonl(DATA / "dev_v0.jsonl")
tok = AutoTokenizer.from_pretrained(MODELS[0], revision=model_revisions()[MODELS[0]])   # hai model chung tokenizer
n_tok = lambda r: len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))

# ---- dựng prompt cho từng mức
prompts = {}
vi = [r for r in rows if r["lang"] == "vi"]
r128 = min(vi, key=lambda r: (abs(r["n_tokens"] - 128), r["id"]))
prompts[128] = {"row_id": r128["id"], "how": "nguyên câu dev_v0", "row": r128}
bele = sorted({r["cluster"]: r for r in vi if r["source"] == "belebele"}.values(), key=lambda r: r["id"])
long_text = "\n\n".join(r["state"] for r in bele)
long_ids = tok.encode(long_text, add_special_tokens=False)
base_row = bele[0]
for tgt in (512, 2048):
    lo, hi = 1, len(long_ids)
    while lo < hi:   # số token state lớn nhất sao cho tổng prompt <= tgt
        mid = (lo + hi + 1) // 2
        if n_tok({**base_row, "state": tok.decode(long_ids[:mid])}) <= tgt:
            lo = mid
        else:
            hi = mid - 1
    row = {**base_row, "state": tok.decode(long_ids[:lo]), "id": f"bench{tgt}"}
    prompts[tgt] = {"row_id": f"nối {len(bele)} đoạn Belebele vi của dev_v0, cắt còn {lo} token state; câu hỏi/lựa chọn của {base_row['id']}",
                    "how": "nối + cắt", "row": row}
for tgt, p in prompts.items():
    p["ids"] = tok.encode(render_prompt(p["row"], tok, frames), add_special_tokens=False)
    p["n_tokens"] = len(p["ids"])
    print(tgt, "->", p["n_tokens"], "tokens |", p["how"])


@torch.inference_mode()
def once(model, x):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model(input_ids=x, logits_to_keep=1)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0)


out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "bfloat16", "attn": "sdpa",
       "warmup": WARMUP, "reps": REPS, "rounds": ROUNDS,
       "prompts": {str(t): {"n_tokens": p["n_tokens"], "source": p["row_id"]} for t, p in prompts.items()},
       "note": "2.048 token vượt mức state <= 1.024 token lúc train; đây chỉ là số đo tốc độ.", "results": []}
for rnd in range(1, ROUNDS + 1):
    for m in MODELS:
        rev = model_revisions()[m]
        model = AutoModelForCausalLM.from_pretrained(m, revision=rev, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, str(ADAPTER[m])).merge_and_unload().eval().to(torch.bfloat16)
        for tgt in TARGETS:
            x = torch.tensor([prompts[tgt]["ids"]], device="cuda")
            for _ in range(WARMUP):
                once(model, x)
            torch.cuda.reset_peak_memory_stats()
            lat = sorted(once(model, x) for _ in range(REPS))
            rec = {"round": rnd, "model": m, "adapter": str(ADAPTER[m]), "target": tgt, "n_tokens": prompts[tgt]["n_tokens"],
                   "p50_ms": statistics.median(lat), "p90_ms": lat[int(0.9 * len(lat))], "min_ms": lat[0], "max_ms": lat[-1],
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
            out["results"].append(rec)
            print(json.dumps(rec))
        del model; gc.collect(); torch.cuda.empty_cache()

# tóm tắt: p50/p90 gộp trung bình hai vòng
summary = {}
for m in MODELS:
    for tgt in TARGETS:
        rs = [r for r in out["results"] if r["model"] == m and r["target"] == tgt]
        summary[f"{m.split('/')[1]}@{tgt}"] = {"p50_ms_by_round": [round(r["p50_ms"], 2) for r in rs],
                                               "p90_ms_by_round": [round(r["p90_ms"], 2) for r in rs],
                                               "peak_vram_gb": round(max(r["peak_vram_gb"] for r in rs), 2)}
out["summary"] = summary
(RESULTS / "ft" / "latency_ft.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
