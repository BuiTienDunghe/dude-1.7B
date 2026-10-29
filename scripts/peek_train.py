"""In 5 dòng train_v0 (mỗi trường cắt 200 ký tự) và một prompt đã render, để kiểm tra bằng mắt."""
from common import CONFIG, DATA, load_json, model_revisions, read_jsonl, render_prompt
from transformers import AutoTokenizer

rows = read_jsonl(DATA / "train_v0.jsonl")
pick = [next(r for r in rows if r["dataset"] == "pholaya/vinli" and r["kind"] == "choice"),
        next(r for r in rows if r["dataset"] == "pholaya/visfd" and r["kind"] == "noul"),
        next(r for r in rows if r["dataset"] == "tasksource" and r["kind"] == "noul"),
        next(r for r in rows if r["dataset"] == "tasksource" and r["kind"] == "score"),
        next(r for r in rows if r["text_lang"] == "multi")]
for r in pick:
    print({k: str(v)[:200] for k, v in r.items() if k in ("source", "lang", "kind", "state", "question", "options", "target")})
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
print(render_prompt(pick[0], tok, load_json(CONFIG / "prompt_frames.json"))[:600])
