"""Chuẩn bị CPU cho results/BRIEF_2b_bac1_train.md (không chạy model, không GPU):
  1. kiểm sha train_2b_v3.jsonl (ea604a01…) và val_v0;
  2. trần state 2.048: đếm token state (tokenizer Qwen3-1.7B) cho MỌI dòng của train_2b_v3 (không tin trường n_state_tokens), báo max,
     số dòng > 2.048 (phải 0), và độ dài prompt đầy đủ lớn nhất (common.render_prompt) của 64 dòng có state dài nhất;
  3. đổi val_kc_v2.jsonl sang dạng chấm của eval_base (target là chỉ số, giữ gold_label) → results/train_2b/eval/_rows/val_kc_v2_eval.jsonl.
Ra results/train_2b/train/prep_bac1.json. Dùng: bash setup/run.sh -I scripts/train_2b/prep_2b_bac1.py"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import CONFIG, load_json, read_jsonl, render_prompt, write_jsonl  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
TR = ROOT / "results/train_2b/train_2b_v3.jsonl"
assert sha(TR).startswith("ea604a01"), sha(TR)
rev = json.load(open(ROOT / "config/models.json", encoding="utf-8"))["Qwen/Qwen3-1.7B"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=rev)
rows = read_jsonl(TR)
lens = [len(x) for x in tok([r["state"] or "" for r in rows], add_special_tokens=False)["input_ids"]]
frames = load_json(CONFIG / "prompt_frames.json")
top = sorted(range(len(rows)), key=lambda i: -lens[i])[:64]
plen = {rows[i]["id"]: len(tok.encode(render_prompt(rows[i], tok, frames), add_special_tokens=False)) for i in top}
out = ROOT / "results/train_2b/eval/_rows"; out.mkdir(parents=True, exist_ok=True)
val = read_jsonl(ROOT / "results/train_2b/val_kc_v2.jsonl")
ev = [{"id": r["id"], "pair_id": r["id"], "source": r["source"], "lang": "vi", "kind": r["kind"], "variant": "nodiac" if "nodiac" in r["flags"] else "orig",
       "state": r["state"], "question": r["question"], "options": r["options"], "target": r["target_argmax"], "gold_label": r["gold_label"],
       "kieu": r["kieu"], "n_tokens": None} for r in val]
write_jsonl(out / "val_kc_v2_eval.jsonl", ev)
res = {"train_file": str(TR.relative_to(ROOT)), "train_sha256": sha(TR), "val_v0_sha256": sha(ROOT / "data/val_v0.jsonl"), "so_dong": len(rows),
       "state_token_max": max(lens), "so_dong_state_gt_2048": sum(x > 2048 for x in lens),
       "prompt_token_max_64_dong_dai_nhat": max(plen.values()), "val_kc_v2_eval": {"file": str((out / "val_kc_v2_eval.jsonl").relative_to(ROOT)),
                                                                                 "sha256": sha(out / "val_kc_v2_eval.jsonl"), "so_dong": len(ev)}}
assert res["so_dong_state_gt_2048"] == 0
(ROOT / "results/train_2b/train").mkdir(parents=True, exist_ok=True)
(ROOT / "results/train_2b/train/prep_bac1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
