"""Bước E: chuẩn bị dữ liệu chấm model ngoài trên DEV 1.2 (không sửa code chấm của dự án, không đụng test).
- kiểm sha dev_v1_2.jsonl;
- dev_v1_2_ntok_null.jsonl: bản sao giống hệt, n_tokens = null (n_tokens đếm bằng tokenizer Qwen3; eval_base bỏ qua phép so khi rỗng);
- dev_v1_2_choice_reversed.jsonl: mọi câu trừ kind = noul (choice, reading_mc, topic, intent, causal), lựa chọn đảo ngược, target = k-1-target, n_tokens = null (cho adapter không có cờ đảo).
  Câu noul (Có/Không) không đưa vào: các model ngoài chấm noul bằng đường yes/no gốc, đầu vào không phụ thuộc thứ tự lựa chọn."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/bench_v1_2/dev_v1_2.jsonl"
OUT = ROOT / "results/baselines_v1_2"
SHA = "ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b"
assert hashlib.sha256(DEV.read_bytes()).hexdigest() == SHA, "dev 1.2 khác sha đã chốt"
rows = [json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip()]
OUT.mkdir(parents=True, exist_ok=True)
w = lambda p, rs: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8", newline="\n")
if not (OUT / "dev_v1_2_ntok_null.jsonl").exists():
    w(OUT / "dev_v1_2_ntok_null.jsonl", [{**r, "n_tokens": None} for r in rows])
rev = [{**r, "options": r["options"][::-1], "target": len(r["options"]) - 1 - r["target"], "n_tokens": None, "reversed": True} for r in rows if r["kind"] != "noul"]
tmp = OUT / "dev_v1_2_choice_reversed.jsonl.tmp"
w(tmp, rev)
tmp.replace(OUT / "dev_v1_2_choice_reversed.jsonl")       # thay nguyên khối
print(json.dumps({"dev_sha": SHA, "rows": len(rows), "choice_reversed": len(rev), "kinds": sorted({r["kind"] for r in rows})}))
