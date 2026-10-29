"""Chẩn đoán phụ: bản dev_v1 tiếng Việt chỉ bỏ dấu ở state (câu hỏi và lựa chọn giữ dấu), dùng common.strip_diacritics.
Nguồn: mọi dòng tiếng Việt có dấu (variant orig) của dev_v1. id thêm hậu tố ":stateonly"; n_tokens = null.
Ra: results/ft_strip/diag/dev_v1_vi_state_only_nodiac.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, apply_strip, read_jsonl, sha256_file, write_jsonl  # noqa: E402

rows = read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")
out = []
for r in rows:
    if r["lang"] == "vi" and r["variant"] == "orig":
        s = apply_strip(r, "state_only")
        out.append({**s, "id": r["id"] + ":stateonly", "variant": "state_only_nodiac", "n_tokens": None})
p = RESULTS / "ft_strip" / "diag" / "dev_v1_vi_state_only_nodiac.jsonl"
write_jsonl(p, out)
print(p, len(out), sha256_file(p))
