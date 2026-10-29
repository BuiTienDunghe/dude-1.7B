"""Chuẩn bị chấm Qwen/Qwen3.5-2B bằng eval_base.py mà không sửa code chấm:
- ghim revision vào config/models.json (thêm một khóa; các khóa cũ giữ nguyên);
- tạo bản sao dev_v0 giống hệt nhưng n_tokens = null (n_tokens của dev_v0 đếm bằng tokenizer Qwen3; eval_base.py bỏ qua
  phép so n_tokens khi trường này rỗng). Nội dung câu, thứ tự lựa chọn và target không đổi.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, MODELS_FILE, RESULTS, read_jsonl, sha256_file, write_jsonl

pins = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
pins["Qwen/Qwen3.5-2B"] = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODELS_FILE.write_text(json.dumps(pins, indent=2), encoding="utf-8")
rows = read_jsonl(DATA / "dev_v0.jsonl")
out = RESULTS / "baselines" / "dev_v0_ntok_null.jsonl"
write_jsonl(out, [{**r, "n_tokens": None} for r in rows])
back = read_jsonl(out)
assert [{k: v for k, v in r.items() if k != "n_tokens"} for r in rows] == [{k: v for k, v in r.items() if k != "n_tokens"} for r in back]
print("models.json:", pins)
print("dev copy:", out, sha256_file(out), len(back), "rows")
