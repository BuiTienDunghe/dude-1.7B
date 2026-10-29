"""So sánh hai lượt chạy: preds.jsonl và metrics.json phải giống hệt từng byte (không tính timing)."""
import sys
from pathlib import Path

from common import sha256_file

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ok = True
for rel in ["Qwen3-0.6B/preds.jsonl", "Qwen3-1.7B/preds.jsonl", "metrics.json", "report.md"]:
    ha, hb = sha256_file(a / rel), sha256_file(b / rel)
    same = ha == hb
    ok &= same
    print(f"{'SAME' if same else 'DIFF'}  {rel}  {ha[:16]}  {hb[:16]}")
print("IDENTICAL" if ok else "NOT IDENTICAL")
